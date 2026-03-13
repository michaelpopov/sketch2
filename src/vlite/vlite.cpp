// Implements the SQLite virtual table that exposes vector search over datasets.

#include "vlite.h"

#include "core/compute/compute_cos.h"
#include "core/compute/scanner.h"
#include "core/storage/dataset.h"
#include "core/utils/shared_consts.h"
#include "core/utils/string_utils.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

SQLITE_EXTENSION_INIT1

namespace {

enum VliteColumn {
    kColumnQuery = 0,
    kColumnK = 1,
    kColumnId = 2,
    kColumnDistance = 3,
};

enum VliteConstraintBit {
    kConstraintQuery = 1 << 0,
    kConstraintK = 1 << 1,
    kConstraintLimit = 1 << 2,
    kConstraintOffset = 1 << 3,
};

// Removes the outer quoting syntax SQLite may preserve in module arguments so
// the dataset path can be passed to Dataset::init verbatim.
std::string dequote_sqlite_arg(const char* text) {
    if (text == nullptr) {
        return "";
    }

    std::string value(text);
    if (value.size() >= 2) {
        const char first = value.front();
        const char last = value.back();
        const bool quoted =
            (first == '\'' && last == '\'') ||
            (first == '"' && last == '"') ||
            (first == '[' && last == ']');
        if (quoted) {
            value = value.substr(1, value.size() - 2);
        }
    }

    return value;
}

void set_vtab_error(sqlite3_vtab* tab, const std::string& message) {
    if (tab == nullptr) {
        return;
    }
    sqlite3_free(tab->zErrMsg);
    tab->zErrMsg = sqlite3_mprintf("%s", message.c_str());
}

void set_err_msg(char** err_msg, const std::string& message) {
    if (err_msg == nullptr) {
        return;
    }
    sqlite3_free(*err_msg);
    *err_msg = sqlite3_mprintf("%s", message.c_str());
}

template <typename Func>
int run_errmsg_callback(char** err_msg, Func func) {
    try {
        return func();
    } catch (const std::bad_alloc&) {
        set_err_msg(err_msg, "vlite: out of memory");
        return SQLITE_NOMEM;
    } catch (const std::exception& ex) {
        set_err_msg(err_msg, ex.what());
        return SQLITE_ERROR;
    } catch (...) {
        set_err_msg(err_msg, "vlite: unexpected error");
        return SQLITE_ERROR;
    }
}

template <typename Func>
int run_vtab_callback(sqlite3_vtab* tab, Func func) {
    try {
        return func();
    } catch (const std::bad_alloc&) {
        set_vtab_error(tab, "vlite: out of memory");
        return SQLITE_NOMEM;
    } catch (const std::exception& ex) {
        set_vtab_error(tab, ex.what());
        return SQLITE_ERROR;
    } catch (...) {
        set_vtab_error(tab, "vlite: unexpected error");
        return SQLITE_ERROR;
    }
}

template <typename Func>
int run_cursor_callback(sqlite3_vtab_cursor* cursor, Func func) {
    return run_vtab_callback(cursor != nullptr ? cursor->pVtab : nullptr, func);
}

// Wraps xColumn-style callbacks so C++ exceptions are converted into both a
// SQLite result error and the virtual table's zErrMsg.
template <typename Func>
int run_column_callback(sqlite3_vtab_cursor* cursor, sqlite3_context* context, Func func) {
    try {
        return func();
    } catch (const std::bad_alloc&) {
        if (context != nullptr) {
            sqlite3_result_error_nomem(context);
        }
        set_vtab_error(cursor != nullptr ? cursor->pVtab : nullptr, "vlite: out of memory");
        return SQLITE_NOMEM;
    } catch (const std::exception& ex) {
        if (context != nullptr) {
            sqlite3_result_error(context, ex.what(), -1);
        }
        set_vtab_error(cursor != nullptr ? cursor->pVtab : nullptr, ex.what());
        return SQLITE_ERROR;
    } catch (...) {
        constexpr const char* kUnexpectedError = "vlite: unexpected error";
        if (context != nullptr) {
            sqlite3_result_error(context, kUnexpectedError, -1);
        }
        set_vtab_error(cursor != nullptr ? cursor->pVtab : nullptr, kUnexpectedError);
        return SQLITE_ERROR;
    }
}

bool is_query_constraint(int op) {
    return op == SQLITE_INDEX_CONSTRAINT_EQ || op == SQLITE_INDEX_CONSTRAINT_MATCH;
}

sqlite3_int64 saturate_negative_to_zero(sqlite3_int64 value) {
    return value > 0 ? value : 0;
}

sqlite3_int64 saturating_add(sqlite3_int64 lhs, sqlite3_int64 rhs) {
    if (lhs > 0 && rhs > 0 && lhs > std::numeric_limits<sqlite3_int64>::max() - rhs) {
        return std::numeric_limits<sqlite3_int64>::max();
    }
    return lhs + rhs;
}

// VliteVTab exists to bind SQLite's virtual-table object to the dataset state
// needed by the extension. It stores the dataset path and the opened Dataset instance.
struct VliteVTab : sqlite3_vtab {
    std::string dataset_path;
    std::unique_ptr<sketch2::Dataset> dataset;
};

// VliteCursor exists to hold one materialized query result set for SQLite. It
// keeps the parsed query buffer, result rows, and iteration state consumed by
// the xFilter/xNext/xColumn callbacks.
struct VliteCursor : sqlite3_vtab_cursor {
    std::vector<sketch2::DistItem> rows;
    std::vector<uint8_t> query_buf;
    std::string query_text;
    sqlite3_int64 k = 0; // Requested/default SQL k, not the internal pushdown-adjusted count.
    size_t index = 0;
    sqlite3_int64 rowid = 1;
};

// Shared xCreate/xConnect path that validates the module arguments, declares
// the schema, opens the backing dataset, and switches it into read-only guest mode.
int vlite_connect_common(sqlite3* db, int argc, const char* const* argv,
    sqlite3_vtab** pp_vtab, char** err_msg) {
    return run_errmsg_callback(err_msg, [&]() -> int {
        if (db == nullptr || pp_vtab == nullptr || argv == nullptr) {
            return SQLITE_ERROR;
        }
        if (argc != 4) {
            set_err_msg(err_msg, "vlite requires exactly one dataset ini path argument");
            return SQLITE_ERROR;
        }

        const int declare_rc = sqlite3_declare_vtab(db, sketch2::kVliteSchema);
        if (declare_rc != SQLITE_OK) {
            return declare_rc;
        }

        auto* vtab = new (std::nothrow) VliteVTab();
        if (vtab == nullptr) {
            return SQLITE_NOMEM;
        }

        vtab->dataset_path = dequote_sqlite_arg(argv[3]);
        if (vtab->dataset_path.empty()) {
            set_err_msg(err_msg, "vlite dataset ini path must not be empty");
            delete vtab;
            return SQLITE_ERROR;
        }

        vtab->dataset = std::make_unique<sketch2::Dataset>();
        const sketch2::Ret ret = vtab->dataset->init(vtab->dataset_path);
        if (ret.code() != 0) {
            set_err_msg(err_msg, ret.message());
            delete vtab;
            return SQLITE_ERROR;
        }
        const sketch2::Ret guest_ret = vtab->dataset->set_guest_mode();
        if (guest_ret.code() != 0) {
            set_err_msg(err_msg, guest_ret.message());
            delete vtab;
            return SQLITE_ERROR;
        }

        *pp_vtab = vtab;
        return SQLITE_OK;
    });
}

int vlite_create(sqlite3* db, void* aux, int argc, const char* const* argv,
    sqlite3_vtab** pp_vtab, char** err_msg) {
    (void)aux;
    return vlite_connect_common(db, argc, argv, pp_vtab, err_msg);
}

int vlite_connect(sqlite3* db, void* aux, int argc, const char* const* argv,
    sqlite3_vtab** pp_vtab, char** err_msg) {
    (void)aux;
    return vlite_connect_common(db, argc, argv, pp_vtab, err_msg);
}

// Advertises which constraints the virtual table can consume and encodes that
// decision in idxNum so xFilter can read query, k, LIMIT, and OFFSET values in order.
int vlite_best_index(sqlite3_vtab* tab, sqlite3_index_info* index_info) {
    return run_vtab_callback(tab, [&]() -> int {
        if (tab == nullptr || index_info == nullptr) {
            return SQLITE_ERROR;
        }

        int query_constraint = -1;
        int k_constraint = -1;
        int limit_constraint = -1;
        int offset_constraint = -1;

        for (int i = 0; i < index_info->nConstraint; ++i) {
            const auto& constraint = index_info->aConstraint[i];
            if (!constraint.usable) {
                continue;
            }
            if (query_constraint < 0 &&
                    constraint.iColumn == kColumnQuery &&
                    is_query_constraint(constraint.op)) {
                query_constraint = i;
            } else if (k_constraint < 0 &&
                    constraint.iColumn == kColumnK &&
                    constraint.op == SQLITE_INDEX_CONSTRAINT_EQ) {
                k_constraint = i;
            } else if (limit_constraint < 0 && constraint.op == SQLITE_INDEX_CONSTRAINT_LIMIT) {
                limit_constraint = i;
            } else if (offset_constraint < 0 && constraint.op == SQLITE_INDEX_CONSTRAINT_OFFSET) {
                offset_constraint = i;
            }
        }

        int idx_num = 0;
        int next_arg = 1;

        if (query_constraint >= 0) {
            index_info->aConstraintUsage[query_constraint].argvIndex = next_arg++;
            index_info->aConstraintUsage[query_constraint].omit = 1;
            idx_num |= kConstraintQuery;
        }
        if (k_constraint >= 0) {
            index_info->aConstraintUsage[k_constraint].argvIndex = next_arg++;
            index_info->aConstraintUsage[k_constraint].omit = 1;
            idx_num |= kConstraintK;
        }
        if (limit_constraint >= 0) {
            index_info->aConstraintUsage[limit_constraint].argvIndex = next_arg++;
            idx_num |= kConstraintLimit;
        }
        if (offset_constraint >= 0) {
            index_info->aConstraintUsage[offset_constraint].argvIndex = next_arg++;
            idx_num |= kConstraintOffset;
        }

        index_info->idxNum = idx_num;
        index_info->estimatedCost = (idx_num & kConstraintQuery) ? 10.0 : 1.0e12;
        index_info->estimatedRows = (idx_num & (kConstraintK | kConstraintLimit)) ? 10 : 1000;
        if (index_info->nOrderBy == 1 &&
                index_info->aOrderBy[0].iColumn == kColumnDistance &&
                index_info->aOrderBy[0].desc == 0) {
            index_info->orderByConsumed = 1;
        }
        return SQLITE_OK;
    });
}

int vlite_disconnect(sqlite3_vtab* tab) {
    delete static_cast<VliteVTab*>(tab);
    return SQLITE_OK;
}

int vlite_destroy(sqlite3_vtab* tab) {
    return vlite_disconnect(tab);
}

int vlite_open(sqlite3_vtab* tab, sqlite3_vtab_cursor** pp_cursor) {
    return run_vtab_callback(tab, [&]() -> int {
        if (tab == nullptr || pp_cursor == nullptr) {
            return SQLITE_ERROR;
        }

        auto* cursor = new (std::nothrow) VliteCursor();
        if (cursor == nullptr) {
            return SQLITE_NOMEM;
        }

        cursor->pVtab = tab;
        *pp_cursor = cursor;
        return SQLITE_OK;
    });
}

int vlite_close(sqlite3_vtab_cursor* cursor) {
    delete static_cast<VliteCursor*>(cursor);
    return SQLITE_OK;
}

// Executes one virtual-table query. It decodes the planner-selected arguments,
// normalizes LIMIT/OFFSET pushdown into an effective k, parses the query vector,
// and materializes the matching rows into the cursor.
int vlite_filter(sqlite3_vtab_cursor* cursor, int idx_num, const char* idx_str,
    int argc, sqlite3_value** argv) {
    return run_cursor_callback(cursor, [&]() -> int {
        (void)idx_str;

        if (cursor == nullptr) {
            return SQLITE_ERROR;
        }

        auto* vlite_cursor = static_cast<VliteCursor*>(cursor);
        auto* vlite_vtab = static_cast<VliteVTab*>(cursor->pVtab);
        vlite_cursor->rows.clear();
        vlite_cursor->query_buf.clear();
        vlite_cursor->query_text.clear();
        vlite_cursor->k = 0;
        vlite_cursor->index = 0;
        vlite_cursor->rowid = 1;

        if ((idx_num & kConstraintQuery) == 0 || argc == 0 || argv == nullptr) {
            set_vtab_error(vlite_vtab, "vlite requires WHERE query = ... or query MATCH ...");
            return SQLITE_ERROR;
        }

        int arg_index = 0;
        const unsigned char* query_text = sqlite3_value_text(argv[arg_index++]);
        if (query_text == nullptr || query_text[0] == '\0') {
            set_vtab_error(vlite_vtab, "vlite query must be a non-empty string");
            return SQLITE_ERROR;
        }
        vlite_cursor->query_text = reinterpret_cast<const char*>(query_text);

        sqlite3_int64 k = 10;
        const bool has_explicit_k = (idx_num & kConstraintK) != 0;
        if ((idx_num & kConstraintK) != 0) {
            if (arg_index >= argc) {
                set_vtab_error(vlite_vtab, "vlite missing k constraint value");
                return SQLITE_ERROR;
            }
            k = sqlite3_value_int64(argv[arg_index++]);
            if (k <= 0) {
                set_vtab_error(vlite_vtab, "vlite k must be > 0");
                return SQLITE_ERROR;
            }
        }
        vlite_cursor->k = k;

        sqlite3_int64 limit = -1;
        if ((idx_num & kConstraintLimit) != 0) {
            if (arg_index >= argc) {
                set_vtab_error(vlite_vtab, "vlite missing LIMIT value");
                return SQLITE_ERROR;
            }
            limit = sqlite3_value_int64(argv[arg_index++]);
        }

        sqlite3_int64 offset = 0;
        if ((idx_num & kConstraintOffset) != 0) {
            if (arg_index >= argc) {
                set_vtab_error(vlite_vtab, "vlite missing OFFSET value");
                return SQLITE_ERROR;
            }
            offset = sqlite3_value_int64(argv[arg_index++]);
        }

        const sqlite3_int64 window =
            (limit >= 0) ? saturating_add(limit, saturate_negative_to_zero(offset)) : -1;
        sqlite3_int64 effective_k = k;
        if (window >= 0) {
            effective_k = has_explicit_k ? std::min(k, window) : window;
        }
        if (effective_k <= 0) {
            return SQLITE_OK;
        }

        if (!vlite_vtab->dataset) {
            set_vtab_error(vlite_vtab, "vlite dataset is not initialized");
            return SQLITE_ERROR;
        }
        sketch2::Dataset& dataset = *vlite_vtab->dataset;
        assert(dataset.dim() >= sketch2::kMinDimension && dataset.dim() <= sketch2::kMaxDimension);
        const uint16_t query_dim = static_cast<uint16_t>(dataset.dim());

        const size_t query_size = sketch2::data_type_size(dataset.type()) * dataset.dim();
        vlite_cursor->query_buf.resize(query_size);
        sketch2::Ret ret = sketch2::parse_vector(vlite_cursor->query_buf.data(),
            vlite_cursor->query_buf.size(), dataset.type(), query_dim,
            vlite_cursor->query_text.c_str());
        if (ret.code() != 0) {
            set_vtab_error(vlite_vtab, ret.message());
            return SQLITE_ERROR;
        }

        sketch2::Scanner scanner;
        ret = scanner.find_items(dataset, static_cast<size_t>(effective_k),
            vlite_cursor->query_buf.data(), vlite_cursor->rows);
        if (ret.code() != 0) {
            set_vtab_error(vlite_vtab, ret.message());
            return SQLITE_ERROR;
        }

        return SQLITE_OK;
    });
}

int vlite_next(sqlite3_vtab_cursor* cursor) {
    return run_cursor_callback(cursor, [&]() -> int {
        if (cursor == nullptr) {
            return SQLITE_ERROR;
        }

        auto* vlite_cursor = static_cast<VliteCursor*>(cursor);
        ++vlite_cursor->index;
        ++vlite_cursor->rowid;
        return SQLITE_OK;
    });
}

int vlite_eof(sqlite3_vtab_cursor* cursor) {
    if (cursor == nullptr) {
        return 1;
    }

    const auto* vlite_cursor = static_cast<VliteCursor*>(cursor);
    return vlite_cursor->index >= vlite_cursor->rows.size() ? 1 : 0;
}

// Returns the requested column for the current result row, including range
// checks when exposing 64-bit vector ids as SQLite INTEGER values.
int vlite_column(sqlite3_vtab_cursor* cursor, sqlite3_context* context, int column) {
    return run_column_callback(cursor, context, [&]() -> int {
        if (cursor == nullptr || context == nullptr) {
            return SQLITE_ERROR;
        }

        auto* vlite_cursor = static_cast<VliteCursor*>(cursor);
        if (vlite_cursor->index >= vlite_cursor->rows.size()) {
            sqlite3_result_null(context);
            return SQLITE_OK;
        }

        const sketch2::DistItem& row = vlite_cursor->rows[vlite_cursor->index];
        switch (column) {
            case kColumnQuery:
                sqlite3_result_text(context, vlite_cursor->query_text.c_str(), -1, SQLITE_TRANSIENT);
                break;
            case kColumnK:
                sqlite3_result_int64(context, vlite_cursor->k);
                break;
            case kColumnId: {
                if (row.id > static_cast<uint64_t>(std::numeric_limits<sqlite3_int64>::max())) {
                    constexpr const char* kIdRangeError = "vlite id exceeds SQLite INTEGER range";
                    set_vtab_error(cursor->pVtab, kIdRangeError);
                    sqlite3_result_error(context, kIdRangeError, -1);
                    return SQLITE_ERROR;
                }
                sqlite3_result_int64(context, static_cast<sqlite3_int64>(row.id));
                break;
            }
            case kColumnDistance:
                sqlite3_result_double(context, row.dist);
                break;
            default:
                sqlite3_result_null(context);
                break;
        }
        return SQLITE_OK;
    });
}

int vlite_rowid(sqlite3_vtab_cursor* cursor, sqlite3_int64* rowid) {
    return run_cursor_callback(cursor, [&]() -> int {
        if (cursor == nullptr || rowid == nullptr) {
            return SQLITE_ERROR;
        }

        *rowid = static_cast<VliteCursor*>(cursor)->rowid;
        return SQLITE_OK;
    });
}

int vlite_update(sqlite3_vtab* tab, int argc, sqlite3_value** argv, sqlite3_int64* rowid) {
    return run_vtab_callback(tab, [&]() -> int {
        (void)tab;
        (void)argc;
        (void)argv;
        (void)rowid;
        return SQLITE_READONLY;
    });
}

int vlite_begin(sqlite3_vtab* tab) {
    return run_vtab_callback(tab, [&]() -> int {
        (void)tab;
        return SQLITE_OK;
    });
}

int vlite_sync(sqlite3_vtab* tab) {
    return run_vtab_callback(tab, [&]() -> int {
        (void)tab;
        return SQLITE_OK;
    });
}

int vlite_commit(sqlite3_vtab* tab) {
    return run_vtab_callback(tab, [&]() -> int {
        (void)tab;
        return SQLITE_OK;
    });
}

int vlite_rollback(sqlite3_vtab* tab) {
    return run_vtab_callback(tab, [&]() -> int {
        (void)tab;
        return SQLITE_OK;
    });
}

int vlite_find_function(sqlite3_vtab* tab, int argc, const char* name,
    void (**func)(sqlite3_context*, int, sqlite3_value**), void** user_data) {
    return run_vtab_callback(tab, [&]() -> int {
        (void)tab;
        (void)argc;
        (void)name;
        (void)func;
        (void)user_data;
        return 0;
    });
}

int vlite_rename(sqlite3_vtab* tab, const char* new_name) {
    return run_vtab_callback(tab, [&]() -> int {
        (void)tab;
        (void)new_name;
        return SQLITE_OK;
    });
}

int vlite_savepoint(sqlite3_vtab* tab, int savepoint_id) {
    return run_vtab_callback(tab, [&]() -> int {
        (void)tab;
        (void)savepoint_id;
        return SQLITE_OK;
    });
}

int vlite_release(sqlite3_vtab* tab, int savepoint_id) {
    return run_vtab_callback(tab, [&]() -> int {
        (void)tab;
        (void)savepoint_id;
        return SQLITE_OK;
    });
}

int vlite_rollback_to(sqlite3_vtab* tab, int savepoint_id) {
    return run_vtab_callback(tab, [&]() -> int {
        (void)tab;
        (void)savepoint_id;
        return SQLITE_OK;
    });
}

int vlite_shadow_name(const char* table_name) {
    (void)table_name;
    return 0;
}

int vlite_integrity(sqlite3_vtab* tab, const char* schema_name, const char* table_name,
    int flags, char** err_msg) {
    return run_errmsg_callback(err_msg, [&]() -> int {
        (void)tab;
        (void)schema_name;
        (void)table_name;
        (void)flags;
        (void)err_msg;
        return SQLITE_OK;
    });
}

sqlite3_module kVliteModule = {
    4,
    vlite_create,
    vlite_connect,
    vlite_best_index,
    vlite_disconnect,
    vlite_destroy,
    vlite_open,
    vlite_close,
    vlite_filter,
    vlite_next,
    vlite_eof,
    vlite_column,
    vlite_rowid,
    vlite_update,
    vlite_begin,
    vlite_sync,
    vlite_commit,
    vlite_rollback,
    vlite_find_function,
    vlite_rename,
    vlite_savepoint,
    vlite_release,
    vlite_rollback_to,
    vlite_shadow_name,
    vlite_integrity,
};

} // namespace

extern "C" int sqlite3_vlite_init(sqlite3* db, char** pz_err_msg, const sqlite3_api_routines* api) {
    return run_errmsg_callback(pz_err_msg, [&]() -> int {
        if (db == nullptr) {
            return SQLITE_ERROR;
        }

        SQLITE_EXTENSION_INIT2(api);
        return sqlite3_create_module_v2(db, sketch2::kVliteModuleName, &kVliteModule, nullptr, nullptr);
    });
}

extern "C" int sqlite3_extension_init(sqlite3* db, char** pz_err_msg, const sqlite3_api_routines* api) {
    return sqlite3_vlite_init(db, pz_err_msg, api);
}
