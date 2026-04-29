// Implements the SQLite virtual table that exposes vector search over datasets.

#include "vlite.h"
#include "sketch2api/sketch2.h"
#include "sketch2api/sketch2api_testing.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

SQLITE_EXTENSION_INIT1

namespace {

enum VliteColumn {
    kColumnQuery = 0,
    kColumnMatchExpr = 1,
    kColumnK = 2,
    kColumnAllowedIds = 3,
    kColumnId = 4,
    kColumnScore = 5,
};

enum VliteConstraintBit {
    kConstraintQuery = 1 << 0,
    kConstraintK = 1 << 1,
    kConstraintLimit = 1 << 2,
    kConstraintOffset = 1 << 3,
    kConstraintAllowedIds = 1 << 4,
};

constexpr const char* kVliteSchemaWithAllowedIds =
    "CREATE TABLE x("
    "query TEXT HIDDEN, "
    "match_expr TEXT HIDDEN, "
    "k INTEGER HIDDEN, "
    "allowed_ids BLOB HIDDEN, "
    "id INTEGER, "
    "score REAL)";
constexpr const char* kVliteModuleName = "vlite";
constexpr const char* kBitsetFilterPointerType = "sketch2.BitsetFilterBlob";

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
        set_err_msg(err_msg, "sketch2: out of memory");
        return SQLITE_NOMEM;
    } catch (const std::exception& ex) {
        set_err_msg(err_msg, ex.what());
        return SQLITE_ERROR;
    } catch (...) {
        set_err_msg(err_msg, "sketch2: unexpected error");
        return SQLITE_ERROR;
    }
}

template <typename Func>
int run_vtab_callback(sqlite3_vtab* tab, Func func) {
    try {
        return func();
    } catch (const std::bad_alloc&) {
        set_vtab_error(tab, "sketch2: out of memory");
        return SQLITE_NOMEM;
    } catch (const std::exception& ex) {
        set_vtab_error(tab, ex.what());
        return SQLITE_ERROR;
    } catch (...) {
        set_vtab_error(tab, "sketch2: unexpected error");
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
        set_vtab_error(cursor != nullptr ? cursor->pVtab : nullptr, "sketch2: out of memory");
        return SQLITE_NOMEM;
    } catch (const std::exception& ex) {
        if (context != nullptr) {
            sqlite3_result_error(context, ex.what(), -1);
        }
        set_vtab_error(cursor != nullptr ? cursor->pVtab : nullptr, ex.what());
        return SQLITE_ERROR;
    } catch (...) {
        constexpr const char* kUnexpectedError = "sketch2: unexpected error";
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

bool consumes_order_by_score(bool smaller_score_is_better, const sqlite3_index_info& index_info) {
    if (index_info.nOrderBy != 1 || index_info.aOrderBy[0].iColumn != kColumnScore) {
        return false;
    }
    const bool desc = index_info.aOrderBy[0].desc != 0;
    return smaller_score_is_better ? !desc : desc;
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

void release_bitset_filter(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    sk_bitset_delete(ptr);
}

// SQLite owns this aggregate context per GROUP BY group. We keep only the
// API-owned builder here. Finalization returns an API-owned typed pointer.
struct BitsetAggState {
    void* bitset_filter_builder = nullptr;
    uint64_t name_hash = 0;
    int name_size = 0;
    bool has_name_fingerprint = false;
    bool has_error = false;
    bool has_nomem = false;
    const char* error_message = nullptr;
};

void set_bitset_agg_error(BitsetAggState* state, bool nomem, const char* message) {
    if (state == nullptr) {
        return;
    }
    state->has_error = true;
    state->has_nomem = nomem;
    state->error_message = message;
}

const char* sqlite_text_value_or_nomem(sqlite3_context* context, sqlite3_value* value) {
    const char* text = reinterpret_cast<const char*>(sqlite3_value_text(value));
    if (text == nullptr) {
        sqlite3_result_error_nomem(context);
    }
    return text;
}

uint64_t hash_sqlite_text(const char* text, int size) {
    uint64_t hash = 1469598103934665603ull;
    for (int i = 0; i < size; ++i) {
        hash ^= static_cast<unsigned char>(text[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

void bitset_agg_step(sqlite3_context* context, int argc, sqlite3_value** argv) {
    if (context == nullptr) {
        return;
    }
    if ((argc != 1 && argc != 2) || argv == nullptr) {
        sqlite3_result_error(context, "bitset_agg: invalid arguments", -1);
        return;
    }

    auto* state = static_cast<BitsetAggState*>(sqlite3_aggregate_context(context, sizeof(BitsetAggState)));
    if (state == nullptr) {
        sqlite3_result_error_nomem(context);
        return;
    }
    if (state->has_error) {
        return;
    }

    if (argc == 2) {
        const int parameter_type = sqlite3_value_type(argv[1]);
        if (parameter_type != SQLITE_NULL && parameter_type != SQLITE_TEXT) {
            set_bitset_agg_error(state, false, "bitset_agg: parameter must be a string");
            sqlite3_result_error(context, state->error_message, -1);
            return;
        }
    }

    const char* name = nullptr;
    if (argc == 2 && sqlite3_value_type(argv[1]) != SQLITE_NULL) {
        name = sqlite_text_value_or_nomem(context, argv[1]);
        if (name == nullptr) {
            set_bitset_agg_error(state, true, "sketch2: out of memory");
            return;
        }
    }

    bool name_checked = false;
    if (name != nullptr) {
        const int name_size = sqlite3_value_bytes(argv[1]);
        const uint64_t name_hash = hash_sqlite_text(name, name_size);
        name_checked = state->has_name_fingerprint &&
            state->name_size == name_size &&
            state->name_hash == name_hash;
        if (!name_checked) {
            if (sk_bitset_create_builder(
                    &state->bitset_filter_builder, &state->has_nomem,
                    &state->error_message, name) != 0) {
                state->has_error = true;
                if (state->has_nomem) {
                    sqlite3_result_error_nomem(context);
                    return;
                }
                const char* message = state->error_message != nullptr
                    ? state->error_message
                    : "bitset_agg: aggregation failed";
                sqlite3_result_error(context, message, -1);
                return;
            }

            state->name_size = name_size;
            state->name_hash = name_hash;
            state->has_name_fingerprint = true;
        }
    }

    sqlite3_value* value = argv[0];
    const int value_type = sqlite3_value_type(value);
    if (value_type == SQLITE_NULL) {
        return;
    }
    if (value_type != SQLITE_INTEGER) {
        set_bitset_agg_error(state, false, "bitset_agg: id must be an integer");
        sqlite3_result_error(context, state->error_message, -1);
        return;
    }

    const sqlite3_int64 id = sqlite3_value_int64(value);
    if (id < 0) {
        set_bitset_agg_error(state, false, "bitset_agg: id must be non-negative");
        sqlite3_result_error(context, state->error_message, -1);
        return;
    }

    const sqlite3_uint64 id_u64 = static_cast<sqlite3_uint64>(id);
    const int add_rc = name != nullptr
        ? sk_bitset_add_id_name(
            &state->bitset_filter_builder, id_u64, &state->has_nomem, &state->error_message)
        : sk_bitset_add_id(
            &state->bitset_filter_builder, id_u64, &state->has_nomem, &state->error_message, nullptr);
    if (add_rc != 0) {
        state->has_error = true;
        if (state->has_nomem) {
            sqlite3_result_error_nomem(context);
            return;
        }
        const char* message =
            state->error_message != nullptr ? state->error_message : "bitset_agg: aggregation failed";
        sqlite3_result_error(context, message, -1);
    }
}

void bitset_agg_final(sqlite3_context* context) {
    auto* state = static_cast<BitsetAggState*>(
        sqlite3_aggregate_context(context, sizeof(BitsetAggState)));
    if (state == nullptr) {
        sqlite3_result_error_nomem(context);
        return;
    }

    void* bitset_filter = nullptr;
    bool finish_nomem = false;
    const char* finish_error = nullptr;
    if (sk_bitset_finish(
            &state->bitset_filter_builder, &bitset_filter, &finish_nomem, &finish_error) != 0) {
        state->has_error = true;
        state->has_nomem = finish_nomem;
        state->error_message = finish_error;
    }

    if (state->has_error) {
        sk_bitset_delete(bitset_filter);
        if (state->has_nomem) {
            sqlite3_result_error_nomem(context);
            return;
        }
        const char* message =
            state->error_message != nullptr ? state->error_message : "bitset_agg: aggregation failed";
        sqlite3_result_error(context, message, -1);
        return;
    }

    assert(bitset_filter != nullptr);

    sqlite3_result_pointer(context, bitset_filter, kBitsetFilterPointerType, release_bitset_filter);
}

void bitset_drop_func(sqlite3_context* context, int argc, sqlite3_value** argv) {
    if (context == nullptr) {
        return;
    }
    if (argc != 1 || argv == nullptr) {
        sqlite3_result_error(context, "bitset_drop: invalid arguments", -1);
        return;
    }

    if (sqlite3_value_type(argv[0]) != SQLITE_TEXT) {
        sqlite3_result_error(context, "bitset_drop: name must be a string", -1);
        return;
    }

    const char* name = sqlite_text_value_or_nomem(context, argv[0]);
    if (name == nullptr) {
        return;
    }
    int removed = 0;
    bool out_of_memory = false;
    const char* error_message = nullptr;
    if (sk_bitset_drop(name, &removed, &out_of_memory, &error_message) != 0) {
        if (out_of_memory) {
            sqlite3_result_error_nomem(context);
            return;
        }
        sqlite3_result_error(
            context, error_message != nullptr ? error_message : "bitset_drop: failed", -1);
        return;
    }

    sqlite3_result_int(context, removed);
}

void bitset_load_func(sqlite3_context* context, int argc, sqlite3_value** argv) {
    if (context == nullptr) {
        return;
    }
    if (argc != 1 || argv == nullptr) {
        sqlite3_result_error(context, "bitset_load: invalid arguments", -1);
        return;
    }

    if (sqlite3_value_type(argv[0]) != SQLITE_TEXT) {
        sqlite3_result_error(context, "bitset_load: name must be a string", -1);
        return;
    }

    try {
        const char* name = sqlite_text_value_or_nomem(context, argv[0]);
        if (name == nullptr) {
            return;
        }

        void* bitset_filter = nullptr;
        bool out_of_memory = false;
        const char* error_message = nullptr;
        if (sk_bitset_load(name, &bitset_filter, &out_of_memory, &error_message) != 0) {
            if (out_of_memory) {
                sqlite3_result_error_nomem(context);
                return;
            }
            sqlite3_result_error(
                context, error_message != nullptr ? error_message : "bitset_load: failed", -1);
            return;
        }

        assert(bitset_filter != nullptr);
        sqlite3_result_pointer(context, bitset_filter, kBitsetFilterPointerType, release_bitset_filter);
    } catch (const std::bad_alloc&) {
        sqlite3_result_error_nomem(context);
    } catch (const std::exception& ex) {
        sqlite3_result_error(context, ex.what(), -1);
    } catch (...) {
        sqlite3_result_error(context, "sketch2: unexpected error", -1);
    }
}

// VliteVTab exists to bind SQLite's virtual-table object to the dataset state
// needed by the extension. It stores the dataset path and the opened Dataset instance.
struct VliteVTab : sqlite3_vtab {
    std::string db_path;
    std::string dataset_name;
    sk_handle_t* handle = nullptr;
    bool smaller_score_is_better = false;
};

struct VliteRow {
    uint64_t id = 0;
    double score = 0.0;
};

// VliteCursor exists to hold one materialized query result set for SQLite. It
// keeps the parsed query buffer, result rows, and iteration state consumed by
// the xFilter/xNext/xColumn callbacks.
struct VliteCursor : sqlite3_vtab_cursor {
    std::vector<VliteRow> rows;
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
        if (argc != 5) {
            set_err_msg(err_msg, "vlite requires exactly two arguments: db path and dataset name");
            return SQLITE_ERROR;
        }

        const int declare_rc = sqlite3_declare_vtab(db, kVliteSchemaWithAllowedIds);
        if (declare_rc != SQLITE_OK) {
            return declare_rc;
        }

        auto* vtab = new (std::nothrow) VliteVTab();
        if (vtab == nullptr) {
            return SQLITE_NOMEM;
        }

        vtab->db_path = dequote_sqlite_arg(argv[3]);
        if (vtab->db_path.empty()) {
            set_err_msg(err_msg, "vlite db path must not be empty");
            delete vtab;
            return SQLITE_ERROR;
        }

        vtab->dataset_name = dequote_sqlite_arg(argv[4]);
        if (vtab->dataset_name.empty()) {
            set_err_msg(err_msg, "vlite dataset name must not be empty");
            delete vtab;
            return SQLITE_ERROR;
        }

        vtab->handle = sk_new_handle(vtab->db_path.c_str());
        if (vtab->handle == nullptr) {
            set_err_msg(err_msg, "vlite failed to create Sketch2 API handle");
            delete vtab;
            return SQLITE_ERROR;
        }
        if (sk_open(vtab->handle, vtab->dataset_name.c_str()) != 0) {
            set_err_msg(err_msg, sk_error_message(vtab->handle));
            sk_release_handle(vtab->handle);
            vtab->handle = nullptr;
            delete vtab;
            return SQLITE_ERROR;
        }

        bool smaller_is_better = false;
        if (sk_score_ascending_is_better(vtab->handle, &smaller_is_better) != 0) {
            set_err_msg(err_msg, sk_error_message(vtab->handle));
            sk_release_handle(vtab->handle);
            vtab->handle = nullptr;
            delete vtab;
            return SQLITE_ERROR;
        }
        vtab->smaller_score_is_better = smaller_is_better;

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
        int allowed_ids_constraint = -1;
        int limit_constraint = -1;
        int offset_constraint = -1;

        for (int i = 0; i < index_info->nConstraint; ++i) {
            const auto& constraint = index_info->aConstraint[i];
            if (!constraint.usable) {
                continue;
            }
            if (query_constraint < 0 &&
                    (constraint.iColumn == kColumnQuery || constraint.iColumn == kColumnMatchExpr) &&
                    is_query_constraint(constraint.op)) {
                query_constraint = i;
            } else if (k_constraint < 0 &&
                    constraint.iColumn == kColumnK &&
                    constraint.op == SQLITE_INDEX_CONSTRAINT_EQ) {
                k_constraint = i;
            } else if (allowed_ids_constraint < 0 &&
                    constraint.iColumn == kColumnAllowedIds &&
                    constraint.op == SQLITE_INDEX_CONSTRAINT_EQ) {
                allowed_ids_constraint = i;
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
        if (allowed_ids_constraint >= 0) {
            index_info->aConstraintUsage[allowed_ids_constraint].argvIndex = next_arg++;
            index_info->aConstraintUsage[allowed_ids_constraint].omit = 1;
            idx_num |= kConstraintAllowedIds;
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
        auto* vlite_vtab = static_cast<VliteVTab*>(tab);
        if (vlite_vtab != nullptr &&
                consumes_order_by_score(vlite_vtab->smaller_score_is_better, *index_info)) {
            index_info->orderByConsumed = 1;
        }
        return SQLITE_OK;
    });
}

int vlite_disconnect(sqlite3_vtab* tab) {
    auto* vtab = static_cast<VliteVTab*>(tab);
    if (vtab != nullptr && vtab->handle != nullptr) {
        sk_release_handle(vtab->handle);
        vtab->handle = nullptr;
    }
    delete vtab;
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

        const void* allowed_ids_blob = nullptr;
        int allowed_ids_blob_size = 0;
        const void* allowed_ids = nullptr;
        if ((idx_num & kConstraintAllowedIds) != 0) {
            if (arg_index >= argc) {
                set_vtab_error(vlite_vtab, "vlite missing allowed_ids value");
                return SQLITE_ERROR;
            }

            sqlite3_value* allowed_ids_value = argv[arg_index++];
            allowed_ids = sqlite3_value_pointer(allowed_ids_value, kBitsetFilterPointerType);
            const int allowed_ids_type = sqlite3_value_type(allowed_ids_value);
            if (allowed_ids == nullptr &&
                    allowed_ids_type != SQLITE_NULL && allowed_ids_type != SQLITE_BLOB) {
                set_vtab_error(vlite_vtab, "vlite allowed_ids must be a BLOB or NULL");
                return SQLITE_ERROR;
            }
            if (allowed_ids == nullptr && allowed_ids_type == SQLITE_BLOB) {
                allowed_ids_blob = sqlite3_value_blob(allowed_ids_value);
                allowed_ids_blob_size = sqlite3_value_bytes(allowed_ids_value);
                if (allowed_ids_blob_size == 0) {
                    set_vtab_error(vlite_vtab, "vlite allowed_ids BLOB must not be empty");
                    return SQLITE_ERROR;
                }
            }
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

        if (vlite_vtab->handle == nullptr) {
            set_vtab_error(vlite_vtab, "vlite dataset handle is not initialized");
            return SQLITE_ERROR;
        }

        if (effective_k > static_cast<sqlite3_int64>(std::numeric_limits<unsigned int>::max())) {
            set_vtab_error(vlite_vtab, "vlite k is too large");
            return SQLITE_ERROR;
        }

        uint64_t* ids = nullptr;
        double* scores = nullptr;
        size_t count = 0;
        const int rc = allowed_ids != nullptr
            ? sk_knn_items_bitset_filter(
                vlite_vtab->handle,
                vlite_cursor->query_text.c_str(),
                static_cast<unsigned int>(effective_k),
                allowed_ids,
                &ids,
                &scores,
                &count)
            : sk_knn_items(
                vlite_vtab->handle,
                vlite_cursor->query_text.c_str(),
                static_cast<unsigned int>(effective_k),
                allowed_ids_blob,
                static_cast<size_t>(std::max(allowed_ids_blob_size, 0)),
                &ids,
                &scores,
                &count);
        if (rc != 0) {
            set_vtab_error(vlite_vtab, sk_error_message(vlite_vtab->handle));
            sk_free(ids);
            sk_free(scores);
            return SQLITE_ERROR;
        }
        vlite_cursor->rows.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            vlite_cursor->rows.push_back(VliteRow{ids[i], scores[i]});
        }
        sk_free(ids);
        sk_free(scores);

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

        const VliteRow& row = vlite_cursor->rows[vlite_cursor->index];
        switch (column) {
            case kColumnQuery:
            case kColumnMatchExpr:
                sqlite3_result_text(context, vlite_cursor->query_text.c_str(), -1, SQLITE_TRANSIENT);
                break;
            case kColumnK:
                sqlite3_result_int64(context, vlite_cursor->k);
                break;
            case kColumnAllowedIds:
                sqlite3_result_null(context);
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
            case kColumnScore:
                sqlite3_result_double(context, row.score);
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

extern "C" int sqlite3_sketch2_init(sqlite3* db, char** pz_err_msg, const sqlite3_api_routines* api) {
    return run_errmsg_callback(pz_err_msg, [&]() -> int {
        if (db == nullptr) {
            return SQLITE_ERROR;
        }

        SQLITE_EXTENSION_INIT2(api);
        int rc = sqlite3_create_module_v2(db, kVliteModuleName, &kVliteModule, nullptr, nullptr);
        if (rc != SQLITE_OK) {
            return rc;
        }

        rc = sqlite3_create_function_v2(
            db,
            "bitset_agg",
            -1,
            SQLITE_UTF8,
            nullptr,
            nullptr,
            bitset_agg_step,
            bitset_agg_final,
            nullptr);
        if (rc != SQLITE_OK) {
            return rc;
        }

        rc = sqlite3_create_function_v2(
            db,
            "bitset_drop",
            1,
            SQLITE_UTF8,
            nullptr,
            bitset_drop_func,
            nullptr,
            nullptr,
            nullptr);
        if (rc != SQLITE_OK) {
            return rc;
        }

        rc = sqlite3_create_function_v2(
            db,
            "bitset_load",
            1,
            SQLITE_UTF8,
            nullptr,
            bitset_load_func,
            nullptr,
            nullptr,
            nullptr);
        if (rc != SQLITE_OK) {
            return rc;
        }

        return SQLITE_OK;
    });
}

extern "C" int sqlite3_extension_init(sqlite3* db, char** pz_err_msg, const sqlite3_api_routines* api) {
    return sqlite3_sketch2_init(db, pz_err_msg, api);
}

extern "C" const char* sqlite3_sketch2_knn_engine_name_for_testing(void) {
    return sk_knn_engine_name_for_testing();
}
