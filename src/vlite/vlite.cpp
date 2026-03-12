#include "vlite.h"

#include "core/compute/scanner.h"
#include "core/storage/dataset.h"
#include "core/utils/string_utils.h"

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <vector>

SQLITE_EXTENSION_INIT1

namespace {

constexpr const char* kVliteModuleName = "vlite";
constexpr const char* kVliteSchema =
    "CREATE TABLE x(query TEXT HIDDEN, k INTEGER HIDDEN, id INTEGER, distance REAL)";

using DatasetInitFromPath = sketch2::Ret (sketch2::Dataset::*)(const std::string&);
using ScannerFindDataset = sketch2::Ret (sketch2::Scanner::*)(
    const sketch2::Dataset&, size_t, const uint8_t*, std::vector<uint64_t>&) const;

[[maybe_unused]] DatasetInitFromPath kDatasetInitAnchor =
    static_cast<DatasetInitFromPath>(&sketch2::Dataset::init);
[[maybe_unused]] ScannerFindDataset kScannerFindAnchor =
    static_cast<ScannerFindDataset>(&sketch2::Scanner::find);
[[maybe_unused]] auto kParseVectorAnchor = &sketch2::parse_vector;

struct VliteVTab : sqlite3_vtab {
};

struct VliteCursor : sqlite3_vtab_cursor {
    bool eof = true;
    sqlite3_int64 rowid = 0;
};

int vlite_connect(sqlite3* db, sqlite3_vtab** pp_vtab) {
    if (db == nullptr || pp_vtab == nullptr) {
        return SQLITE_ERROR;
    }

    const int declare_rc = sqlite3_declare_vtab(db, kVliteSchema);
    if (declare_rc != SQLITE_OK) {
        return declare_rc;
    }

    auto* vtab = new (std::nothrow) VliteVTab();
    if (vtab == nullptr) {
        return SQLITE_NOMEM;
    }

    *pp_vtab = vtab;
    return SQLITE_OK;
}

int vlite_create(sqlite3* db, void* aux, int argc, const char* const* argv,
    sqlite3_vtab** pp_vtab, char** err_msg) {
    (void)aux;
    (void)argc;
    (void)argv;
    (void)err_msg;
    return vlite_connect(db, pp_vtab);
}

int vlite_connect(sqlite3* db, void* aux, int argc, const char* const* argv,
    sqlite3_vtab** pp_vtab, char** err_msg) {
    (void)aux;
    (void)argc;
    (void)argv;
    (void)err_msg;
    return vlite_connect(db, pp_vtab);
}

int vlite_best_index(sqlite3_vtab* tab, sqlite3_index_info* index_info) {
    (void)tab;
    (void)index_info;
    return SQLITE_OK;
}

int vlite_disconnect(sqlite3_vtab* tab) {
    delete static_cast<VliteVTab*>(tab);
    return SQLITE_OK;
}

int vlite_destroy(sqlite3_vtab* tab) {
    return vlite_disconnect(tab);
}

int vlite_open(sqlite3_vtab* tab, sqlite3_vtab_cursor** pp_cursor) {
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
}

int vlite_close(sqlite3_vtab_cursor* cursor) {
    delete static_cast<VliteCursor*>(cursor);
    return SQLITE_OK;
}

int vlite_filter(sqlite3_vtab_cursor* cursor, int idx_num, const char* idx_str,
    int argc, sqlite3_value** argv) {
    (void)idx_num;
    (void)idx_str;
    (void)argc;
    (void)argv;

    if (cursor == nullptr) {
        return SQLITE_ERROR;
    }

    auto* vlite_cursor = static_cast<VliteCursor*>(cursor);
    vlite_cursor->eof = true;
    vlite_cursor->rowid = 0;
    return SQLITE_OK;
}

int vlite_next(sqlite3_vtab_cursor* cursor) {
    if (cursor == nullptr) {
        return SQLITE_ERROR;
    }

    static_cast<VliteCursor*>(cursor)->eof = true;
    return SQLITE_OK;
}

int vlite_eof(sqlite3_vtab_cursor* cursor) {
    if (cursor == nullptr) {
        return 1;
    }

    return static_cast<VliteCursor*>(cursor)->eof ? 1 : 0;
}

int vlite_column(sqlite3_vtab_cursor* cursor, sqlite3_context* context, int column) {
    (void)cursor;
    (void)column;

    if (context != nullptr) {
        sqlite3_result_null(context);
    }
    return SQLITE_OK;
}

int vlite_rowid(sqlite3_vtab_cursor* cursor, sqlite3_int64* rowid) {
    if (cursor == nullptr || rowid == nullptr) {
        return SQLITE_ERROR;
    }

    *rowid = static_cast<VliteCursor*>(cursor)->rowid;
    return SQLITE_OK;
}

int vlite_update(sqlite3_vtab* tab, int argc, sqlite3_value** argv, sqlite3_int64* rowid) {
    (void)tab;
    (void)argc;
    (void)argv;
    (void)rowid;
    return SQLITE_READONLY;
}

int vlite_begin(sqlite3_vtab* tab) {
    (void)tab;
    return SQLITE_OK;
}

int vlite_sync(sqlite3_vtab* tab) {
    (void)tab;
    return SQLITE_OK;
}

int vlite_commit(sqlite3_vtab* tab) {
    (void)tab;
    return SQLITE_OK;
}

int vlite_rollback(sqlite3_vtab* tab) {
    (void)tab;
    return SQLITE_OK;
}

int vlite_find_function(sqlite3_vtab* tab, int argc, const char* name,
    void (**func)(sqlite3_context*, int, sqlite3_value**), void** user_data) {
    (void)tab;
    (void)argc;
    (void)name;
    (void)func;
    (void)user_data;
    return 0;
}

int vlite_rename(sqlite3_vtab* tab, const char* new_name) {
    (void)tab;
    (void)new_name;
    return SQLITE_OK;
}

int vlite_savepoint(sqlite3_vtab* tab, int savepoint_id) {
    (void)tab;
    (void)savepoint_id;
    return SQLITE_OK;
}

int vlite_release(sqlite3_vtab* tab, int savepoint_id) {
    (void)tab;
    (void)savepoint_id;
    return SQLITE_OK;
}

int vlite_rollback_to(sqlite3_vtab* tab, int savepoint_id) {
    (void)tab;
    (void)savepoint_id;
    return SQLITE_OK;
}

int vlite_shadow_name(const char* table_name) {
    (void)table_name;
    return 0;
}

int vlite_integrity(sqlite3_vtab* tab, const char* schema_name, const char* table_name,
    int flags, char** err_msg) {
    (void)tab;
    (void)schema_name;
    (void)table_name;
    (void)flags;
    (void)err_msg;
    return SQLITE_OK;
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
    (void)pz_err_msg;

    if (db == nullptr) {
        return SQLITE_ERROR;
    }

    SQLITE_EXTENSION_INIT2(api);
    return sqlite3_create_module_v2(db, kVliteModuleName, &kVliteModule, nullptr, nullptr);
}

extern "C" int sqlite3_extension_init(sqlite3* db, char** pz_err_msg, const sqlite3_api_routines* api) {
    return sqlite3_vlite_init(db, pz_err_msg, api);
}
