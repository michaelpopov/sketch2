// Declares the SQLite extension entry points for the vlite module.

#pragma once

#include "sqlite3ext.h"

#ifdef __cplusplus
extern "C" {
#endif

int sqlite3_sketch2_init(sqlite3* db, char** pzErrMsg, const sqlite3_api_routines* api);
int sqlite3_extension_init(sqlite3* db, char** pzErrMsg, const sqlite3_api_routines* api);

#ifdef __cplusplus
}
#endif
