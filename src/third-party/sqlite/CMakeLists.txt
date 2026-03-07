add_library(sqlite3_lib STATIC sqlite3.c)

set_target_properties(sqlite3_lib PROPERTIES
  OUTPUT_NAME sqlite3
)

target_include_directories(sqlite3_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(sqlite3_lib
  PUBLIC
    SQLITE_THREADSAFE=1
    SQLITE_ENABLE_LOAD_EXTENSION=1
)

target_link_libraries(sqlite3_lib
  PUBLIC
    Threads::Threads
)

target_compile_options(sqlite3_lib PRIVATE -w)

add_executable(sqlite3_cli shell.c)

set_target_properties(sqlite3_cli PROPERTIES
  OUTPUT_NAME sqlite3
)

target_include_directories(sqlite3_cli
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(sqlite3_cli
  PRIVATE
    SQLITE_THREADSAFE=1
    SQLITE_ENABLE_LOAD_EXTENSION=1
)

target_compile_options(sqlite3_cli PRIVATE -w)

target_link_libraries(sqlite3_cli
  PRIVATE
    sqlite3_lib
    Threads::Threads
    dl
    m
)
