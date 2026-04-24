add_subdirectory(third-party/croaring)
add_subdirectory(core)
add_subdirectory(db)
add_subdirectory(libsketch2)
add_subdirectory(sketch2api)
add_subdirectory(third-party/sqlite)
# Google Highway — vendored SIMD library.
# Disable tests, examples, contrib, and install to keep the build lightweight.
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(HWY_WARNINGS_ARE_ERRORS OFF CACHE BOOL "" FORCE)
add_subdirectory(third-party/google_highway)
