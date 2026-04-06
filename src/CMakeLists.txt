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
# Third-party code should not be subject to our -Werror / -Wpedantic flags.
foreach(_hwy_tgt hwy hwy_test hwy_list_targets)
  target_compile_options(${_hwy_tgt} PRIVATE -Wno-error -Wno-pedantic)
endforeach()
