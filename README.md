# sketch2 (dummy skeleton)

Generated C++20 **CMake + Ninja** project skeleton for Sketch2.

## Quick start

See `commands.txt` for copy/paste shell commands.

After a Release build, you should have:

- `bin/libparasol.so` (or platform equivalent)
- `bin/libvlite.so` (or platform equivalent)
- `bin/sqlite3`

After a Debug build, you should have:

- `bin-dbg/libparasol.so` (or platform equivalent)
- `bin-dbg/libvlite.so` (or platform equivalent)
- `bin-dbg/sqlite3`

Unit tests are also built and can be run via `ctest`.

## Notes

- Uses **g++** when configured with `-DCMAKE_CXX_COMPILER=g++`
- Uses **-Wall -Wextra -Wpedantic -Werror** on GCC/Clang
- Downloads GoogleTest automatically via `FetchContent`
