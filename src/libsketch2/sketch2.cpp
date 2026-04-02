// libsketch2.so - Unified Sketch2 runtime library
// Consolidates the sketch2api C API and the vlite SQLite virtual table extension.

#include "sketch2api/sketch2api.h"
#include "db/vlite/vlite.h"

// Anchor recently added C API entry points so the linker pulls them into the
// shared library even on toolchains that aggressively drop unreferenced
// archive objects.
namespace {
[[maybe_unused]] auto* ensure_export_sk_new_handler = &sk_new_handler;
[[maybe_unused]] auto* ensure_export_sk_release_handler = &sk_release_handler;
} // namespace
