// Declares UpdateNotifier, a file-backed change counter for cross-process
// cache invalidation between dataset writers and readers.

#pragma once

#include "utils/shared_types.h"

#include <cstdint>
#include <string>

namespace sketch2 {

// UpdateNotifier uses an 8-byte counter stored in the dataset owner lock file
// to let reader processes detect that writer processes have changed data/delta
// files.  The writer side calls init_updater() + update(); the reader side
// calls init_checker() + check_updated().
//
// Each instance must be used from a single thread.
class UpdateNotifier {
public:
    UpdateNotifier() = default;
    UpdateNotifier(const UpdateNotifier&) = delete;
    UpdateNotifier& operator=(const UpdateNotifier&) = delete;
    UpdateNotifier(UpdateNotifier&&) = delete;
    UpdateNotifier& operator=(UpdateNotifier&&) = delete;
    ~UpdateNotifier();

    // Writer side: open or create the counter file in read-write mode.
    // Reads the existing counter value if the file already has one.
    Ret init_updater(const std::string& path);

    // Writer side: increment the counter, write it to the file, and
    // fdatasync() so readers on other processes see the new value.
    Ret update();

    // Reader side: store the file path for later use by check_updated().
    Ret init_checker(const std::string& path);

    // Reader side: returns true when the counter has changed since the last
    // call (or on the very first call).  Also returns true on any I/O error
    // so callers conservatively flush their caches.
    bool check_updated();

private:
    int fd_ = -1;
    uint64_t counter_ = 0;
    std::string path_;
    bool is_updater_ = false;
};

} // namespace sketch2
