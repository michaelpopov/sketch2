// Unit tests for BitsetFilterControl storage ownership.

#include "bitset_file_cache.h"
#include "bitset_filter_control.h"
#include "utest_chunked_bits_helpers.h"
#include "utils/singleton.h"

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace sketch2 {
class ChunkedBitsTestPeer {
public:
    static void mark_finish_failed(ChunkedBits* bits, const Ret& ret) {
        bits->finished_ = true;
        bits->cached_serialized_size_ = 0;
        bits->finish_ret_ = ret;
    }
};

namespace {

using test::expect_persisted_filter_contains;
using test::unique_filter_name;

std::string named_temp_file_prefix_suffix() {
    std::string suffix = kBitsetFilterNamedTempFileTemplateSuffix;
    suffix.resize(suffix.size() - 6);
    return suffix;
}

size_t count_visible_temporary_spill_files() {
    size_t count = 0;
    const std::filesystem::path spill_dir = get_singleton().bitset_filter_spill_dir();
    if (!std::filesystem::exists(spill_dir)) {
        return 0;
    }
    for (const std::filesystem::directory_entry& entry :
            std::filesystem::directory_iterator(spill_dir)) {
        if (entry.path().filename().string().rfind("sketch2_bitset_filter_", 0) == 0) {
            ++count;
        }
    }
    return count;
}

size_t count_visible_named_temp_files(const std::string& name) {
    size_t count = 0;
    const std::filesystem::path spill_dir = get_singleton().bitset_filter_spill_dir();
    if (!std::filesystem::exists(spill_dir)) {
        return 0;
    }
    const std::string prefix = name + named_temp_file_prefix_suffix();
    for (const std::filesystem::directory_entry& entry :
            std::filesystem::directory_iterator(spill_dir)) {
        if (entry.path().filename().string().rfind(prefix, 0) == 0) {
            ++count;
        }
    }
    return count;
}

std::vector<uint64_t> make_spill_bitset_filter_ids(
        std::initializer_list<uint64_t> selected_ids = {20, 40}) {
    std::vector<uint64_t> ids;
    ids.reserve(50000 + selected_ids.size());
    for (uint64_t id : selected_ids) {
        ids.push_back(id);
    }
    for (uint64_t i = 0; i < 50000; ++i) {
        ids.push_back((i + 1) << kChunkBits);
    }
    return ids;
}

BitsetFilterControlPtr create_control(
        const std::vector<uint64_t>& ids, const char* name = nullptr) {
    ChunkedBits bits;
    EXPECT_EQ(0, bits.set_name(name).code());
    if (::testing::Test::HasFailure()) {
        return nullptr;
    }
    for (uint64_t id : ids) {
        EXPECT_EQ(0, bits.add(id).code());
        if (::testing::Test::HasFailure()) {
            return nullptr;
        }
    }

    BitsetFilterControlPtr control;
    EXPECT_EQ(0, BitsetFilterControl::create(bits, &control).code());
    EXPECT_NE(nullptr, control);
    return control;
}

} // namespace

TEST(bitset_filter_control, default_control_is_empty_heap_filter) {
    BitsetFilterControlPtr control;
    ASSERT_EQ(0, BitsetFilterControl::create_empty(&control).code());
    ASSERT_NE(nullptr, control);

    EXPECT_EQ(BitsetFilterStorageKind::Heap, bitset_filter_storage_kind_for_testing(control.get()));
    EXPECT_TRUE(control->view.begin().eof());
}

TEST(bitset_filter_control, small_filter_uses_heap_and_releases_cleanly) {
    BitsetFilterControlPtr control = create_control({20, 40});

    EXPECT_EQ(BitsetFilterStorageKind::Heap, bitset_filter_storage_kind_for_testing(control.get()));
    EXPECT_FALSE(control->view.begin().eof());

    control.reset();
    EXPECT_EQ(nullptr, control);
}

TEST(bitset_filter_control, named_small_filter_spills_to_persistent_named_file) {
    const std::string name = unique_filter_name("small_named_filter");
    const std::filesystem::path expected_path = named_bitset_filter_path(name);
    std::filesystem::remove(expected_path);

    BitsetFilterControlPtr control = create_control({20, 40}, name.c_str());
    ASSERT_EQ(BitsetFilterStorageKind::MappedFile,
        bitset_filter_storage_kind_for_testing(control.get()));

    EXPECT_TRUE(std::filesystem::exists(expected_path));
    EXPECT_GT(std::filesystem::file_size(expected_path), 0u);
    EXPECT_GE(control->storage->fd, 0);
    expect_persisted_filter_contains(expected_path, {20, 40});

    control.reset();
    EXPECT_EQ(nullptr, control);
    EXPECT_TRUE(std::filesystem::exists(expected_path));
    expect_persisted_filter_contains(expected_path, {20, 40});

    std::filesystem::remove(expected_path);
}

TEST(bitset_filter_control, load_named_filter_maps_persistent_file) {
    const std::string name = unique_filter_name("load_filter");
    const std::filesystem::path expected_path = named_bitset_filter_path(name);
    std::filesystem::remove(expected_path);

    BitsetFilterControlPtr created = create_control({20, 40}, name.c_str());
    ASSERT_NE(nullptr, created);
    created.reset();

    // Force the load to go through the on-disk open + mmap path so we can
    // assert the read-only mapping behavior independently of any cache hit.
    bitset_file_cache().clear();

    BitsetFilterControlPtr loaded;
    ASSERT_EQ(0, load_named_bitset_filter(name.c_str(), &loaded).code());
    ASSERT_NE(nullptr, loaded);
    EXPECT_EQ(BitsetFilterStorageKind::MappedFile,
        bitset_filter_storage_kind_for_testing(loaded.get()));
    EXPECT_NE(nullptr, loaded->storage->data());
    EXPECT_EQ(nullptr, loaded->storage->writable_data());
    EXPECT_GE(loaded->storage->fd, 0);

    auto it = loaded->view.begin();
    EXPECT_TRUE(it.consume_if_equal(20));
    EXPECT_TRUE(it.consume_if_equal(40));
    EXPECT_FALSE(it.consume_if_equal(60));

    loaded.reset();
    bitset_file_cache().remove(name);
    std::filesystem::remove(expected_path);
}

TEST(bitset_filter_control, large_filter_spills_to_temporary_mapped_file) {
    const size_t visible_spill_files_before = count_visible_temporary_spill_files();

    BitsetFilterControlPtr control = create_control(make_spill_bitset_filter_ids());

    ASSERT_EQ(BitsetFilterStorageKind::MappedFileTemporary,
        bitset_filter_storage_kind_for_testing(control.get()));
    EXPECT_NE(nullptr, control->storage->data());
    EXPECT_NE(nullptr, control->storage->writable_data());
    EXPECT_GE(control->storage->fd, 0);
    EXPECT_EQ(visible_spill_files_before, count_visible_temporary_spill_files());

    control.reset();
    EXPECT_EQ(nullptr, control);
    EXPECT_EQ(visible_spill_files_before, count_visible_temporary_spill_files());
}

TEST(bitset_filter_control, named_large_filter_spills_to_persistent_named_file) {
    const std::string name = unique_filter_name("named_filter");
    const std::filesystem::path expected_path = named_bitset_filter_path(name);
    std::filesystem::remove(expected_path);

    BitsetFilterControlPtr control =
        create_control(make_spill_bitset_filter_ids(), name.c_str());
    ASSERT_EQ(BitsetFilterStorageKind::MappedFile,
        bitset_filter_storage_kind_for_testing(control.get()));

    EXPECT_TRUE(std::filesystem::exists(expected_path));
    EXPECT_GT(std::filesystem::file_size(expected_path), 0u);
    EXPECT_GE(control->storage->fd, 0);
    expect_persisted_filter_contains(expected_path, {20, 40});

    control.reset();
    EXPECT_EQ(nullptr, control);
    EXPECT_TRUE(std::filesystem::exists(expected_path));
    EXPECT_GT(std::filesystem::file_size(expected_path), 0u);
    expect_persisted_filter_contains(expected_path, {20, 40});

    std::filesystem::remove(expected_path);
}

TEST(bitset_filter_control, named_rebuild_does_not_truncate_existing_mapping) {
    const std::string name = unique_filter_name("rebuild_filter");
    const std::filesystem::path expected_path = named_bitset_filter_path(name);
    std::filesystem::remove(expected_path);

    BitsetFilterControlPtr first =
        create_control(make_spill_bitset_filter_ids({20, 40}), name.c_str());
    ASSERT_EQ(BitsetFilterStorageKind::MappedFile,
        bitset_filter_storage_kind_for_testing(first.get()));
    ASSERT_TRUE(std::filesystem::exists(expected_path));

    BitsetFilterControlPtr second =
        create_control(make_spill_bitset_filter_ids({30}), name.c_str());
    ASSERT_EQ(BitsetFilterStorageKind::MappedFile,
        bitset_filter_storage_kind_for_testing(second.get()));
    ASSERT_TRUE(std::filesystem::exists(expected_path));

    auto first_it = first->view.begin();
    EXPECT_TRUE(first_it.consume_if_equal(20));
    EXPECT_TRUE(first_it.consume_if_equal(40));
    EXPECT_FALSE(first_it.consume_if_equal(30));

    auto second_it = second->view.begin();
    EXPECT_FALSE(second_it.consume_if_equal(20));
    EXPECT_TRUE(second_it.consume_if_equal(30));
    EXPECT_FALSE(second_it.consume_if_equal(40));

    first.reset();
    second.reset();
    EXPECT_TRUE(std::filesystem::exists(expected_path));
    expect_persisted_filter_contains(expected_path, {30});
    std::filesystem::remove(expected_path);
}

TEST(bitset_filter_control, drop_named_filter_removes_file_and_reports_status) {
    const std::string name = unique_filter_name("drop_filter");
    const std::filesystem::path expected_path = named_bitset_filter_path(name);
    std::filesystem::remove(expected_path);

    BitsetFilterControlPtr control = create_control({20, 40}, name.c_str());
    ASSERT_NE(nullptr, control);
    ASSERT_TRUE(std::filesystem::exists(expected_path));
    control.reset();

    bool removed = false;
    EXPECT_EQ(0, drop_named_bitset_filter(name.c_str(), &removed).code());
    EXPECT_TRUE(removed);
    EXPECT_FALSE(std::filesystem::exists(expected_path));

    removed = true;
    EXPECT_EQ(0, drop_named_bitset_filter(name.c_str(), &removed).code());
    EXPECT_FALSE(removed);
}

TEST(bitset_filter_control, load_named_filter_reports_missing_and_invalid_names) {
    BitsetFilterControlPtr loaded;
    Ret ret = load_named_bitset_filter("", &loaded);
    EXPECT_NE(0, ret.code());
    EXPECT_NE(ret.message().find("bitset_load: invalid bitset filter name"), std::string::npos);
    EXPECT_EQ(nullptr, loaded);

    const std::string name = unique_filter_name("missing_filter");
    ret = load_named_bitset_filter(name.c_str(), &loaded);
    EXPECT_NE(0, ret.code());
    EXPECT_NE(ret.message().find("bitset_load: failed to open named bitset filter"),
        std::string::npos);
    EXPECT_EQ(nullptr, loaded);
}

TEST(bitset_filter_control, load_named_filter_rejects_truncated_file) {
    const std::string name = unique_filter_name("truncated_filter");
    const std::filesystem::path expected_path = named_bitset_filter_path(name);
    std::filesystem::remove(expected_path);

    {
        std::ofstream out(expected_path, std::ios::binary);
        ASSERT_TRUE(out.is_open());
        out.put('x');
    }

    BitsetFilterControlPtr loaded;
    const Ret ret = load_named_bitset_filter(name.c_str(), &loaded);
    EXPECT_NE(0, ret.code());
    EXPECT_NE(ret.message().find("bitset_load: malformed bitset filter"),
        std::string::npos);
    EXPECT_NE(ret.message().find("ChunkedBitsView::init_blob: blob is too small"),
        std::string::npos);
    EXPECT_EQ(nullptr, loaded);

    std::filesystem::remove(expected_path);
}

TEST(bitset_filter_control, rejects_empty_named_filter_name) {
    ChunkedBits bits;
    EXPECT_NE(0, bits.set_name("").code());

    bool removed = true;
    const Ret ret = drop_named_bitset_filter("", &removed);
    EXPECT_NE(0, ret.code());
    EXPECT_NE(ret.message().find("bitset_drop: invalid bitset filter name"), std::string::npos);
    EXPECT_FALSE(removed);
}

TEST(bitset_filter_control, create_preserves_finish_root_cause) {
    ChunkedBits bits;
    const Ret overflow("ChunkedBits::serialized_size_bytes: payload size overflow");
    ChunkedBitsTestPeer::mark_finish_failed(&bits, overflow);

    BitsetFilterControlPtr control;
    const Ret ret = BitsetFilterControl::create(bits, &control);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ(nullptr, control);
    EXPECT_EQ(
        "bitset filter builder: ChunkedBits::serialized_size_bytes: payload size overflow",
        ret.message());
}

TEST(bitset_filter_control, named_publish_failure_removes_temporary_file) {
    const std::string name = unique_filter_name("publish_failure_filter");
    const std::filesystem::path expected_path = named_bitset_filter_path(name);
    std::filesystem::remove_all(expected_path);
    std::filesystem::create_directory(expected_path);

    const size_t visible_temp_files_before = count_visible_named_temp_files(name);

    ChunkedBits bits;
    ASSERT_EQ(0, bits.set_name(name.c_str()).code());
    for (uint64_t id : make_spill_bitset_filter_ids()) {
        ASSERT_EQ(0, bits.add(id).code());
    }

    BitsetFilterControlPtr control;
    const Ret ret = BitsetFilterControl::create(bits, &control);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ(nullptr, control);
    EXPECT_EQ(visible_temp_files_before, count_visible_named_temp_files(name));
    EXPECT_TRUE(std::filesystem::is_directory(expected_path));

    std::filesystem::remove_all(expected_path);
}

TEST(bitset_filter_control, create_publishes_to_cache) {
    const std::string name = unique_filter_name("cache_create");
    const std::filesystem::path expected_path = named_bitset_filter_path(name);
    std::filesystem::remove(expected_path);
    bitset_file_cache().remove(name);

    BitsetFilterControlPtr control = create_control({20, 40}, name.c_str());
    ASSERT_NE(nullptr, control);

    EXPECT_TRUE(bitset_file_cache().contains(name));

    control.reset();
    bitset_file_cache().remove(name);
    std::filesystem::remove(expected_path);
}

TEST(bitset_filter_control, load_after_create_hits_cache) {
    const std::string name = unique_filter_name("cache_hit_filter");
    const std::filesystem::path expected_path = named_bitset_filter_path(name);
    std::filesystem::remove(expected_path);
    bitset_file_cache().remove(name);

    BitsetFilterControlPtr created = create_control({20, 40}, name.c_str());
    ASSERT_NE(nullptr, created);

    BitsetFilterControlPtr loaded;
    ASSERT_EQ(0, load_named_bitset_filter(name.c_str(), &loaded).code());
    ASSERT_NE(nullptr, loaded);
    // Cache hit produces an independent mapping — verify the data is correct,
    // and that no fd is owned by the borrower (the cache retains it).
    EXPECT_NE(created->storage.get(), loaded->storage.get());
    EXPECT_EQ(-1, loaded->storage->fd);
    EXPECT_TRUE(bitset_file_cache().contains(name));

    auto it = loaded->view.begin();
    EXPECT_TRUE(it.consume_if_equal(20));
    EXPECT_TRUE(it.consume_if_equal(40));

    created.reset();
    loaded.reset();
    bitset_file_cache().remove(name);
    std::filesystem::remove(expected_path);
}

TEST(bitset_filter_control, load_from_disk_publishes_to_cache) {
    const std::string name = unique_filter_name("cache_warm_filter");
    const std::filesystem::path expected_path = named_bitset_filter_path(name);
    std::filesystem::remove(expected_path);
    bitset_file_cache().remove(name);

    BitsetFilterControlPtr created = create_control({20, 40}, name.c_str());
    ASSERT_NE(nullptr, created);
    created.reset();
    bitset_file_cache().clear();

    BitsetFilterControlPtr loaded;
    ASSERT_EQ(0, load_named_bitset_filter(name.c_str(), &loaded).code());
    ASSERT_NE(nullptr, loaded);

    EXPECT_TRUE(bitset_file_cache().contains(name));

    loaded.reset();
    bitset_file_cache().remove(name);
    std::filesystem::remove(expected_path);
}

TEST(bitset_filter_control, drop_evicts_cache_entry) {
    const std::string name = unique_filter_name("cache_drop_filter");
    const std::filesystem::path expected_path = named_bitset_filter_path(name);
    std::filesystem::remove(expected_path);
    bitset_file_cache().remove(name);

    BitsetFilterControlPtr control = create_control({20, 40}, name.c_str());
    ASSERT_NE(nullptr, control);
    EXPECT_TRUE(bitset_file_cache().contains(name));
    control.reset();

    bool removed = false;
    EXPECT_EQ(0, drop_named_bitset_filter(name.c_str(), &removed).code());
    EXPECT_TRUE(removed);
    EXPECT_FALSE(bitset_file_cache().contains(name));
    EXPECT_FALSE(std::filesystem::exists(expected_path));
}

TEST(bitset_filter_control, borrower_keeps_mapping_after_drop) {
    const std::string name = unique_filter_name("borrower_filter");
    const std::filesystem::path expected_path = named_bitset_filter_path(name);
    std::filesystem::remove(expected_path);
    bitset_file_cache().remove(name);

    BitsetFilterControlPtr control = create_control({20, 40}, name.c_str());
    ASSERT_NE(nullptr, control);

    bool removed = false;
    EXPECT_EQ(0, drop_named_bitset_filter(name.c_str(), &removed).code());
    EXPECT_TRUE(removed);
    EXPECT_FALSE(bitset_file_cache().contains(name));

    // The borrower still holds the mapping; the data remains accessible.
    auto it = control->view.begin();
    EXPECT_TRUE(it.consume_if_equal(20));
    EXPECT_TRUE(it.consume_if_equal(40));

    control.reset();
}

TEST(bitset_filter_control, create_publishes_readonly_storage_to_cache) {
    const std::string name = unique_filter_name("readonly_cache_filter");
    const std::filesystem::path expected_path = named_bitset_filter_path(name);
    std::filesystem::remove(expected_path);
    bitset_file_cache().remove(name);

    BitsetFilterControlPtr control = create_control({20, 40}, name.c_str());
    ASSERT_NE(nullptr, control);

    // After create, the control's storage is read-only — no writable pointer
    // is exposed via writable_data().
    EXPECT_NE(nullptr, control->storage->data());
    EXPECT_EQ(nullptr, control->storage->writable_data());

    // A fresh acquire from the cache also produces a read-only mapping.
    std::unique_ptr<BitsetFilterStorage> cached;
    ASSERT_EQ(0, bitset_file_cache().acquire(name, &cached).code());
    ASSERT_NE(nullptr, cached);
    EXPECT_EQ(nullptr, cached->writable_data());

    control.reset();
    bitset_file_cache().remove(name);
    std::filesystem::remove(expected_path);
}

TEST(bitset_filter_control, named_rebuild_replaces_cache_entry) {
    const std::string name = unique_filter_name("rebuild_cache_filter");
    const std::filesystem::path expected_path = named_bitset_filter_path(name);
    std::filesystem::remove(expected_path);
    bitset_file_cache().remove(name);

    BitsetFilterControlPtr first = create_control({20, 40}, name.c_str());
    ASSERT_NE(nullptr, first);
    EXPECT_TRUE(bitset_file_cache().contains(name));

    BitsetFilterControlPtr second = create_control({30}, name.c_str());
    ASSERT_NE(nullptr, second);
    EXPECT_NE(first->storage.get(), second->storage.get());
    EXPECT_TRUE(bitset_file_cache().contains(name));

    // After rebuild, a fresh load reads the new payload.
    BitsetFilterControlPtr loaded;
    ASSERT_EQ(0, load_named_bitset_filter(name.c_str(), &loaded).code());
    ASSERT_NE(nullptr, loaded);
    auto loaded_it = loaded->view.begin();
    EXPECT_TRUE(loaded_it.consume_if_equal(30));
    EXPECT_FALSE(loaded_it.consume_if_equal(40));

    // The first borrower still sees its original mapping.
    auto first_it = first->view.begin();
    EXPECT_TRUE(first_it.consume_if_equal(20));
    EXPECT_TRUE(first_it.consume_if_equal(40));

    first.reset();
    second.reset();
    loaded.reset();
    bitset_file_cache().remove(name);
    std::filesystem::remove(expected_path);
}

} // namespace sketch2
