#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// 需求细化：
// 1) hashtable 与 std::unordered_map 一样支持“可哈希的任意 key 类型”和“任意 value 类型”。
// 2) 采用开放寻址 + 二次探测：index = (h + i + i*i) & (bucket_count - 1)，桶数保持 2 的幂。
// 3) 初始桶数 16，扩容阈值 0.5，超过阈值后桶数 * 2。
// 4) 删除使用墓碑，避免探测链断裂；插入时优先复用墓碑位置。
// 5) 对哈希值做 64 位混合扰动，降低聚集风险。

template <typename Key,
          typename T,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class hashtable {
public:
    using key_type = Key;
    using mapped_type = T;
    using size_type = std::size_t;

    hashtable() : buckets_(kInitialBucketCount), size_(0), tombstones_(0), hash_(), eq_() {
        static_assert(std::is_default_constructible_v<Hash>, "Hash 必须可默认构造");
        static_assert(std::is_default_constructible_v<KeyEqual>, "KeyEqual 必须可默认构造");
    }

    bool insert(const key_type& key, const mapped_type& value) {
        ensure_capacity_for_insert();
        return insert_internal(key, value);
    }

    bool erase(const key_type& key) {
        const size_type idx = find_index(key);
        if (idx == npos) return false;

        Bucket& bucket = buckets_[idx];
        bucket.state = BucketState::Tombstone;
        bucket.key = key_type{};
        bucket.value = mapped_type{};
        --size_;
        ++tombstones_;
        return true;
    }

    mapped_type* find(const key_type& key) {
        const size_type idx = find_index(key);
        return (idx == npos) ? nullptr : &buckets_[idx].value;
    }

    const mapped_type* find(const key_type& key) const {
        const size_type idx = find_index(key);
        return (idx == npos) ? nullptr : &buckets_[idx].value;
    }

    bool contains(const key_type& key) const { return find_index(key) != npos; }

    mapped_type& operator[](const key_type& key) {
        if (mapped_type* v = find(key)) return *v;
        insert(key, mapped_type{});
        return *find(key);
    }

    size_type size() const { return size_; }
    size_type bucket_count() const { return buckets_.size(); }

private:
    enum class BucketState : uint8_t { Empty, Occupied, Tombstone };

    struct Bucket {
        key_type key{};
        mapped_type value{};
        BucketState state = BucketState::Empty;
    };

    static constexpr size_type kInitialBucketCount = 16;
    static constexpr double kMaxLoadFactor = 0.5;
    static constexpr size_type npos = static_cast<size_type>(-1);

    std::vector<Bucket> buckets_;
    size_type size_;
    size_type tombstones_;
    Hash hash_;
    KeyEqual eq_;

    static uint64_t mix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    size_type hashed_base(const key_type& key) const {
        const uint64_t raw = static_cast<uint64_t>(hash_(key));
        return static_cast<size_type>(mix64(raw)) & (buckets_.size() - 1);
    }

    static size_type probe_index(const size_type base, const size_type i, const size_type mask) {
        return (base + i + i * i) & mask;
    }

    void ensure_capacity_for_insert() {
        const double used_ratio = static_cast<double>(size_ + tombstones_ + 1) /
                                  static_cast<double>(buckets_.size());
        if (used_ratio > kMaxLoadFactor) rehash(buckets_.size() * 2);
    }

    bool insert_internal(const key_type& key, const mapped_type& value) {
        size_type first_tombstone = npos;
        const size_type mask = buckets_.size() - 1;
        const size_type base = hashed_base(key);

        for (size_type i = 0; i < buckets_.size(); ++i) {
            const size_type idx = probe_index(base, i, mask);
            Bucket& bucket = buckets_[idx];

            if (bucket.state == BucketState::Empty) {
                const size_type target = (first_tombstone == npos) ? idx : first_tombstone;
                place_new_bucket(target, key, value);
                return true;
            }

            if (bucket.state == BucketState::Tombstone) {
                if (first_tombstone == npos) first_tombstone = idx;
                continue;
            }

            if (eq_(bucket.key, key)) {
                bucket.value = value;
                return false;
            }
        }

        rehash(buckets_.size() * 2);
        return insert_internal(key, value);
    }

    void place_new_bucket(const size_type idx, const key_type& key, const mapped_type& value) {
        Bucket& bucket = buckets_[idx];
        if (bucket.state == BucketState::Tombstone) --tombstones_;
        bucket.key = key;
        bucket.value = value;
        bucket.state = BucketState::Occupied;
        ++size_;
    }

    size_type find_index(const key_type& key) const {
        const size_type mask = buckets_.size() - 1;
        const size_type base = hashed_base(key);

        for (size_type i = 0; i < buckets_.size(); ++i) {
            const size_type idx = probe_index(base, i, mask);
            const Bucket& bucket = buckets_[idx];

            if (bucket.state == BucketState::Empty) return npos;
            if (bucket.state == BucketState::Occupied && eq_(bucket.key, key)) return idx;
        }
        return npos;
    }

    void rehash(const size_type new_bucket_count) {
        std::vector<Bucket> old = std::move(buckets_);
        buckets_.assign(new_bucket_count, Bucket{});
        size_ = 0;
        tombstones_ = 0;

        for (const Bucket& b : old) {
            if (b.state == BucketState::Occupied) {
                insert_internal(b.key, b.value);
            }
        }
    }
};

static void run_functional_tests() {
    // 测试 1：int -> std::string
    hashtable<int, std::string> t1;
    assert(t1.insert(1, "one"));
    assert(!t1.insert(1, "ONE"));
    assert(t1.contains(1));
    assert(*t1.find(1) == "ONE");
    assert(t1.erase(1));
    assert(!t1.contains(1));

    // 测试 2：std::string -> int
    hashtable<std::string, int> t2;
    t2["apple"] += 7;
    assert(t2.contains("apple"));
    assert(*t2.find("apple") == 7);

    for (int i = 0; i < 5000; ++i) {
        t2.insert("k" + std::to_string(i), i);
    }
    for (int i = 0; i < 5000; ++i) {
        const int* v = t2.find("k" + std::to_string(i));
        assert(v != nullptr && *v == i);
    }

    std::cout << "[TEST] 泛型功能测试全部通过。\n";
}

static void run_benchmark() {
    constexpr int kCount = 200000;
    std::vector<int> keys;
    keys.reserve(kCount);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(1, 1'000'000'000);
    for (int i = 0; i < kCount; ++i) keys.push_back(dist(rng));

    using Clock = std::chrono::steady_clock;

    hashtable<int, int> custom;
    auto t1 = Clock::now();
    for (int i = 0; i < kCount; ++i) custom.insert(keys[i], i);
    auto t2 = Clock::now();
    const long long custom_insert_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    t1 = Clock::now();
    volatile int sum1 = 0;
    for (int i = 0; i < kCount; ++i) {
        const int* v = custom.find(keys[i]);
        if (v) sum1 += *v;
    }
    t2 = Clock::now();
    const long long custom_find_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::unordered_map<int, int> stdmap;
    stdmap.reserve(kCount);
    t1 = Clock::now();
    for (int i = 0; i < kCount; ++i) stdmap[keys[i]] = i;
    t2 = Clock::now();
    const long long std_insert_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    t1 = Clock::now();
    volatile int sum2 = 0;
    for (int i = 0; i < kCount; ++i) {
        auto it = stdmap.find(keys[i]);
        if (it != stdmap.end()) sum2 += it->second;
    }
    t2 = Clock::now();
    const long long std_find_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "\n[Benchmark] 数据量: " << kCount << '\n';
    std::cout << std::left << std::setw(24) << "容器" << std::setw(16) << "插入(us)"
              << std::setw(16) << "查找(us)" << '\n';
    std::cout << std::left << std::setw(24) << "自定义 hashtable" << std::setw(16)
              << custom_insert_us << std::setw(16) << custom_find_us << '\n';
    std::cout << std::left << std::setw(24) << "std::unordered_map" << std::setw(16)
              << std_insert_us << std::setw(16) << std_find_us << '\n';

    if (sum1 == sum2) {
        std::cout << "[Benchmark] 校验通过。\n";
    }
}

int main() {
    run_functional_tests();
    run_benchmark();
    return 0;
}
