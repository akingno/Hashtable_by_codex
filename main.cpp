#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// 需求细化（工程化版本）：
// A) 增加 reserve/rehash 接口，支持提前预留容量，减少插入阶段重哈希。
// B) 支持 max_load_factor 调整，默认 0.75，可在 0.1~0.95 区间设置。
// C) 删除仅打墓碑，不再重置 key/value（避免对复杂类型造成额外开销）。
// F) benchmark 改为多轮统计（平均值），并保证两边都 reserve，做更公平对比。
// 其他：
// - 泛型 Key/T（默认 Hash/KeyEqual），行为对齐 unordered_map 的常见使用习惯。
// - 桶数始终保持 2 的幂；初始桶数 16；二次探测；墓碑复用。

template <typename Key, typename T, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
class hashtable {
public:
    using key_type = Key;
    using mapped_type = T;
    using size_type = std::size_t;

    hashtable() : buckets_(kInitialBucketCount), size_(0), tombstones_(0), max_load_factor_(0.75f) {}

    bool insert(const key_type& key, const mapped_type& value) {
        ensure_capacity_for_insert(1);
        return insert_internal(key, value);
    }

    bool erase(const key_type& key) {
        const size_type idx = find_index(key);
        if (idx == npos) return false;

        Bucket& bucket = buckets_[idx];
        bucket.state = BucketState::Tombstone;
        // C: 不重置 key/value，避免无谓对象操作。
        --size_;
        ++tombstones_;
        return true;
    }

    mapped_type* find(const key_type& key) {
        const size_type idx = find_index(key);
        if (idx == npos) return nullptr;
        return &(*buckets_[idx].value);
    }

    const mapped_type* find(const key_type& key) const {
        const size_type idx = find_index(key);
        if (idx == npos) return nullptr;
        return &(*buckets_[idx].value);
    }

    bool contains(const key_type& key) const { return find_index(key) != npos; }

    mapped_type& operator[](const key_type& key) {
        if (mapped_type* ptr = find(key)) return *ptr;
        insert(key, mapped_type{});
        return *find(key);
    }

    void clear() {
        buckets_.assign(kInitialBucketCount, Bucket{});
        size_ = 0;
        tombstones_ = 0;
    }

    // A: 以元素数量为语义预留。
    void reserve(const size_type expected_size) {
        const double target = std::ceil(static_cast<double>(expected_size) / max_load_factor_);
        const size_type need_buckets = next_power_of_two(static_cast<size_type>(std::max(16.0, target)));
        if (need_buckets > buckets_.size()) rehash(need_buckets);
    }

    // A: 以桶数量为语义重哈希（会自动转成 2 的幂）。
    void rehash(size_type requested_bucket_count) {
        requested_bucket_count = std::max(kInitialBucketCount, next_power_of_two(requested_bucket_count));
        rehash_to_bucket_count(requested_bucket_count);
    }

    // B: 与 std::unordered_map 风格一致。
    void max_load_factor(float new_factor) {
        if (new_factor < 0.1f || new_factor > 0.95f) {
            throw std::invalid_argument("max_load_factor 必须在 [0.1, 0.95] 区间内");
        }
        max_load_factor_ = new_factor;
        ensure_capacity_for_insert(0);
    }

    float max_load_factor() const { return max_load_factor_; }

    size_type size() const { return size_; }
    size_type bucket_count() const { return buckets_.size(); }
    double load_factor() const {
        return buckets_.empty() ? 0.0 : static_cast<double>(size_) / static_cast<double>(buckets_.size());
    }

private:
    enum class BucketState : uint8_t { Empty, Occupied, Tombstone };

    struct Bucket {
        std::optional<key_type> key;
        std::optional<mapped_type> value;
        BucketState state = BucketState::Empty;
    };

    static constexpr size_type kInitialBucketCount = 16;
    static constexpr size_type npos = static_cast<size_type>(-1);

    std::vector<Bucket> buckets_;
    size_type size_;
    size_type tombstones_;
    float max_load_factor_;
    Hash hash_;
    KeyEqual eq_;

    static uint64_t mix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    static size_type next_power_of_two(size_type n) {
        if (n <= 1) return 1;
        --n;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        if constexpr (sizeof(size_type) >= 8) n |= n >> 32;
        return n + 1;
    }

    size_type hashed_base(const key_type& key) const {
        const uint64_t raw = static_cast<uint64_t>(hash_(key));
        return static_cast<size_type>(mix64(raw)) & (buckets_.size() - 1);
    }

    static size_type probe_index(const size_type base, const size_type i, const size_type mask) {
        return (base + i + i * i) & mask;
    }

    void ensure_capacity_for_insert(size_type incoming) {
        if (buckets_.empty()) {
            buckets_.assign(kInitialBucketCount, Bucket{});
        }
        const double used_ratio = static_cast<double>(size_ + tombstones_ + incoming) /
                                  static_cast<double>(buckets_.size());
        if (used_ratio > max_load_factor_) {
            rehash_to_bucket_count(buckets_.size() * 2);
        }
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
            if (eq_(*bucket.key, key)) {
                *bucket.value = value;
                return false;
            }
        }

        rehash_to_bucket_count(buckets_.size() * 2);
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
        if (buckets_.empty()) return npos;
        const size_type mask = buckets_.size() - 1;
        const size_type base = hashed_base(key);

        for (size_type i = 0; i < buckets_.size(); ++i) {
            const size_type idx = probe_index(base, i, mask);
            const Bucket& bucket = buckets_[idx];
            if (bucket.state == BucketState::Empty) return npos;
            if (bucket.state == BucketState::Occupied && eq_(*bucket.key, key)) return idx;
        }
        return npos;
    }

    void rehash_to_bucket_count(const size_type new_bucket_count) {
        std::vector<Bucket> old = std::move(buckets_);
        buckets_.assign(new_bucket_count, Bucket{});
        size_ = 0;
        tombstones_ = 0;

        for (Bucket& b : old) {
            if (b.state == BucketState::Occupied) {
                insert_internal(*b.key, *b.value);
            }
        }
    }
};

static void run_functional_tests() {
    hashtable<int, std::string> t1;
    assert(t1.insert(1, "one"));
    assert(!t1.insert(1, "ONE"));
    assert(t1.contains(1));
    assert(*t1.find(1) == "ONE");
    assert(t1.erase(1));
    assert(!t1.contains(1));

    hashtable<std::string, int> t2;
    t2.max_load_factor(0.8f);
    t2.reserve(5000);
    t2["apple"] += 7;
    assert(t2.contains("apple"));
    assert(*t2.find("apple") == 7);

    for (int i = 0; i < 5000; ++i) t2.insert("k" + std::to_string(i), i);
    for (int i = 0; i < 5000; ++i) {
        const int* v = t2.find("k" + std::to_string(i));
        assert(v != nullptr && *v == i);
    }

    assert(t2.size() >= 5001);
    assert(t2.load_factor() <= t2.max_load_factor());

    std::cout << "[TEST] 工程化功能测试全部通过。\n";
}

struct BenchResult {
    long long insert_us_avg = 0;
    long long find_us_avg = 0;
};

template <typename F>
static BenchResult run_multi_round_bench(F&& fn, int rounds) {
    std::vector<long long> insert_costs;
    std::vector<long long> find_costs;
    insert_costs.reserve(rounds);
    find_costs.reserve(rounds);

    for (int r = 0; r < rounds; ++r) {
        auto [ins, fnd] = fn();
        insert_costs.push_back(ins);
        find_costs.push_back(fnd);
    }

    BenchResult res;
    res.insert_us_avg = std::accumulate(insert_costs.begin(), insert_costs.end(), 0LL) / rounds;
    res.find_us_avg = std::accumulate(find_costs.begin(), find_costs.end(), 0LL) / rounds;
    return res;
}

static void run_benchmark() {
    constexpr int kCount = 300000;
    constexpr int kRounds = 5;

    std::vector<int> keys;
    keys.reserve(kCount);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(1, 1'000'000'000);
    for (int i = 0; i < kCount; ++i) keys.push_back(dist(rng));

    using Clock = std::chrono::steady_clock;

    BenchResult custom = run_multi_round_bench(
        [&]() -> std::pair<long long, long long> {
            hashtable<int, int> table;
            table.max_load_factor(0.8f);
            table.reserve(kCount);  // F: 公平起见，和 std::unordered_map 都预留容量。

            auto t1 = Clock::now();
            for (int i = 0; i < kCount; ++i) table.insert(keys[i], i);
            auto t2 = Clock::now();
            const long long ins = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

            t1 = Clock::now();
            volatile int sum = 0;
            for (int i = 0; i < kCount; ++i) {
                const int* v = table.find(keys[i]);
                if (v) sum += *v;
            }
            t2 = Clock::now();
            const long long fnd = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            return {ins, fnd};
        },
        kRounds);

    BenchResult stl = run_multi_round_bench(
        [&]() -> std::pair<long long, long long> {
            std::unordered_map<int, int> table;
            table.reserve(kCount);

            auto t1 = Clock::now();
            for (int i = 0; i < kCount; ++i) table[keys[i]] = i;
            auto t2 = Clock::now();
            const long long ins = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

            t1 = Clock::now();
            volatile int sum = 0;
            for (int i = 0; i < kCount; ++i) {
                auto it = table.find(keys[i]);
                if (it != table.end()) sum += it->second;
            }
            t2 = Clock::now();
            const long long fnd = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            return {ins, fnd};
        },
        kRounds);

    std::cout << "\n[Benchmark] 数据量: " << kCount << ", 轮数: " << kRounds << " (平均值)\n";
    std::cout << std::left << std::setw(24) << "容器" << std::setw(16) << "插入(us)"
              << std::setw(16) << "查找(us)" << '\n';
    std::cout << std::left << std::setw(24) << "自定义 hashtable" << std::setw(16)
              << custom.insert_us_avg << std::setw(16) << custom.find_us_avg << '\n';
    std::cout << std::left << std::setw(24) << "std::unordered_map" << std::setw(16)
              << stl.insert_us_avg << std::setw(16) << stl.find_us_avg << '\n';
}

int main() {
    run_functional_tests();
    run_benchmark();
    return 0;
}
