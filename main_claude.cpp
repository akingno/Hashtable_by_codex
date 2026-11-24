// =============================================================================
//  claude::hashtable 的功能测试 + 三方性能对比
//    对比对象：原版 hashtable（hashtable.h）、claude::hashtable、std::unordered_map
//  编译：g++ -std=c++17 -O2 -march=native main_claude.cpp -o bench_claude
// =============================================================================
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hashtable.h"         // 原版（codex 实现）
#include "hashtable_claude.h"  // 优化版（claude 实现）

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// --------------------------------------------------------------------------
// 功能测试
// --------------------------------------------------------------------------
static void test_basic() {
    claude::hashtable<int, std::string> t;
    assert(t.size() == 0 && t.empty());
    assert(t.insert(1, "one"));
    assert(!t.insert(1, "ONE"));      // 已存在 -> 覆盖
    assert(*t.find(1) == "ONE");
    assert(t.contains(1) && t.count(1) == 1);
    assert(t.find(2) == nullptr);
    assert(t.erase(1));
    assert(!t.erase(1));
    assert(t.empty() && t.find(1) == nullptr);

    claude::hashtable<std::string, int> s;
    s["apple"] += 7;
    assert(*s.find("apple") == 7);
    s["apple"] = 9;
    assert(s.at("apple") == 9);
    bool threw = false;
    try { s.at("pear"); } catch (const std::out_of_range&) { threw = true; }
    assert(threw);

    auto r = s.try_emplace("pear", 3);
    assert(r.second && *r.first == 3);
    r = s.try_emplace("pear", 100);
    assert(!r.second && *r.first == 3);  // 不覆盖

    std::cout << "[TEST] 基础接口通过\n";
}

static void test_growth_and_iteration() {
    claude::hashtable<int, int> t;
    constexpr int kN = 100000;
    for (int i = 0; i < kN; ++i) assert(t.insert(i, i * 3));
    assert(t.size() == static_cast<size_t>(kN));
    for (int i = 0; i < kN; ++i) assert(*t.find(i) == i * 3);
    assert(t.load_factor() <= 0.875 + 1e-9);

    long long sum = 0;
    size_t cnt = 0;
    for (const auto& kv : t) { sum += kv.second; ++cnt; }
    assert(cnt == static_cast<size_t>(kN));
    long long expect = 0;
    for (int i = 0; i < kN; ++i) expect += static_cast<long long>(i) * 3;
    assert(sum == expect);

    // 拷贝 / 移动 / swap
    claude::hashtable<int, int> copy = t;
    assert(copy.size() == t.size() && *copy.find(777) == 777 * 3);
    claude::hashtable<int, int> moved = std::move(copy);
    assert(moved.size() == static_cast<size_t>(kN));
    claude::hashtable<int, int> other;
    other.swap(moved);
    assert(other.size() == static_cast<size_t>(kN) && moved.empty());

    t.clear();
    assert(t.empty() && t.begin() == t.end());
    for (int i = 0; i < 1000; ++i) assert(t.insert(i, i));
    assert(t.size() == 1000);

    std::cout << "[TEST] 扩容/迭代/拷贝移动通过\n";
}

// 与 std::unordered_map 做随机对照（插入、覆盖、删除、查找混合）
static void test_fuzz_against_stdmap() {
    std::mt19937_64 rng(20261124);
    std::unordered_map<int, int> ref;
    claude::hashtable<int, int> mine;

    for (int step = 0; step < 400000; ++step) {
        const int key = static_cast<int>(rng() % 5000);
        const int op = static_cast<int>(rng() % 10);
        if (op < 5) {  // 插入/覆盖
            const int val = static_cast<int>(rng());
            const bool fresh_ref = ref.find(key) == ref.end();
            const bool fresh_mine = mine.insert(key, val);
            assert(fresh_ref == fresh_mine);
            ref[key] = val;
        } else if (op < 7) {  // operator[]
            mine[key] += 1;
            ref[key] += 1;
        } else if (op < 9) {  // 删除
            const bool a = ref.erase(key) != 0;
            const bool b = mine.erase(key);
            assert(a == b);
        } else {  // 查找
            const auto it = ref.find(key);
            const int* p = mine.find(key);
            assert((it == ref.end()) == (p == nullptr));
            if (p) assert(*p == it->second);
        }
        assert(mine.size() == ref.size());
    }

    // 全量一致性 + 迭代一致性
    for (const auto& kv : ref) {
        const int* p = mine.find(kv.first);
        assert(p && *p == kv.second);
    }
    size_t seen = 0;
    for (const auto& kv : mine) {
        auto it = ref.find(kv.first);
        assert(it != ref.end() && it->second == kv.second);
        ++seen;
    }
    assert(seen == ref.size());
    std::cout << "[TEST] 40 万步随机对照（vs std::unordered_map）通过，最终 size = "
              << mine.size() << "\n";
}

// 字符串 key 的随机对照
static void test_fuzz_string() {
    std::mt19937_64 rng(7);
    std::unordered_map<std::string, int> ref;
    claude::hashtable<std::string, int> mine;
    for (int step = 0; step < 200000; ++step) {
        const std::string key = "key_" + std::to_string(rng() % 3000);
        const int op = static_cast<int>(rng() % 10);
        if (op < 6) {
            const int val = static_cast<int>(rng() % 1000);
            mine.insert(key, val);
            ref[key] = val;
        } else if (op < 8) {
            mine.erase(key);
            ref.erase(key);
        } else {
            const int* p = mine.find(key);
            const auto it = ref.find(key);
            assert((it == ref.end()) == (p == nullptr));
            if (p) assert(*p == it->second);
        }
        assert(mine.size() == ref.size());
    }
    std::cout << "[TEST] 字符串 key 随机对照通过，最终 size = " << mine.size() << "\n";
}

// 插入/删除长期churn：验证“少留墓碑 + 按原容量重建”能把桶数稳住，
// 而不是像原版那样被墓碑不断推着扩容。
static void test_churn_is_bounded() {
    claude::hashtable<int, int> t;
    const size_t kLive = 1000;
    for (size_t i = 0; i < kLive; ++i) t.insert(static_cast<int>(i), 0);
    for (int step = 0; step < 500000; ++step) {
        const int old_key = step % static_cast<int>(kLive);
        t.erase(old_key);
        t.insert(old_key, step);
        assert(t.size() == kLive);
    }
    assert(t.bucket_count() <= 4 * kLive);  // 不应无界膨胀
    for (size_t i = 0; i < kLive; ++i) assert(t.find(static_cast<int>(i)) != nullptr);
    std::cout << "[TEST] 50 万次插删 churn 后桶数稳定在 " << t.bucket_count() << "\n";
}

// 非平凡类型：构造/析构必须严格配对（配合 ASan 能查出泄漏与重复释放）
struct Counted {
    static int alive;
    int v = 0;
    Counted() { ++alive; }
    explicit Counted(int x) : v(x) { ++alive; }
    Counted(const Counted& o) : v(o.v) { ++alive; }
    Counted(Counted&& o) noexcept : v(o.v) { ++alive; }
    Counted& operator=(const Counted&) = default;
    Counted& operator=(Counted&&) noexcept = default;
    ~Counted() { --alive; }
};
int Counted::alive = 0;

static void test_nontrivial_lifetime() {
    {
        claude::hashtable<std::string, Counted> t;
        for (int i = 0; i < 20000; ++i) {
            t.insert("k" + std::to_string(i), Counted(i));
        }
        assert(t.size() == 20000);
        for (int i = 0; i < 20000; i += 3) assert(t.erase("k" + std::to_string(i)));
        for (int i = 0; i < 20000; ++i) {
            const Counted* p = t.find("k" + std::to_string(i));
            if (i % 3 == 0) assert(p == nullptr);
            else assert(p && p->v == i);
        }
        t.clear();
        assert(Counted::alive == 0);
        for (int i = 0; i < 5000; ++i) t.insert("x" + std::to_string(i), Counted(i));
    }
    assert(Counted::alive == 0);
    std::cout << "[TEST] 非平凡类型生命周期（构造/析构配对）通过\n";
}

// --------------------------------------------------------------------------
// 性能对比
// --------------------------------------------------------------------------
struct Row {
    std::string name;
    double insert_ms = 0, insert_reserved_ms = 0, find_hit_ms = 0, find_miss_ms = 0,
           erase_ms = 0, iterate_ms = 0;
};

static void print_table(const std::string& title, const std::vector<Row>& rows) {
    std::cout << "\n=== " << title << " (单位: ms，越小越快) ===\n";
    std::cout << std::left << std::setw(24) << "容器" << std::right << std::setw(12) << "插入"
              << std::setw(14) << "插入(预留)" << std::setw(12) << "命中查找" << std::setw(12)
              << "未命中查找" << std::setw(12) << "删除" << std::setw(12) << "遍历" << '\n';
    std::cout << std::string(98, '-') << '\n';
    std::cout << std::fixed << std::setprecision(2);
    for (const Row& r : rows) {
        std::cout << std::left << std::setw(24) << r.name << std::right << std::setw(12)
                  << r.insert_ms << std::setw(14) << r.insert_reserved_ms << std::setw(12)
                  << r.find_hit_ms << std::setw(12) << r.find_miss_ms << std::setw(12)
                  << r.erase_ms << std::setw(12) << r.iterate_ms << '\n';
    }
    if (rows.size() >= 3) {
        const Row& base = rows[2];  // std::unordered_map
        const Row& mine = rows[1];  // claude
        std::cout << "\n相对 std::unordered_map 的加速比（claude，>1 表示更快）：\n"
                  << "  插入 x" << base.insert_ms / mine.insert_ms
                  << "   插入(预留) x" << base.insert_reserved_ms / mine.insert_reserved_ms
                  << "   命中查找 x" << base.find_hit_ms / mine.find_hit_ms
                  << "   未命中查找 x" << base.find_miss_ms / mine.find_miss_ms
                  << "   删除 x" << base.erase_ms / mine.erase_ms
                  << "   遍历 x" << base.iterate_ms / mine.iterate_ms << '\n';
    }
}

template <typename Map, bool HasIterate, typename Key>
static Row bench_map(const std::string& name,
                     const std::vector<Key>& keys,
                     const std::vector<Key>& missing) {
    Row row;
    row.name = name;
    volatile long long sink = 0;

    // 插入（不预留）
    {
        Map m;
        auto t0 = Clock::now();
        for (size_t i = 0; i < keys.size(); ++i) m.insert(keys[i], static_cast<int>(i));
        row.insert_ms = ms_since(t0);
        sink += static_cast<long long>(m.size());
    }

    Map m;
    m.reserve(keys.size());
    auto t0 = Clock::now();
    for (size_t i = 0; i < keys.size(); ++i) m.insert(keys[i], static_cast<int>(i));
    row.insert_reserved_ms = ms_since(t0);

    // 命中查找
    t0 = Clock::now();
    for (size_t i = 0; i < keys.size(); ++i) {
        const int* p = m.find(keys[i]);
        if (p) sink += *p;
    }
    row.find_hit_ms = ms_since(t0);

    // 未命中查找
    t0 = Clock::now();
    for (size_t i = 0; i < missing.size(); ++i) {
        const int* p = m.find(missing[i]);
        if (p) sink += *p;
    }
    row.find_miss_ms = ms_since(t0);

    // 遍历
    if constexpr (HasIterate) {
        t0 = Clock::now();
        long long s = 0;
        for (auto it = m.begin(); it != m.end(); ++it) s += it->second;
        row.iterate_ms = ms_since(t0);
        sink += s;
    }

    // 删除（删一半）
    t0 = Clock::now();
    for (size_t i = 0; i < keys.size(); i += 2) m.erase(keys[i]);
    row.erase_ms = ms_since(t0);
    sink += static_cast<long long>(m.size());

    (void)sink;
    return row;
}

// 给 std::unordered_map 套一层同形接口，保证三者测的是同一段代码
template <typename K>
struct StdMapAdapter {
    std::unordered_map<K, int> m;
    void reserve(size_t n) { m.reserve(n); }
    bool insert(const K& k, int v) {
        auto r = m.emplace(k, v);
        if (!r.second) r.first->second = v;
        return r.second;
    }
    const int* find(const K& k) const {
        auto it = m.find(k);
        return it == m.end() ? nullptr : &it->second;
    }
    bool erase(const K& k) { return m.erase(k) != 0; }
    size_t size() const { return m.size(); }
    typename std::unordered_map<K, int>::const_iterator begin() const { return m.begin(); }
    typename std::unordered_map<K, int>::const_iterator end() const { return m.end(); }
};

// 原版 hashtable 没有迭代器，这里只跑其余项
template <typename K>
struct CodexAdapter {
    hashtable<K, int> m;
    void reserve(size_t n) { m.reserve(n); }
    bool insert(const K& k, int v) { return m.insert(k, v); }
    const int* find(const K& k) const { return m.find(k); }
    bool erase(const K& k) { return m.erase(k); }
    size_t size() const { return m.size(); }
};

template <typename K>
struct ClaudeAdapter {
    claude::hashtable<K, int> m;
    void reserve(size_t n) { m.reserve(n); }
    bool insert(const K& k, int v) { return m.insert(k, v); }
    const int* find(const K& k) const { return m.find(k); }
    bool erase(const K& k) { return m.erase(k); }
    size_t size() const { return m.size(); }
    typename claude::hashtable<K, int>::const_iterator begin() const { return m.begin(); }
    typename claude::hashtable<K, int>::const_iterator end() const { return m.end(); }
};

static void bench_int(size_t n) {
    std::mt19937_64 rng(42);
    std::vector<int64_t> keys, missing;
    keys.reserve(n);
    missing.reserve(n);
    std::unordered_set<int64_t> used;
    used.reserve(n * 2);
    while (keys.size() < n) {
        const int64_t k = static_cast<int64_t>(rng() % 1000000007ull);
        if (used.insert(k).second) keys.push_back(k);
    }
    while (missing.size() < n) {
        const int64_t k = static_cast<int64_t>(rng() % 1000000007ull) + 2000000000ll;
        if (used.insert(k).second) missing.push_back(k);
    }

    std::vector<Row> rows;
    rows.push_back(bench_map<CodexAdapter<int64_t>, false>("hashtable (原版 codex)", keys, missing));
    rows.push_back(bench_map<ClaudeAdapter<int64_t>, true>("claude::hashtable", keys, missing));
    rows.push_back(bench_map<StdMapAdapter<int64_t>, true>("std::unordered_map", keys, missing));
    print_table("int64 -> int，元素数 " + std::to_string(n), rows);
}

static void bench_string(size_t n) {
    std::mt19937_64 rng(1234);
    std::vector<std::string> keys, missing;
    keys.reserve(n);
    missing.reserve(n);
    for (size_t i = 0; i < n; ++i) keys.push_back("user_token_" + std::to_string(rng()));
    for (size_t i = 0; i < n; ++i) missing.push_back("absent_key_" + std::to_string(rng()));

    std::vector<Row> rows;
    rows.push_back(bench_map<CodexAdapter<std::string>, false>("hashtable (原版 codex)", keys, missing));
    rows.push_back(bench_map<ClaudeAdapter<std::string>, true>("claude::hashtable", keys, missing));
    rows.push_back(bench_map<StdMapAdapter<std::string>, true>("std::unordered_map", keys, missing));
    print_table("std::string -> int，元素数 " + std::to_string(n), rows);
}

int main() {
    std::cout << "SIMD 探测: " << (CLAUDE_HASHTABLE_SSE2 ? "SSE2 (16 路)" : "SWAR (8 路)")
              << "\n\n";
    test_basic();
    test_growth_and_iteration();
    test_fuzz_against_stdmap();
    test_fuzz_string();
    test_churn_is_bounded();
    test_nontrivial_lifetime();

    bench_int(1000000);
    bench_int(100000);
    bench_string(300000);
    std::cout << "\n注：原版 hashtable 没有迭代器，遍历一列记为 0。\n";
    return 0;
}
