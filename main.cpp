#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hashtable.h"

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

    std::cout << "[TEST] 功能测试通过。\n";
}

struct BenchResult { long long ins = 0, find = 0; };

template <typename Fn>
BenchResult multi_round(Fn&& fn, int rounds) {
    std::vector<long long> ins, fnd;
    for (int i = 0; i < rounds; ++i) {
        auto [a, b] = fn();
        ins.push_back(a);
        fnd.push_back(b);
    }
    return {std::accumulate(ins.begin(), ins.end(), 0LL) / rounds,
            std::accumulate(fnd.begin(), fnd.end(), 0LL) / rounds};
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

    auto custom = multi_round([&]() {
        hashtable<int, int> table;
        table.max_load_factor(0.8f);
        table.reserve(kCount);

        auto t1 = Clock::now();
        for (int i = 0; i < kCount; ++i) table.insert(keys[i], i);
        auto t2 = Clock::now();
        auto ins = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        volatile int sink = 0;
        t1 = Clock::now();
        for (int i = 0; i < kCount; ++i) {
            if (auto* v = table.find(keys[i])) sink += *v;
        }
        t2 = Clock::now();
        auto fnd = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        return std::pair<long long, long long>{ins, fnd};
    }, kRounds);

    auto stl = multi_round([&]() {
        std::unordered_map<int, int> table;
        table.reserve(kCount);

        auto t1 = Clock::now();
        for (int i = 0; i < kCount; ++i) table[keys[i]] = i;
        auto t2 = Clock::now();
        auto ins = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        volatile int sink = 0;
        t1 = Clock::now();
        for (int i = 0; i < kCount; ++i) {
            auto it = table.find(keys[i]);
            if (it != table.end()) sink += it->second;
        }
        t2 = Clock::now();
        auto fnd = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        return std::pair<long long, long long>{ins, fnd};
    }, kRounds);

    std::cout << "[Benchmark] N=" << kCount << ", rounds=" << kRounds << "\n";
    std::cout << std::left << std::setw(22) << "container" << std::setw(14) << "insert(us)" << "find(us)\n";
    std::cout << std::left << std::setw(22) << "custom hashtable" << std::setw(14) << custom.ins << custom.find << "\n";
    std::cout << std::left << std::setw(22) << "std::unordered_map" << std::setw(14) << stl.ins << stl.find << "\n";
}

int main() {
    run_functional_tests();
    run_benchmark();
}
