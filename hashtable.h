#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

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
        --size_;
        ++tombstones_;
        return true;
    }

    mapped_type* find(const key_type& key) {
        const size_type idx = find_index(key);
        return (idx == npos) ? nullptr : &(*buckets_[idx].value);
    }

    const mapped_type* find(const key_type& key) const {
        const size_type idx = find_index(key);
        return (idx == npos) ? nullptr : &(*buckets_[idx].value);
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

    void reserve(const size_type expected_size) {
        const double target = std::ceil(static_cast<double>(expected_size) / max_load_factor_);
        const size_type need_buckets = next_power_of_two(static_cast<size_type>(target < 16.0 ? 16.0 : target));
        if (need_buckets > buckets_.size()) rehash(need_buckets);
    }

    void rehash(size_type requested_bucket_count) {
        requested_bucket_count = requested_bucket_count < kInitialBucketCount ? kInitialBucketCount : requested_bucket_count;
        requested_bucket_count = next_power_of_two(requested_bucket_count);
        rehash_to_bucket_count(requested_bucket_count);
    }

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
        return static_cast<size_type>(mix64(static_cast<uint64_t>(hash_(key)))) & (buckets_.size() - 1);
    }

    static size_type probe_index(const size_type base, const size_type i, const size_type mask) {
        return (base + i + i * i) & mask;
    }

    void ensure_capacity_for_insert(size_type incoming) {
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
