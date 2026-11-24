#pragma once
// =============================================================================
//  claude::hashtable —— 在 hashtable.h 基础上复制改写的高性能开放寻址哈希表
// -----------------------------------------------------------------------------
//  与原版 hashtable.h 的差异（都是为了更快）：
//
//  1) 元数据与数据分离（SoA）。每个槽位的状态压缩成 1 个控制字节，
//     与 key/value 分开存放。原版 Bucket 内含两个 std::optional，
//     各自带 bool 标志与对齐填充（<int,int> 时 Bucket 要 24 字节），
//     探测一次就要碰一条新的 cache line。现在一条 64 字节 cache line
//     能装 64 个控制字节，探测几乎不产生额外 cache miss。
//
//  2) SIMD 并行探测。控制字节里存放哈希值的低 7 位作为“指纹”(H2)。
//     一条 SSE2 指令同时比较 16 个槽位的指纹，得到候选位掩码；
//     指纹不匹配的槽位永远不会去读 key，于是把原版
//     “逐槽位循环 + 逐次 KeyEqual + 随机访问 key”压成
//     “一次 16 路比较 + 极少量真实 key 比较”。
//     没有 SSE2 的平台退化为 8 路 SWAR（纯 64 位整数位运算）。
//
//  3) 探测序列改为“按组二次探测”：pos += kWidth * 三角数。
//     仍是二次探测（保留原始设计要求，且在 2 的幂容量下能遍历全表），
//     但每步推进一整组，组内 16 个槽位连续，对硬件预取极友好。
//
//  4) 负载因子由 0.5 / 0.75 提高到 7/8。指纹过滤让高负载下探测依然廉价，
//     内存更小、cache 命中更高、扩容次数更少。
//
//  5) 更便宜的哈希扰动。原版 mix64 是 3 次乘法加若干移位；
//     这里用一次 64x64→128 位乘法的异或折叠，雪崩性质够用而延迟更低。
//
//  6) 墓碑更少。删除时若能证明“没有任何探测窗口曾把该槽位当成满组跳过”
//     （前后两组都存在空槽，且左右连续非空长度之和 < 组宽），
//     就直接写回 Empty 并归还一个增长配额。原版每次删除必留墓碑，
//     删除密集的场景会不断触发毫无意义的扩容。
//     此外墓碑过多而元素很少时，按原容量重建（而不是翻倍）即可清理。
//
//  7) 原地构造 + 移动重散列。槽位是裸内存 + placement new；
//     扩容时移动元素而非拷贝，且因为 key 互不相同，
//     重散列只找空位、完全不做 key 比较（原版 rehash 走的是带比较的拷贝插入）。
//     平凡可析构类型直接跳过析构循环。
//
//  接口与原版兼容：insert / find / contains / erase / operator[] / clear /
//  reserve / rehash / size / bucket_count / load_factor / max_load_factor，
//  另外补充了迭代器、try_emplace、at、count、swap、拷贝/移动语义。
//
//  槽位数恒为 2^n - 1（15 / 31 / 63 …），控制字节数组尾部额外放
//  1 个哨兵字节和 kWidth 个镜像字节，使“跨越末尾的组加载”无需任何分支。
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

// 定义 CLAUDE_HASHTABLE_FORCE_PORTABLE 可强制走 SWAR 分支（便于在 x86 上测试该路径）
#if defined(CLAUDE_HASHTABLE_FORCE_PORTABLE)
#define CLAUDE_HASHTABLE_SSE2 0
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define CLAUDE_HASHTABLE_SSE2 1
#include <emmintrin.h>
#else
#define CLAUDE_HASHTABLE_SSE2 0
#endif

namespace claude {
namespace detail {

// --------------------------------------------------------------------------
// 位操作工具（不强制要求 C++20 的 <bit>）
// --------------------------------------------------------------------------
template <typename T>
inline int count_trailing_zeros(T x);
template <typename T>
inline int count_leading_zeros(T x);

template <>
inline int count_trailing_zeros<uint32_t>(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(x);
#else
    int n = 0;
    while (!(x & 1u)) { x >>= 1; ++n; }
    return n;
#endif
}
template <>
inline int count_trailing_zeros<uint64_t>(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    int n = 0;
    while (!(x & 1ull)) { x >>= 1; ++n; }
    return n;
#endif
}
template <>
inline int count_leading_zeros<uint32_t>(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clz(x);
#else
    int n = 0;
    while (!(x & 0x80000000u)) { x <<= 1; ++n; }
    return n;
#endif
}
template <>
inline int count_leading_zeros<uint64_t>(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clzll(x);
#else
    int n = 0;
    while (!(x & (1ull << 63))) { x <<= 1; ++n; }
    return n;
#endif
}

// --------------------------------------------------------------------------
// 控制字节
//   Empty    0b1000'0000 (0x80)
//   Deleted  0b1111'1110 (0xFE)  —— 墓碑
//   Sentinel 0b1111'1111 (0xFF)  —— 末尾哨兵，仅供迭代器停步
//   Occupied 0b0xxx'xxxx         —— 低 7 位为哈希指纹 H2
// 最高位即可区分“占用”与“非占用”，一条指令完成筛选。
// --------------------------------------------------------------------------
enum : int8_t {
    kCtrlEmpty = static_cast<int8_t>(0x80),
    kCtrlDeleted = static_cast<int8_t>(0xFE),
    kCtrlSentinel = static_cast<int8_t>(0xFF),
};

inline bool ctrl_is_full(int8_t c) { return c >= 0; }

// 位掩码：每个“通道”(lane) 对应组内一个槽位。
// SSE2 版每 lane 占 1 bit（Shift = 0）；SWAR 版每 lane 占 8 bit（Shift = 3）。
template <typename T, int SignificantLanes, int Shift>
class bit_mask {
public:
    explicit bit_mask(T mask) : mask_(mask) {}

    explicit operator bool() const { return mask_ != 0; }
    size_t lowest() const {
        return static_cast<size_t>(count_trailing_zeros<T>(mask_) >> Shift);
    }
    void clear_lowest() { mask_ &= static_cast<T>(mask_ - 1); }
    // 最高被置位的 lane 之上还剩多少个 lane
    size_t leading_zero_lanes() const {
        constexpr int kExtra = static_cast<int>(sizeof(T) * 8) - (SignificantLanes << Shift);
        return static_cast<size_t>(
            count_leading_zeros<T>(static_cast<T>(mask_ << kExtra)) >> Shift);
    }

private:
    T mask_;
};

#if CLAUDE_HASHTABLE_SSE2
// 16 路 SIMD 组
struct group {
    static constexpr size_t kWidth = 16;
    using mask_type = bit_mask<uint32_t, 16, 0>;

    explicit group(const int8_t* pos)
        : ctrl(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pos))) {}

    // 指纹等于 h2 的槽位：一条指令覆盖 16 个槽位
    mask_type match(uint8_t h2) const {
        return mask_type(static_cast<uint32_t>(_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_set1_epi8(static_cast<char>(h2)), ctrl))));
    }
    // 空槽（== 0x80），不含墓碑与哨兵
    mask_type match_empty() const {
        return mask_type(static_cast<uint32_t>(_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_set1_epi8(static_cast<char>(kCtrlEmpty)), ctrl))));
    }
    // 空槽或墓碑（即 ctrl < kSentinel），刻意排除哨兵
    mask_type match_empty_or_deleted() const {
        return mask_type(static_cast<uint32_t>(_mm_movemask_epi8(
            _mm_cmpgt_epi8(_mm_set1_epi8(static_cast<char>(kCtrlSentinel)), ctrl))));
    }

    __m128i ctrl;
};
#else
// 8 路 SWAR 组（纯整数实现，任何平台可用）
struct group {
    static constexpr size_t kWidth = 8;
    using mask_type = bit_mask<uint64_t, 8, 3>;

    explicit group(const int8_t* pos) { std::memcpy(&ctrl, pos, sizeof(ctrl)); }

    mask_type match(uint8_t h2) const {
        constexpr uint64_t kMsbs = 0x8080808080808080ULL;
        constexpr uint64_t kLsbs = 0x0101010101010101ULL;
        const uint64_t x = ctrl ^ (kLsbs * h2);
        return mask_type((x - kLsbs) & ~x & kMsbs);
    }
    mask_type match_empty() const {
        constexpr uint64_t kMsbs = 0x8080808080808080ULL;
        return mask_type((ctrl & (~ctrl << 6)) & kMsbs);
    }
    mask_type match_empty_or_deleted() const {
        constexpr uint64_t kMsbs = 0x8080808080808080ULL;
        return mask_type((ctrl & ~(ctrl << 7)) & kMsbs);
    }

    uint64_t ctrl;
};
#endif

// 一次 64x64→128 位乘法的异或折叠，比三轮 xor-shift-multiply 便宜，
// 足以打散 std::hash<int> 的恒等映射。
inline uint64_t mix_hash(uint64_t x) {
    constexpr uint64_t kMul = 0x9E3779B97F4A7C15ULL;
#if defined(__SIZEOF_INT128__)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
    const unsigned __int128 product = static_cast<unsigned __int128>(x) * kMul;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    return static_cast<uint64_t>(product) ^ static_cast<uint64_t>(product >> 64);
#else
    x ^= x >> 33;
    x *= kMul;
    x ^= x >> 29;
    x *= 0xC2B2AE3D27D4EB4FULL;
    return x ^ (x >> 32);
#endif
}

}  // namespace detail

// =============================================================================
//  claude::hashtable
// =============================================================================
template <typename Key,
          typename T,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class hashtable {
public:
    using key_type = Key;
    using mapped_type = T;
    using value_type = std::pair<Key, T>;
    using size_type = std::size_t;
    using hasher = Hash;
    using key_equal = KeyEqual;

private:
    using slot_type = value_type;
    using group = detail::group;
    static constexpr size_type kWidth = group::kWidth;
    static constexpr size_type kMinCapacity = 15;  // 槽位数 2^4 - 1
    static constexpr size_type npos = static_cast<size_type>(-1);

    static_assert(std::is_default_constructible_v<Hash>, "Hash 必须可默认构造");
    static_assert(std::is_default_constructible_v<KeyEqual>, "KeyEqual 必须可默认构造");

    // capacity_ = 槽位数 = 2^n - 1，同时直接当掩码用（hash & capacity_ ∈ [0, capacity_]）。
    // ctrl_ 长度 = capacity_ + 1 + kWidth：
    //   [0, capacity_)              真实控制字节
    //   [capacity_]                 哨兵
    //   [capacity_+1, +kWidth]      开头 kWidth 个字节的镜像，用于无分支回绕
    int8_t* ctrl_;
    slot_type* slots_;
    size_type size_;
    size_type capacity_;
    size_type growth_left_;  // 还能往“空槽”里插入多少个元素
    hasher hash_;
    key_equal eq_;

    // capacity_ == 0 时 ctrl_ 指向这里：首字节是哨兵（begin() == end()），
    // 其余为空槽（查找立即失败）。插入因 growth_left_ == 0 必先扩容，
    // 所以永远不会写到这块共享数据。
    static int8_t* empty_group() {
        alignas(16) static const int8_t kEmptyGroup[32] = {
            detail::kCtrlSentinel, detail::kCtrlEmpty, detail::kCtrlEmpty, detail::kCtrlEmpty,
            detail::kCtrlEmpty,    detail::kCtrlEmpty, detail::kCtrlEmpty, detail::kCtrlEmpty,
            detail::kCtrlEmpty,    detail::kCtrlEmpty, detail::kCtrlEmpty, detail::kCtrlEmpty,
            detail::kCtrlEmpty,    detail::kCtrlEmpty, detail::kCtrlEmpty, detail::kCtrlEmpty,
            detail::kCtrlEmpty,    detail::kCtrlEmpty, detail::kCtrlEmpty, detail::kCtrlEmpty,
            detail::kCtrlEmpty,    detail::kCtrlEmpty, detail::kCtrlEmpty, detail::kCtrlEmpty,
            detail::kCtrlEmpty,    detail::kCtrlEmpty, detail::kCtrlEmpty, detail::kCtrlEmpty,
            detail::kCtrlEmpty,    detail::kCtrlEmpty, detail::kCtrlEmpty, detail::kCtrlEmpty};
        return const_cast<int8_t*>(kEmptyGroup);
    }

    // 返回 >= n 的最小 2^k - 1（且不小于 kMinCapacity）
    static size_type normalize_capacity(size_type n) {
        if (n <= kMinCapacity) return kMinCapacity;
        --n;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        if constexpr (sizeof(size_type) >= 8) n |= n >> 32;
        return n;
    }

    // 负载因子 7/8
    static size_type capacity_to_growth(size_type cap) { return cap - cap / 8; }
    // 容纳 n 个元素所需的最小容量
    static size_type growth_to_capacity(size_type n) {
        return normalize_capacity(n + (n + 6) / 7);
    }

    static size_type h1(uint64_t hash) { return static_cast<size_type>(hash >> 7); }
    static uint8_t h2(uint64_t hash) { return static_cast<uint8_t>(hash & 0x7F); }

    uint64_t hash_key(const key_type& key) const {
        return detail::mix_hash(static_cast<uint64_t>(hash_(key)));
    }

    // 写控制字节并同步尾部镜像；当 i 不需要镜像时该表达式恰好退化为写 i 本身，
    // 因此完全无分支。
    void set_ctrl(size_type i, int8_t c) {
        ctrl_[i] = c;
        ctrl_[((i - kWidth) & capacity_) + 1 + ((kWidth - 1) & capacity_)] = c;
    }

public:
    // ------------------------------------------------------------- 迭代器
    class iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = hashtable::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        iterator() : ctrl_(nullptr), slot_(nullptr) {}
        reference operator*() const { return *slot_; }
        pointer operator->() const { return slot_; }
        iterator& operator++() {
            ++ctrl_;
            ++slot_;
            skip_empty();
            return *this;
        }
        iterator operator++(int) { iterator tmp = *this; ++*this; return tmp; }
        friend bool operator==(const iterator& a, const iterator& b) { return a.ctrl_ == b.ctrl_; }
        friend bool operator!=(const iterator& a, const iterator& b) { return a.ctrl_ != b.ctrl_; }

    private:
        friend class hashtable;
        iterator(int8_t* c, slot_type* s) : ctrl_(c), slot_(s) {}
        void skip_empty() {
            while (*ctrl_ != detail::kCtrlSentinel && !detail::ctrl_is_full(*ctrl_)) {
                ++ctrl_;
                ++slot_;
            }
        }
        int8_t* ctrl_;
        slot_type* slot_;
    };

    class const_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const hashtable::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = const hashtable::value_type*;
        using reference = const hashtable::value_type&;

        const_iterator() = default;
        const_iterator(iterator it) : it_(it) {}
        reference operator*() const { return *it_; }
        pointer operator->() const { return it_.operator->(); }
        const_iterator& operator++() { ++it_; return *this; }
        const_iterator operator++(int) { const_iterator tmp = *this; ++*this; return tmp; }
        friend bool operator==(const const_iterator& a, const const_iterator& b) {
            return a.it_ == b.it_;
        }
        friend bool operator!=(const const_iterator& a, const const_iterator& b) {
            return a.it_ != b.it_;
        }

    private:
        iterator it_;
    };

    // ---------------------------------------------------------- 构造/析构
    hashtable()
        : ctrl_(empty_group()),
          slots_(nullptr),
          size_(0),
          capacity_(0),
          growth_left_(0),
          hash_(),
          eq_() {}

    explicit hashtable(size_type expected_size) : hashtable() { reserve(expected_size); }

    hashtable(std::initializer_list<value_type> init) : hashtable() {
        reserve(init.size());
        for (const value_type& kv : init) insert(kv.first, kv.second);
    }

    hashtable(const hashtable& other) : hashtable() {
        reserve(other.size_);
        for (const value_type& kv : other) insert(kv.first, kv.second);
    }

    hashtable(hashtable&& other) noexcept
        : ctrl_(other.ctrl_),
          slots_(other.slots_),
          size_(other.size_),
          capacity_(other.capacity_),
          growth_left_(other.growth_left_),
          hash_(std::move(other.hash_)),
          eq_(std::move(other.eq_)) {
        other.reset_to_empty();
    }

    hashtable& operator=(const hashtable& other) {
        if (this != &other) {
            hashtable tmp(other);
            swap(tmp);
        }
        return *this;
    }

    hashtable& operator=(hashtable&& other) noexcept {
        if (this != &other) {
            destroy_all();
            ctrl_ = other.ctrl_;
            slots_ = other.slots_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            growth_left_ = other.growth_left_;
            hash_ = std::move(other.hash_);
            eq_ = std::move(other.eq_);
            other.reset_to_empty();
        }
        return *this;
    }

    ~hashtable() { destroy_all(); }

    void swap(hashtable& other) noexcept {
        std::swap(ctrl_, other.ctrl_);
        std::swap(slots_, other.slots_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        std::swap(growth_left_, other.growth_left_);
        std::swap(hash_, other.hash_);
        std::swap(eq_, other.eq_);
    }

    // ---------------------------------------------------------------- 迭代
    iterator begin() {
        iterator it(ctrl_, slots_);
        it.skip_empty();
        return it;
    }
    iterator end() { return iterator(ctrl_ + capacity_, slots_ + capacity_); }
    const_iterator begin() const { return const_cast<hashtable*>(this)->begin(); }
    const_iterator end() const { return const_cast<hashtable*>(this)->end(); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    // ---------------------------------------------------------------- 查询
    mapped_type* find(const key_type& key) {
        const size_type idx = find_index(key);
        return idx == npos ? nullptr : &slots_[idx].second;
    }
    const mapped_type* find(const key_type& key) const {
        const size_type idx = find_index(key);
        return idx == npos ? nullptr : &slots_[idx].second;
    }
    bool contains(const key_type& key) const { return find_index(key) != npos; }
    size_type count(const key_type& key) const { return contains(key) ? 1 : 0; }

    mapped_type& at(const key_type& key) {
        mapped_type* p = find(key);
        if (!p) throw std::out_of_range("claude::hashtable::at: key 不存在");
        return *p;
    }
    const mapped_type& at(const key_type& key) const {
        const mapped_type* p = find(key);
        if (!p) throw std::out_of_range("claude::hashtable::at: key 不存在");
        return *p;
    }

    // ---------------------------------------------------------------- 修改
    // 返回值语义与原版一致：true = 新插入；false = key 已存在（覆盖 value）
    bool insert(const key_type& key, const mapped_type& value) {
        const std::pair<size_type, bool> res = find_or_prepare_insert(key);
        if (res.second) {
            construct_slot(res.first, key, value);
            return true;
        }
        slots_[res.first].second = value;
        return false;
    }

    bool insert(key_type&& key, mapped_type&& value) {
        const std::pair<size_type, bool> res = find_or_prepare_insert(key);
        if (res.second) {
            construct_slot(res.first, std::move(key), std::move(value));
            return true;
        }
        slots_[res.first].second = std::move(value);
        return false;
    }

    bool insert(const value_type& kv) { return insert(kv.first, kv.second); }

    // 仅当 key 不存在时才构造 value；返回 {value 指针, 是否新插入}
    template <typename... Args>
    std::pair<mapped_type*, bool> try_emplace(const key_type& key, Args&&... args) {
        const std::pair<size_type, bool> res = find_or_prepare_insert(key);
        if (res.second) {
            construct_slot_piecewise(res.first, key, std::forward<Args>(args)...);
        }
        return {&slots_[res.first].second, res.second};
    }

    mapped_type& operator[](const key_type& key) {
        const std::pair<size_type, bool> res = find_or_prepare_insert(key);
        if (res.second) construct_slot(res.first, key, mapped_type());
        return slots_[res.first].second;
    }

    bool erase(const key_type& key) {
        const size_type idx = find_index(key);
        if (idx == npos) return false;
        erase_at(idx);
        return true;
    }

    void clear() {
        if (capacity_ == 0) return;
        destroy_slots_only();
        std::memset(ctrl_, detail::kCtrlEmpty, capacity_ + 1 + kWidth);
        ctrl_[capacity_] = detail::kCtrlSentinel;
        size_ = 0;
        growth_left_ = capacity_to_growth(capacity_);
    }

    void reserve(size_type expected_size) {
        if (expected_size == 0) return;
        const size_type need = growth_to_capacity(expected_size);
        if (need > capacity_) resize(need);
    }

    // 兼容原版：参数是“期望桶数”
    void rehash(size_type requested_bucket_count) {
        const size_type want =
            normalize_capacity(requested_bucket_count == 0 ? kMinCapacity : requested_bucket_count);
        const size_type floor_cap = growth_to_capacity(size_);
        const size_type target = want > floor_cap ? want : floor_cap;
        if (target != capacity_) resize(target);
    }

    // ---------------------------------------------------------------- 统计
    size_type size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_type bucket_count() const { return capacity_; }
    float max_load_factor() const { return 0.875f; }
    double load_factor() const {
        return capacity_ == 0 ? 0.0
                              : static_cast<double>(size_) / static_cast<double>(capacity_);
    }

private:
    // ----------------------------------------------------------- 内部实现
    void reset_to_empty() {
        ctrl_ = empty_group();
        slots_ = nullptr;
        size_ = 0;
        capacity_ = 0;
        growth_left_ = 0;
    }

    void destroy_slots_only() {
        if constexpr (!std::is_trivially_destructible_v<slot_type>) {
            if (slots_ != nullptr) {
                for (size_type i = 0; i < capacity_; ++i) {
                    if (detail::ctrl_is_full(ctrl_[i])) slots_[i].~slot_type();
                }
            }
        }
    }

    void destroy_all() {
        if (capacity_ == 0) return;
        destroy_slots_only();
        ::operator delete(static_cast<void*>(ctrl_));
        ::operator delete(static_cast<void*>(slots_), std::align_val_t(alignof(slot_type)));
        reset_to_empty();
    }

    template <typename K, typename V>
    void construct_slot(size_type idx, K&& key, V&& value) {
        ::new (static_cast<void*>(slots_ + idx))
            slot_type(std::forward<K>(key), std::forward<V>(value));
    }

    template <typename... Args>
    void construct_slot_piecewise(size_type idx, const key_type& key, Args&&... args) {
        ::new (static_cast<void*>(slots_ + idx))
            slot_type(std::piecewise_construct, std::forward_as_tuple(key),
                      std::forward_as_tuple(std::forward<Args>(args)...));
    }

    // 查找 key 所在槽位；找不到返回 npos
    size_type find_index(const key_type& key) const {
        const uint64_t hash = hash_key(key);
        const uint8_t fp = h2(hash);
        const size_type mask = capacity_;
        size_type pos = h1(hash) & mask;
        size_type stride = 0;

        while (true) {
            group g(ctrl_ + pos);
            for (auto m = g.match(fp); m; m.clear_lowest()) {
                const size_type idx = (pos + m.lowest()) & mask;
                if (eq_(slots_[idx].first, key)) return idx;
            }
            if (g.match_empty()) return npos;  // 探测链到此结束
            stride += kWidth;                  // 按组二次探测：+W, +3W, +6W, ...
            pos = (pos + stride) & mask;
        }
    }

    // 找探测链上第一个可写槽位（空槽或墓碑）；重散列时用，无需比较 key
    size_type find_first_non_full(uint64_t hash) const {
        const size_type mask = capacity_;
        size_type pos = h1(hash) & mask;
        size_type stride = 0;
        while (true) {
            group g(ctrl_ + pos);
            if (auto m = g.match_empty_or_deleted()) return (pos + m.lowest()) & mask;
            stride += kWidth;
            pos = (pos + stride) & mask;
        }
    }

    // 返回 {槽位下标, 是否为新 key（需要在该槽位构造）}
    std::pair<size_type, bool> find_or_prepare_insert(const key_type& key) {
        const uint64_t hash = hash_key(key);
        const uint8_t fp = h2(hash);

        while (true) {
            const size_type mask = capacity_;
            size_type pos = h1(hash) & mask;
            size_type stride = 0;
            size_type insert_pos = npos;

            while (true) {
                group g(ctrl_ + pos);
                for (auto m = g.match(fp); m; m.clear_lowest()) {
                    const size_type idx = (pos + m.lowest()) & mask;
                    if (eq_(slots_[idx].first, key)) return {idx, false};
                }
                if (insert_pos == npos) {
                    // 复用探测链上最早出现的墓碑/空槽：之后查找同一 key
                    // 会沿同样顺序先到达这里，不会漏判。
                    if (auto med = g.match_empty_or_deleted()) {
                        insert_pos = (pos + med.lowest()) & mask;
                    }
                }
                if (g.match_empty()) break;
                stride += kWidth;
                pos = (pos + stride) & mask;
            }

            // 复用墓碑不消耗增长配额；写入空槽才需要配额，配额用尽则先扩容。
            // （capacity_ == 0 时 ctrl_[0] 是共享只读哨兵，这里同样会走扩容分支。）
            const bool reuse_tombstone =
                insert_pos != npos && ctrl_[insert_pos] == detail::kCtrlDeleted;
            if (!reuse_tombstone && growth_left_ == 0) {
                grow();
                continue;  // 扩容后重新探测
            }

            if (!reuse_tombstone) --growth_left_;
            set_ctrl(insert_pos, static_cast<int8_t>(fp));
            ++size_;
            return {insert_pos, true};
        }
    }

    void erase_at(size_type idx) {
        if constexpr (!std::is_trivially_destructible_v<slot_type>) {
            slots_[idx].~slot_type();
        }
        --size_;

        // 能否直接写回 Empty 而不留墓碑？
        // 若 idx 左右两组里都存在空槽，且“右侧连续非空长度 + 左侧连续非空长度 < 组宽”，
        // 则不存在任何探测窗口曾把该槽位当成满组跳过，墓碑可省。
        // idx 所在组的控制字节刚刚才访问过（必然在 cache 里）；
        // 只有它存在空槽时才值得再去读“前一组”，否则直接留墓碑。
        bool never_full = false;
        auto empty_after = group(ctrl_ + idx).match_empty();
        if (empty_after) {
            const size_type before = (idx - kWidth) & capacity_;
            auto empty_before = group(ctrl_ + before).match_empty();
            never_full = static_cast<bool>(empty_before) &&
                         (empty_after.lowest() + empty_before.leading_zero_lanes()) < kWidth;
        }

        if (never_full) {
            set_ctrl(idx, detail::kCtrlEmpty);
            ++growth_left_;
        } else {
            set_ctrl(idx, detail::kCtrlDeleted);
        }
    }

    void grow() {
        if (capacity_ == 0) {
            resize(kMinCapacity);
        } else if (size_ * 2 <= capacity_to_growth(capacity_)) {
            // 元素少但墓碑多：按原容量重建即可清掉墓碑，不必翻倍
            resize(capacity_);
        } else {
            resize(capacity_ * 2 + 1);
        }
    }

    void resize(size_type new_capacity) {
        int8_t* old_ctrl = ctrl_;
        slot_type* old_slots = slots_;
        const size_type old_capacity = capacity_;

        int8_t* new_ctrl = static_cast<int8_t*>(::operator new(new_capacity + 1 + kWidth));
        slot_type* new_slots = static_cast<slot_type*>(::operator new(
            sizeof(slot_type) * new_capacity, std::align_val_t(alignof(slot_type))));
        std::memset(new_ctrl, detail::kCtrlEmpty, new_capacity + 1 + kWidth);
        new_ctrl[new_capacity] = detail::kCtrlSentinel;

        ctrl_ = new_ctrl;
        slots_ = new_slots;
        capacity_ = new_capacity;
        growth_left_ = capacity_to_growth(new_capacity) - size_;

        if (old_capacity != 0) {
            for (size_type i = 0; i < old_capacity; ++i) {
                if (!detail::ctrl_is_full(old_ctrl[i])) continue;
                const uint64_t hash = hash_key(old_slots[i].first);
                const size_type target = find_first_non_full(hash);
                set_ctrl(target, static_cast<int8_t>(h2(hash)));
                ::new (static_cast<void*>(slots_ + target)) slot_type(std::move(old_slots[i]));
                if constexpr (!std::is_trivially_destructible_v<slot_type>) {
                    old_slots[i].~slot_type();
                }
            }
            ::operator delete(static_cast<void*>(old_ctrl));
            ::operator delete(static_cast<void*>(old_slots),
                              std::align_val_t(alignof(slot_type)));
        }
    }
};

template <typename Key, typename T, typename Hash, typename KeyEqual>
inline void swap(hashtable<Key, T, Hash, KeyEqual>& a,
                 hashtable<Key, T, Hash, KeyEqual>& b) noexcept {
    a.swap(b);
}

}  // namespace claude
