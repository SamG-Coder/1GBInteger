#pragma once
#include <algorithm>
#include <cstdint>

namespace temporal {
constexpr uint64_t C = 0x9e3779b97f4a7c15ULL;
inline uint64_t mix(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
inline uint64_t rotated(uint64_t x, uint64_t previous, unsigned bit) {
    return bit ? (x << bit) | (previous >> (64 - bit)) : x;
}
inline uint64_t prefix_left(uint64_t x) {
    x ^= x << 1; x ^= x << 2; x ^= x << 4;
    x ^= x << 8; x ^= x << 16; x ^= x << 32;
    return x;
}
inline uint64_t prefix_right(uint64_t x) {
    x ^= x >> 1; x ^= x >> 2; x ^= x >> 4;
    x ^= x >> 8; x ^= x >> 16; x ^= x >> 32;
    return x;
}

// S(t)[index] = ROTL(S(0),t)[index] XOR sum_{k<t} ROTL(M,k)[index].
// The two source limbs only change at multiples of 64, including ring wrap.
// Each callback receives one exact observed limb, in increasing time order.
template<class Observe>
void sample(uint64_t seed, uint64_t index, uint64_t limbs, uint64_t steps,
            uint64_t interval, Observe observe) {
    uint64_t acc = 0, next = 0;
    const uint64_t last = steps - steps % interval;
    for (uint64_t base = 0; base <= last; base += 64) {
        const uint64_t previous = index ? index - 1 : limbs - 1;
        const uint64_t x = mix(seed ^ (index * C));
        const uint64_t y = mix(seed ^ (previous * C));
        // L = XOR_{k=0..63}(x << k), R = XOR_{k=1..63}(y >> k).
        // For 0 < b < 64, XOR_{k<b} rotated(x,y,k)
        // is L XOR (L << b) XOR (R >> (64-b)). No approximation.
        const uint64_t left = prefix_left(x), right = prefix_right(y >> 1);
        if (next < base + 64 && next <= last) {
            const uint64_t sx = mix(seed + (index + 1) * C);
            const uint64_t sy = mix(seed + (previous + 1) * C);
            do {
                const unsigned bit = unsigned(next - base);
                const uint64_t partial = bit ? left ^ (left << bit) ^ (right >> (64-bit)) : 0;
                observe(next, rotated(sx, sy, bit) ^ acc ^ partial);
                if (interval > last - next) return;
                next += interval;
            } while (next < base + 64);
        }
        acc ^= left ^ right;
        index = previous;
    }
}
} // namespace temporal
