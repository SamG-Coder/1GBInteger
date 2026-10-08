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

// S(t)[index] = ROTL(S(0),t)[index] XOR sum_{k<t} ROTL(M,k)[index].
// The two source limbs only change at multiples of 64, including ring wrap.
// Each callback receives one exact observed limb, in increasing time order.
template<class Observe>
void sample(uint64_t seed, uint64_t index, uint64_t limbs, uint64_t steps,
            uint64_t interval, Observe observe) {
    uint64_t acc = 0, next = 0;
    for (uint64_t base = 0; base <= steps; base += 64) {
        const uint64_t previous = index ? index - 1 : limbs - 1;
        const uint64_t x = mix(seed ^ (index * C));
        const uint64_t y = mix(seed ^ (previous * C));
        const uint64_t sx = mix(seed + (index + 1) * C);
        const uint64_t sy = mix(seed + (previous + 1) * C);
        const unsigned end = unsigned(std::min<uint64_t>(64, steps - base + 1));
        for (unsigned bit = 0; bit < end; ++bit) {
            const uint64_t t = base + bit;
            if (t == next) {
                observe(t, rotated(sx, sy, bit) ^ acc);
                // Saturating scheduling also handles UINT64_MAX intervals.
                next = interval > steps - t ? UINT64_MAX : t + interval;
            }
            acc ^= rotated(x, y, bit);
        }
        index = previous;
    }
}
} // namespace temporal
