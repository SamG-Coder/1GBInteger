#include "temporal_kernel.h"
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    // Small complete rings exercise repeated wrap, not just the first limb.
    for (uint64_t limbs : {1, 2, 3, 7, 17})
    for (uint64_t seed : {uint64_t(0), uint64_t(12345), UINT64_MAX}) {
        constexpr uint64_t steps = 2177;
        std::vector<std::vector<uint64_t>> history(steps + 1, std::vector<uint64_t>(limbs));
        for (uint64_t i = 0; i < limbs; ++i)
            history[0][i] = temporal::mix(seed + (i + 1) * temporal::C);
        for (uint64_t t = 1; t <= steps; ++t)
            for (uint64_t i = 0; i < limbs; ++i)
                history[t][i] = ((history[t-1][i] << 1) |
                    (history[t-1][i ? i-1 : limbs-1] >> 63)) ^ temporal::mix(seed ^ (i * temporal::C));
        for (uint64_t interval : {uint64_t(1), uint64_t(7), uint64_t(10), uint64_t(63), uint64_t(64),
                                  uint64_t(65), uint64_t(127), uint64_t(1024), UINT64_MAX})
        for (uint64_t end : {uint64_t(0), uint64_t(1), uint64_t(63), uint64_t(64), uint64_t(65), steps})
        for (uint64_t i = 0; i < limbs; ++i) {
            uint64_t count = 0;
            temporal::sample(seed, i, limbs, end, interval, [&](uint64_t t, uint64_t value) {
                if (t != count * interval || value != history[t][i])
                    throw std::runtime_error("Temporal/full ring mismatch");
                ++count;
            });
            if (count != end / interval + 1) throw std::runtime_error("Missing observation");
        }
    }
    std::cout << "PASS: exact full-ring evolution, boundaries, long intervals, repeated wrap\n";
}
