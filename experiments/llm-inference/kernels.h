// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <immintrin.h>
#include "llama.h"
namespace experiment {
struct Block {
    float scale, binary_scale, ternary_scale;
    uint32_t planes[4], positive, negative;
    uint8_t u[32];
};
struct Activation {
    float scale;
    int8_t q[32];
    uint32_t planes[8];
    int32_t correction[8];
};
Block pack(const int8_t * q, float scale);
Activation pack_activation(const int8_t * q, float scale);
int bit_dot(const Block &, const Activation &);
int binary_dot(const Block &, const Activation &);
int ternary_dot(const Block &, const Activation &);
float simd_dot(const Block *, const Activation *, int blocks);
void prepare(llama_model *, const std::string & mode);
uint64_t storage_bytes();
uint64_t calls();
void reset_calls();
}
