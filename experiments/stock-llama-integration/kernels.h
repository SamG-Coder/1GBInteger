// SPDX-License-Identifier: MIT
#pragma once
#include "llama.h"
#include <cstdint>
#include <string>
namespace experiment {
void prepare(llama_model *,const std::string &);
void set_decode_phase(bool);
uint64_t calls();
uint64_t storage_bytes();
void reset_calls();
bool supported();
// Native Q8_0 blocks, four consecutive rows. Public for independent tests.
void dot4(const void * weights,const void * activation,int blocks,float * out);
}
