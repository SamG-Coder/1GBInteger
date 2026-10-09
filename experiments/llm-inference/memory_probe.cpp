// SPDX-License-Identifier: MIT
// Read-only XOR reduction: cache/DRAM working-set probe, not a model benchmark.
#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
#include <immintrin.h>
int main(){
    std::cout<<"mib,threads,repeat,passes,milliseconds,read_gb_s,checksum\n";
    for(size_t mib:{32,96,384})for(int nt:{1,4,8,16}){
        std::vector<uint64_t> data(mib*1024*1024/8);
        for(size_t i=0;i<data.size();++i)data[i]=uint64_t(i)*0x9e3779b97f4a7c15ULL;
        std::vector<uint64_t> sums(nt);int passes=int(2048/mib)+1;
        for(int rep=0;rep<3;++rep){
            auto start=std::chrono::steady_clock::now();std::vector<std::thread> workers;
            for(int th=0;th<nt;++th)workers.emplace_back([&,th]{
                const size_t begin=data.size()*th/nt,end=data.size()*(th+1)/nt;
                uint64_t total=0;
                for(int p=0;p<passes;++p){
                    asm volatile("" : : "r"(data.data()) : "memory");
                    __m256i sum=_mm256_setzero_si256();
                    for(size_t i=begin;i<end;i+=4)sum=_mm256_xor_si256(sum,_mm256_loadu_si256((const __m256i*)&data[i]));
                    uint64_t lane[4];_mm256_storeu_si256((__m256i*)lane,sum);total+=lane[0]^lane[1]^lane[2]^lane[3];
                }sums[th]=total;
            });
            for(auto & w:workers)w.join();
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();uint64_t checksum=0;for(auto s:sums)checksum^=s;
            std::cout<<mib<<','<<nt<<','<<rep<<','<<passes<<','<<ms<<','<<(mib*1024.*1024.*passes/ms/1e6)<<','<<checksum<<'\n';
        }
    }
}
