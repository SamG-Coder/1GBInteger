#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <string>
#include <algorithm>
// Lazy closed-form evaluation of the EXISTING integer-universe evolution:
// S(t+1) = ROTL(S(t), 1) XOR M, where M is a fixed per-limb mask.
// S(t) = ROTL(S(0),t) XOR XOR_{k=0}^{t-1} ROTL(M,k).
// This does not allocate or scan the logical 1 GB integer.
// Complexity: O(sampled_limbs * steps); full-state materialization remains O(N).
constexpr uint64_t C=0x9e3779b97f4a7c15ULL;
static uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
static uint64_t initial(uint64_t seed,uint64_t i){return mix(seed+(i+1)*C);}
static uint64_t mask(uint64_t seed,uint64_t i){return mix(seed^(i*C));}
static uint64_t rotated(uint64_t seed,uint64_t idx,uint64_t shifts,uint64_t limbs,bool useInitial){
 uint64_t words=(shifts/64)%limbs;unsigned bits=shifts%64;
 uint64_t a=(idx+limbs-words)%limbs,prev=(a+limbs-1)%limbs;
 auto val=[&](uint64_t i){return useInitial?initial(seed,i):mask(seed,i);};
 return bits?((val(a)<<bits)|(val(prev)>>(64-bits))):val(a);
}
static uint64_t lazy(uint64_t seed,uint64_t idx,uint64_t steps,uint64_t limbs){
 uint64_t result=rotated(seed,idx,steps,limbs,true);
 for(uint64_t k=0;k<steps;k++)result^=rotated(seed,idx,k,limbs,false);
 return result;
}
int main(int argc,char**argv){try{
 uint64_t seed=12345,steps=100,gb=1,mb=0,sampleBytes=1048576;
 std::string path="sample.bin";
 for(int i=1;i<argc;i++){
  std::string a=argv[i];if(i+1>=argc)throw std::runtime_error("Missing value for "+a);
  std::string v=argv[++i];
  if(a=="--out")path=v;else if(a=="--seed")seed=std::stoull(v);else if(a=="--steps")steps=std::stoull(v);else if(a=="--gb")gb=std::stoull(v);else if(a=="--mb")mb=std::stoull(v);else if(a=="--sample-bytes")sampleBytes=std::stoull(v);else throw std::runtime_error("Unknown option "+a);
 }
 uint64_t bytes=mb?mb*1000000ULL:gb*1000000000ULL;
 if(!bytes||bytes%8||bytes>100000000000ULL||steps>100000||sampleBytes>16000000)throw std::runtime_error("Invalid size, step count or sample size");
 uint64_t limbs=bytes/8,n=std::min(sampleBytes,bytes)/8;
 if(!n)throw std::runtime_error("Sample must contain at least one limb");
 std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("Cannot create output");
 auto t=std::chrono::steady_clock::now();
 std::vector<char> data(size_t(n)*8);
 for(uint64_t j=0;j<n;j++){
  uint64_t idx=(j*limbs)/n,v=lazy(seed,idx,steps,limbs);
  for(int k=0;k<8;k++)data[size_t(j)*8+k]=char(v>>(k*8));
 }
 out.write(data.data(),data.size());if(!out)throw std::runtime_error("Write failed");
 double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
 std::cout<<"logical_bytes="<<bytes<<" sampled_bytes="<<data.size()<<" steps="<<steps<<" elapsed_seconds="<<sec<<" samples_per_sec="<<(n/sec)<<" output="<<path<<"\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
