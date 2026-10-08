#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Direct temporal observation of one fixed limb of the logical integer.
// S(t) = ROTL(S(0),t) XOR XOR_{k=0}^{t-1} ROTL(M,k).
// Each successive observation takes O(1) arithmetic per atom (rather than
// rescanning the entire 1 GB state). Does not allocate the 1 GB integer.
static constexpr uint64_t C=0x9e3779b97f4a7c15ULL;
static uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
static uint64_t initial(uint64_t seed,uint64_t i){return mix(seed+(i+1)*C);}
static uint64_t mask(uint64_t seed,uint64_t i){return mix(seed^(i*C));}
static uint64_t rotate_at(uint64_t seed,uint64_t idx,uint64_t shift,uint64_t limbs,bool init){
 uint64_t words=(shift/64)%limbs;
 unsigned bits=unsigned(shift%64);
 uint64_t a=(idx+limbs-words)%limbs;
 uint64_t prev=(a+limbs-1)%limbs;
 uint64_t x=init?initial(seed,a):mask(seed,a);
 if(!bits)return x;
 uint64_t y=init?initial(seed,prev):mask(seed,prev);
 return (x<<bits)|(y>>(64-bits));
}
struct Atom {uint64_t index,maskAccum=0;unsigned bit;};
int main(int argc,char**argv){try{
 uint64_t gb=1,mb=0,seed=12345,steps=10000,atoms=64,interval=1;
 std::string output="temporal.csv";
 for(int i=1;i<argc;i++){
  std::string k=argv[i];if(i+1>=argc)throw std::runtime_error("Missing argument: "+k);
  std::string v=argv[++i];uint64_t n=0;
  if(k!="--out")n=std::stoull(v);
  if(k=="--gb")gb=n;else if(k=="--mb")mb=n;else if(k=="--seed")seed=n;
  else if(k=="--steps")steps=n;else if(k=="--atoms")atoms=n;
  else if(k=="--interval")interval=n;else if(k=="--out")output=v;
  else throw std::runtime_error("Unknown argument: "+k);
 }
 uint64_t bytes=mb?mb*1000000ULL:gb*1000000000ULL;
 if(!bytes||bytes%8||bytes>100000000000ULL||atoms<1||atoms>100000||steps>10000000||!interval)
  throw std::runtime_error("Invalid parameter");
 uint64_t limbs=bytes/8;
 if(atoms>limbs)throw std::runtime_error("Atom count exceeds limb count");
 std::vector<Atom> selected;selected.reserve(size_t(atoms));
 for(uint64_t j=0;j<atoms;j++)selected.push_back({(j*limbs)/atoms,0,unsigned(j%64)});
 std::ofstream out(output);if(!out)throw std::runtime_error("Cannot open output");
 out<<"step,atom,limb_index,bit_index,measurement,limb_hex\n";
 auto start=std::chrono::steady_clock::now();
 uint64_t measurements=0;
 for(uint64_t t=0;t<=steps;t++){
  if(t%interval==0){
   for(size_t j=0;j<selected.size();j++){
    auto &a=selected[j];
    uint64_t value=rotate_at(seed,a.index,t,limbs,true)^a.maskAccum;
    out<<t<<','<<j<<','<<a.index<<','<<a.bit<<','<<((value>>a.bit)&1ULL)<<','
       <<std::hex<<std::setw(16)<<std::setfill('0')<<value<<std::dec<<'\n';
    ++measurements;
   }
  }
  if(t<steps)for(auto &a:selected)a.maskAccum^=rotate_at(seed,a.index,t,limbs,false);
 }
 out.flush();if(!out)throw std::runtime_error("Write failed");
 double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::cout<<"logical_bytes="<<bytes<<" steps="<<steps<<" atoms="<<atoms
          <<" observations="<<measurements<<" elapsed_seconds="<<elapsed
          <<" observations_per_second="<<(measurements/elapsed)<<" output="<<output<<"\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
