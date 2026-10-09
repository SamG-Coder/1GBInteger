// Native inference from exported learned coefficients; no task-label oracle.
#include "xor_inference.h"
#include <fstream>
#include <iostream>
int main(int argc,char** argv){try{
    if(argc<3)throw std::runtime_error("Usage: xor_predict MODEL_FILE INPUT [INPUT ...], inputs 0..65535 (decimal or 0x hex)");
    std::ifstream in(argv[1]);std::string magic,name;size_t count=0;
    if(!(in>>magic>>name>>count)||magic!="XOR_MODEL_V1")throw std::runtime_error("Invalid model header");
    inference::Features features(name);if(count!=features.size())throw std::runtime_error("Feature count mismatch");
    auto weights=inference::zeros(count);for(auto& w:weights)if(!(in>>std::hex>>w))throw std::runtime_error("Invalid weights");
    if(count%64&&weights.back()>>(count%64))throw std::runtime_error("Weights outside feature bank");
    for(int i=2;i<argc;++i){std::string value=argv[i];size_t used=0;
        if(value.empty()||value[0]=='-')throw std::runtime_error("Input must be unsigned");
        const int base=value.size()>2&&value[0]=='0'&&(value[1]=='x'||value[1]=='X')?16:10;
        auto x=std::stoul(value,&used,base);if(used!=value.size()||x>65535)throw std::runtime_error("Input out of range");
        std::cout<<x<<','<<inference::parity(features.encode(uint16_t(x)),weights)<<'\n';
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
