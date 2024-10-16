#include "../header_files/NN_Mod.h"

Net::Net(int64_t N, int64_t M):linear(register_module("linear", torch::nn::Linear(N, M))){
    bias = register_parameter("b", torch::randn(M));
}

torch::Tensor Net::forward(torch::Tensor input){
    return linear(input) + bias;
}
