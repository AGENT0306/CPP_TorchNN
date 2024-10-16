#ifndef NN_MOD_H
#define NN_MOD_H

#include <torch/torch.h>

struct Net : torch::nn::Module{
    Net(int64_t N, int64_t M);
    torch::Tensor forward(torch::Tensor input);
    torch::nn::Linear linear;
    torch::Tensor bias;
};

#endif