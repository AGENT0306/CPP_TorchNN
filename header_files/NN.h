#ifndef NN_H
#define NN_H

#include <torch/torch.h>

struct NNet : public torch::nn::Module{
    public:
        NNet(int inputs, int outputs);
        torch::Tensor forward(torch::Tensor data);
        torch::nn::Linear L1, L2, L3, L4, L5, L6;
        int in_features;
        int out_features;
};

#endif