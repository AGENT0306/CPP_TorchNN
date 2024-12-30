#include "../header_files/NN.h"

NNet::NNet(const int inputs, const int outputs):L1(torch::nn::LinearOptions(inputs,124).bias(true)),
                                    L2(torch::nn::LinearOptions(124,256).bias(true)),
                                    L3(torch::nn::LinearOptions(256,1).bias(true)),
                                    L4(torch::nn::LinearOptions(28, 124).bias(true)),
                                    L5(torch::nn::LinearOptions(124,256).bias(true)),
                                    L6(torch::nn::LinearOptions(256,outputs).bias(true)){
    in_features = inputs;
    out_features = outputs;
    register_module("L1", L1);
    register_module("L2", L2);
    register_module("L3", L3);
    register_module("L4", L4);
    register_module("L5", L5);
    register_module("L6", L6);
}

torch::Tensor NNet::forward(torch::Tensor data) {
    data = L1(data);
    data = torch::relu(L2(data));
    data = torch::relu(L3(data));
    data = data.transpose(2,3);
    data = torch::relu(L4(data));
    data = torch::relu(L5(data));
    data = L6(data).softmax(3);
    return data;
}