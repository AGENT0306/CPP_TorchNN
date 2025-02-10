//
// Created by reitr on 2/9/2025.
//

#include <torch/torch.h>

#ifndef CUSTOMDATASET_H
#define CUSTOMDATASET_H

struct CustomDataset : torch::data::Dataset<CustomDataset> {
    enum class Mode { kTrain, kTest };
    CustomDataset(std::string path);
    // Override the get method to load custom data.
    torch::data::Example<> get(size_t index) override;
    // Return the length of data.
    torch::optional<size_t> size() const override;

private:
    int unique_words;
    torch::Tensor data;
    Mode mode_;
    torch::nn::Embedding embeddings;
};

#endif //CUSTOMDATASET_H
