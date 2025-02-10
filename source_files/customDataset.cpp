//
// Created by reitr on 2/9/2025.
//
#include "../header_files/customDataset.h"
#include <iostream>
#include <fstream>

CustomDataset::CustomDataset(std::string path) {
    std::fstream file;
    file.open(path);

    std::map<std::string , int> map;

    std::string word;
    int x = 0;
    while (file >> word) {
        int i = 0;

        for (char c : word) {
            if ((c < 'A' || c > 'Z') && (c < 'a' || c > 'z')) {
                word.erase(i, 1);
            }
            i++;
        }

        if (!map.count(word)) {
            map.insert(std::make_pair(word, 1));
        }else {
            map[word]++;
        }
    }

    unique_words = map.size();

    embeddings = torch::nn::Embedding(torch::nn::EmbeddingOptions(unique_words, 128).padding_idx(0));


    file.close();

}
