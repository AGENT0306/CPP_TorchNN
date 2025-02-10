#include <fstream>

#include "../header_files/NN.h"
#include <iostream>

const std::string mnistPath = "../mnist";

const int batches = 1; // how many batches to train on


int main(){
  //checks if CUDA GPU is available
  //if not will set device to CPU
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU!" << std::endl;
    device = torch::kCUDA;
  }else{
    std::cout << "ERROR!!! Training on CPU." << std::endl;
  }



  return 0;
}

/*NNet* net = new NNet(28, 10); // defines neural network

  net->to(device); //sets active device to CPU or GPU

  auto dataset = torch::data::datasets::MNIST(mnistPath) // loads MNIST dataset
    .map(torch::data::transforms::Stack<>());

  auto data_loader = torch::data::make_data_loader( //creates data batches for training
    std::move(dataset),
    torch::data::DataLoaderOptions().batch_size(64).workers(2));

  int i = 0;

  for(torch::data::Example<>& batch : *data_loader){
    if(i == batches){
      break;
    }
    auto data = batch.data.to(device);

    for (int i = 0; i < data.size(0); i++) {
      data[i][0] = data[i][0].flatten();
      std::cout << data[i] << std::endl;
    }

    //data.flatten(2,3);
    //std::cout << data;
    i++;
  }*/