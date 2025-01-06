#include "../header_files/NN.h"

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

  NNet* net = new NNet(28, 10); // defines neural network

  net->to(device); //sets active device to CPU or GPU

  auto dataset = torch::data::datasets::MNIST(mnistPath) // loads MNIST dataset
    .map(torch::data::transforms::Stack<>());

  auto data_loader = torch::data::make_data_loader( //creates data batches for training
    std::move(dataset),
    torch::data::DataLoaderOptions().batch_size(64).workers(2));
  
  torch::optim::Adam net_optimizer(net->parameters(), torch::optim::AdamOptions(2e-4)); // optimizer for neural network
                                                                                                          // handles the models' learning/gradients

  int i = 0;

  for(torch::data::Example<>& batch : *data_loader){
    if(i == batches){
      break;
    }
    net->zero_grad();

    torch::Tensor outputs = net->forward(batch.data.to(device)); // this runs neural network

    outputs = outputs.squeeze(); // removes extra dimensions
    outputs = torch::mean(outputs, 1); // will change to find highest value later, currently just finds mean
    std::cout << "test 4" << std::endl;
    torch::Tensor loss = torch::binary_cross_entropy(outputs, batch.target).to(device); // calculates loss of neural network

    loss.backward(); //runs backpropagation
    net_optimizer.step(); //updates weights

    std::cout << outputs << '\n';
    i++;
  }

  
}