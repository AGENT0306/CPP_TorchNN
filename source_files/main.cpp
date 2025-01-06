#include "../header_files/NN.h"

const std::string mnistPath = "C:/Coding_Projects/C++/CPP_TorchNN/out/mnist";

const int batches = 1;

int main(){
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU!" << std::endl;
    device = torch::kCUDA;
  }else{
    std::cout << "ERROR!!! Training on CPU." << std::endl;
  }

  NNet* net = new NNet(28, 10);

  net->to(device);

  auto dataset = torch::data::datasets::MNIST(mnistPath)
    .map(torch::data::transforms::Stack<>());

  auto data_loader = torch::data::make_data_loader(
    std::move(dataset),
    torch::data::DataLoaderOptions().batch_size(64).workers(2));
  int i = 0;
  
  torch::optim::Adam net_optimizer(net->parameters(), torch::optim::AdamOptions(2e-4));
  //std::cout << "TEST 1!!" << '\n';
  for(torch::data::Example<>& batch : *data_loader){
    if(i == batches){
      break;
    }
    //net->zero_grad();
    std::cout << "Test 2" << '\n';
    torch::Tensor outputs = net->forward(batch.data.to(device));
    //std::cout << outputs << std::endl;
    outputs = outputs.squeeze();
    outputs = torch::mean(outputs, 1);
    std::cout << "Test 3" << '\n';
    std::cout << "outputs shape: " << outputs.sizes() << std::endl;
    std::cout << "batch.target shape: " << batch.target.sizes() << std::endl;
    //torch::Tensor loss = torch::binary_cross_entropy(outputs, batch.target).to(device);
    std::cout << "Test 4" << '\n';
    //loss.backward();
    //net_optimizer.step();
    
    
    //std::cout << batch.data;
    std::cout << outputs << '\n';
    //std::cout << loss << '\n';
    i++;
  }

  
}