#include "../header_files/NN_Mod.h"

int main(){
  Net net(4,5);
  for(const auto& p : net.parameters()){
    std::cout << p << std::endl;
  }
}