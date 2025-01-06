cmake_minimum_required(VERSION 3.5.0)
project(Neural_Net)

set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe")



file(GLOB SOURCES
    header_files/*.h
    source_files/*.cpp
    )

add_executable(Neural_Net ${SOURCES})

list(APPEND CMAKE_PREFIX_PATH "C:/Coding_Projects/C++/libtorch")
list(APPEND CMAKE_PREFIX_PATH "C:/Coding_Projects/C++/debugLibTorch")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} debug "${TORCH_DEBUG_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME}
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${TORCH_DLLS}

                 $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)

