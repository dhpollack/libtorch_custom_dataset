# Custom Datasets with Libtorch

A dataset with custom inputs using the pytorch c++ frontend, libtorch.  This is based off the MNIST example, but we are replacing the `Example<>` type which has two members with a type that has three members.

# Requirements
* [Libtorch](https://pytorch.org/)

# Installation
```sh
git clone https://github.com/dhpollack/libtorch_custom_dataset.git
cd libtorch_custom_dataset
# only required if you need to install libtorch
mkdir third_party
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.4.0%2Bcpu.zip
unzip libtorch-shared-with-deps-1.4.0+cpu.zip && rm libtorch-shared-with-deps-1.4.0+cpu.zip
cd ..
# start here if you already have libtorch installed
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$(realpath ../third_party/libtorch) ..
# Torch_ROOT=$(realpath ../third_party/libtorch/share/cmake/Torch) cmake ..  # this also works
cmake --build .
./libtorch-custom-dataset-template
```
