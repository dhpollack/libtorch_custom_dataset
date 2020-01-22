cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(libtorch-custom-dataset-template 
        VERSION 0.0.1 
        DESCRIPTION "a custom dataset with PyTorch's c++ frontend")

# add torch
find_package(Torch REQUIRED)

set(SOURCES
    custom_dataset.cpp
    main.cpp
    )

add_executable(libtorch-custom-dataset-template ${SOURCES})
target_link_libraries(libtorch-custom-dataset-template "${TORCH_LIBRARIES}")
