#pragma once

#include <torch/data/transforms.h>
#include <vector>

#include "custom_input_type.h"

template <>
struct torch::data::transforms::Stack<ThreeTensorInput<>> : public torch::data::transforms::Collation<ThreeTensorInput<>> {
  ThreeTensorInput<> apply_batch(std::vector<ThreeTensorInput<>> examples) override {
    std::vector<torch::Tensor> inputone, inputtwo, label;
    inputone.reserve(examples.size());
    inputtwo.reserve(examples.size());
    label.reserve(examples.size());
    for (auto &example : examples) {
      inputone.push_back(std::move(example.inputone));
      inputtwo.push_back(std::move(example.inputtwo));
      label.push_back(std::move(example.label));
    }
    return {torch::stack(inputone), torch::stack(inputtwo), torch::stack(label)};
  }
};

