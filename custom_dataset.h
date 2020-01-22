#pragma once

#include <torch/data/datasets/base.h>
#include <torch/types.h>

#include "custom_input_type.h"

template <typename CustomSingleExample = ThreeTensorInput<>>
class CustomDataset
    : public torch::data::datasets::Dataset<CustomDataset<CustomSingleExample>,
                                            CustomSingleExample> {
public:
  using CustomExampleType = CustomSingleExample;
  // constructor
  explicit CustomDataset(const std::vector<CustomExampleType> &examples);
  // get item
  virtual CustomExampleType get(std::size_t index) override;
  // dataset size
  torch::optional<std::size_t> size() const override;
  // get all examples
  const std::vector<CustomExampleType> &examples() const;

private:
  std::vector<CustomExampleType> examples_;
};

// the following line is required for the linker to work correctly
template class CustomDataset<>; // add our custom example with default argument
