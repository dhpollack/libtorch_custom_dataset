#pragma once
#include <torch/types.h>

template <typename InputOne = torch::Tensor, typename InputTwo = torch::Tensor,
	  typename Label = torch::Tensor>
struct ThreeTensorInput {
  using InputOneType = InputOne;
  using InputTwoType = InputTwo;
  using LabelType = Label;

  ThreeTensorInput() = default;
  ThreeTensorInput(InputOne inputone, InputTwo inputtwo, Label label)
      : inputone(std::move(inputone)), inputtwo(std::move(inputtwo)),
	label(std::move(label)) {}

  InputOne inputone;
  InputTwo inputtwo;
  Label label;
};
