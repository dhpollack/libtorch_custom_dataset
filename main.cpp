#include <iostream>
#include <torch/torch.h>

#include "custom_dataset.h"
#include "custom_input_type.h"
#include "custom_stack.h"

using namespace std;
using namespace torch;

int main() {
  int dataset_sz = 10;
  int batch_size = 3;

  vector<Tensor> ones, twos, labels;
  ones.reserve(dataset_sz);
  twos.reserve(dataset_sz);
  labels.reserve(dataset_sz);

  vector<ThreeTensorInput<>> examples_;
  for (int i = 0; i < dataset_sz; ++i) {
    ones.push_back(torch::rand({2, 3}));
    twos.push_back(torch::rand({3, 2}));
    labels.push_back(torch::randint(5, 1));
    ThreeTensorInput<> inp = {ones[i], twos[i], labels[i]};
    examples_.push_back(inp);
  }

  cout << examples_[0].inputone << "\n"
       << examples_[0].inputtwo << "\n"
       << examples_[0].label << endl;

  CustomDataset<> ds(examples_);
  auto ds_map = ds.map(data::transforms::Stack<ThreeTensorInput<>>());
  auto dl = data::make_data_loader<data::samplers::SequentialSampler>(
      move(ds_map), batch_size);
  for (auto &mb : *dl) {
    cout << mb.inputone << "\n" << mb.inputtwo << "\n" << mb.label << endl;
  }

  return 1;
}
