#include "custom_dataset.h"

using namespace std;

template <typename T>
CustomDataset<T>::CustomDataset(const vector<T> &examples)
    : examples_(examples) {}

template <typename T> T CustomDataset<T>::get(size_t index) {
  T ex = examples_[index];
  return std::move(ex);
}

template <typename T> torch::optional<size_t> CustomDataset<T>::size() const {
  torch::optional<size_t> sz(examples_.size());
  return sz;
}
