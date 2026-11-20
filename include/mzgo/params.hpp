// Views onto a model's learnable state, so the optimizer and the checkpoint
// code can work on any model without knowing its layers.
#pragma once
#include <cstddef>
#include <string>

namespace mzgo {

// One learnable tensor: its value, its gradient (same size), and whether
// weight decay applies (conv and linear weights only, not biases or batch norm).
struct ParamView  { std::string name; float* value; float* grad; size_t size; bool decay; };
// A tensor that is saved but not learned (batch norm running statistics).
struct BufferView { std::string name; float* value; size_t size; };

} // namespace mzgo
