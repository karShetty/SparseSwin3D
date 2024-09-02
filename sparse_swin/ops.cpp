#include <torch/extension.h>
#include <vector>
#include "ops.h"



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda_naive", &matmul_cuda_naive, "Compute matmul");
    m.def("matmul_grad_cuda_naive", &matmul_grad_cuda_naive, "Compute grad matmul");
    m.def("qkv_sparse_indices", &qkv_sparse_indices, "Compute indices");
    m.def("flatten_indices", &flatten_indices_impl, "Flatten indices");
    m.def("sort_indices", &sort_indices_impl, "Sort indices");
    m.def("return_inverse", &return_inverse_impl, "Return indices based on unique");
    m.def("unique_consecutive_indices", &unique_consecutive_indices, "Return indices for unique_consecutive");
    m.def("softmax_last_dim_forward", &softmax_last_dim_forward, "Softmax Forward");
    m.def("softmax_last_dim_backward", &softmax_last_dim_backward, "Softmax Backward");
    m.def("qkv_transpose", &qkv_transpose, "Custom Trasnpose for qkv");
}
