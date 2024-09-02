#include <torch/extension.h>


std::tuple<at::Tensor, at::Tensor, std::vector<int32_t>, at::Tensor> matmul_cuda_naive(
    const at::Tensor& mat1_v,
    const at::Tensor& mat1_idx,
    const std::vector<int32_t>& mat1_shape,
    const at::Tensor& mat2_v,
    const at::Tensor& mat2_idx,
    const std::vector<int32_t>& mat2_shape,
    std::string mul_type
    );


at::Tensor matmul_grad_cuda_naive(
    const at::Tensor& mat1_v,
    const at::Tensor& mat2_v,
    const at::Tensor& _c,
    int32_t& C,
    std::string& mul_type
    );

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> qkv_sparse_indices(
    const at::Tensor indices,
    int& batch_size,
    int& spatial_dim,
    int& num_heads,
    int& c);

at::Tensor qkv_transpose(
    const at::Tensor& mat1_v,
    const at::Tensor& _c,
    int32_t& C,
    std::string& mul_type
  );


at::Tensor flatten_indices_impl(const at::Tensor& indices, const std::vector<int>& sizes);
at::Tensor sort_indices_impl(const at::Tensor& flattened_indices);
std::tuple<at::Tensor, int64_t> return_inverse_impl(const at::Tensor& x, int64_t maxnum);
at::Tensor unique_consecutive_indices(const at::Tensor& nums);

at::Tensor softmax_last_dim_forward(
    const at::Tensor& indices,
    const at::Tensor& values);

at::Tensor softmax_last_dim_backward(
    const at::Tensor& indices,
    const at::Tensor& values,
    const at::Tensor& grad_values);

