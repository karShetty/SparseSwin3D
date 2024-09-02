#include <vector>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/adjacent_difference.h>
#include "SparseCUDABlas.h"
#include <cusparse.h>


#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>


#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/cuda/Atomic.cuh>

template <
  typename policy_t, typename scalar_t,
  typename equal_t, typename not_equal_t
>
at::Tensor compute_unique_impl(
  const policy_t &policy,
  scalar_t *data,
  int64_t num_inp,
  const at::Tensor &sorted_indices,
  at::TensorOptions options,
  equal_t equal,
  not_equal_t not_equal
) {
  // inverse indices
    const int32_t *sorted_indices_ptr = sorted_indices.const_data_ptr<int32_t>();
    at::Tensor inv_loc = at::empty({num_inp}, options);
    at::Tensor inverse_indices = at::empty({num_inp}, options);
    int32_t* inv_loc_ptr = inv_loc.mutable_data_ptr<int32_t>();
    int32_t* inverse_indices_ptr = inverse_indices.mutable_data_ptr<int32_t>();
    thrust::adjacent_difference(policy, data, data + num_inp, inv_loc_ptr, not_equal);
    inv_loc[0] = 0;
    thrust::inclusive_scan(policy, inv_loc_ptr, inv_loc_ptr + num_inp, inv_loc_ptr);
    thrust::scatter(policy, inv_loc_ptr, inv_loc_ptr + num_inp, sorted_indices_ptr, inverse_indices_ptr);
    AT_CUDA_CHECK(cudaGetLastError());
    return inverse_indices;
}


template<typename scalar_t>
at::Tensor unique_consecutive_indices_impl(const at::Tensor& nums) {
    //Based of pytorch/aten/src/ATen/native/cuda/Unique.cu

    TORCH_CHECK(nums.dim() == 1);
    at::Tensor inverse_indices;

    scalar_t* nums_data = nums.data_ptr<scalar_t>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    at::cuda::ThrustAllocator allocator;
    auto policy = thrust::cuda::par(allocator).on(stream);
    int64_t num_inp = nums.size(0);

    auto options = at::TensorOptions().dtype(at::kInt).device(nums.device());
    at::Tensor sorted_indices = at::arange(0, num_inp, options);
    int32_t *sorted_indices_ptr = sorted_indices.mutable_data_ptr<int32_t>();


    inverse_indices = compute_unique_impl(
        policy, nums_data, num_inp, sorted_indices,
        options,
        thrust::equal_to<scalar_t>(),
        thrust::not_equal_to<scalar_t>()
    );
    return inverse_indices;
}




at::Tensor unique_consecutive_indices(const at::Tensor& nums) {
    //Based of pytorch/aten/src/ATen/native/cuda/Unique.cu

    TORCH_CHECK(nums.dim() == 1);
    at::Tensor inverse_indices;

    AT_DISPATCH_ALL_TYPES(
        nums.scalar_type(), "unique_consecutive_indices_impl", [&] {
            inverse_indices = unique_consecutive_indices_impl<scalar_t>(
            nums);
    });
    return inverse_indices;
}



template <typename scalar_t>
__global__ void scatter_reduce_max(scalar_t* src_max, const scalar_t* vals, const int32_t* idx_row, int64_t nnz) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        gpuAtomicMax(&src_max[idx_row[idx]], vals[idx]);
    }
}


template <typename scalar_t>
__global__ void scatter_add(scalar_t* out_sum, scalar_t* out, int32_t* idx_row, int32_t nnz, const int32_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        at::native::fastAtomicAdd(out_sum, idx_row[idx], N, out[idx], true);
    }
}





at::Tensor softmax_last_dim_forward(
    const at::Tensor& indices,
    const at::Tensor& values) {

    auto device = values.device();
    auto dtype = values.scalar_type();
    int64_t nnz = values.size(0);
    int32_t N = indices[-1].item<int32_t>() + 1;//assuming it is already sorted

    torch::Tensor src_max = torch::zeros({N}, values.options());
    torch::Tensor out = torch::empty_like(values);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "scatter_reduce_max", ([&] {
        scalar_t* src_max_ptr = src_max.data_ptr<scalar_t>();
        scalar_t* values_ptr = values.data_ptr<scalar_t>();
        int32_t* idx_row_ptr = indices.data_ptr<int32_t>();

        scatter_reduce_max<<<(nnz + 255) / 256, 256>>>(
            src_max_ptr, values_ptr, idx_row_ptr, nnz);
    }));

    torch::Tensor max_values = src_max.index_select(0, indices);
    out = (values - max_values).exp();

    // Scatter add for normalization
    torch::Tensor out_sum = torch::zeros({N}, out.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(out.scalar_type(), "scatter_add", ([&] {
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        scalar_t* out_sum_ptr = out_sum.data_ptr<scalar_t>();
        int32_t* idx_row_ptr = indices.data_ptr<int32_t>();

        scatter_add<<<(nnz + 255) / 256, 256>>>(
            out_sum_ptr, out_ptr, idx_row_ptr, nnz, N);
    }));

    out_sum += 1e-8;
    auto norm_out = out / out_sum.index_select(0, indices);
    return norm_out;
}

at::Tensor softmax_last_dim_backward(
    const at::Tensor& indices,
    const at::Tensor& values,
    const at::Tensor& grad_values
    ) {

    auto device = values.device();
    auto dtype = values.scalar_type();
    int64_t nnz = values.size(0);
    int32_t N = indices[-1].item<int32_t>() + 1;//assuming it is already sorted

    at::Tensor grad_input = grad_values * values;
    at::Tensor sum_grad_input = torch::zeros_like(values);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "scatter_add", ([&] {
        scalar_t* out_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t* out_sum_ptr = sum_grad_input.data_ptr<scalar_t>();
        int32_t* idx_row_ptr = indices.data_ptr<int32_t>();

        scatter_add<<<(nnz + 255) / 256, 256>>>(
            out_sum_ptr, out_ptr, idx_row_ptr, nnz, N);
    }));

    at::Tensor grad_out = grad_input - values * sum_grad_input.index_select(0, indices);
    return grad_out;
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////




__global__ void expand_indices_kernel(const int32_t* __restrict__ indices,
                                      int32_t* __restrict__ expanded_indices_0,
                                      int32_t* __restrict__ expanded_indices_1,
                                      int32_t* __restrict__ depth_indices,
                                      int32_t* __restrict__ channel_indices,
                                      int num_indices, int num_heads, int c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = num_indices * num_heads * c;

    if (idx < total_size) {
        int batch_idx = idx / (num_heads * c);   // index along the batch dimension
        int local_idx = idx % (num_heads * c);   // index within the repeated part
        int depth_idx = local_idx / c;           // index along the head dimension
        int channel_idx = local_idx % c;         // index along the channel dimension

        int base_index = batch_idx * 2;
        expanded_indices_0[idx] = __ldg(&indices[base_index]);
        expanded_indices_1[idx] = __ldg(&indices[base_index + 1]);
        depth_indices[idx] = depth_idx;
        channel_indices[idx] = channel_idx;
    }
}




at::Tensor flatten_indices_impl(const at::Tensor& indices, const std::vector<int>& sizes) {
    auto device = indices.device();
    auto num_dims = sizes.size();
    auto num_indices = indices.size(1);

    std::vector<int32_t> strides(num_dims - 1);
    std::vector<int32_t> flipped_sizes(sizes.begin() + 1, sizes.end());
    std::reverse(flipped_sizes.begin(), flipped_sizes.end());
    strides[0] = flipped_sizes[0];
    for (int i = 1; i < num_dims - 1; ++i) {
        strides[i] = strides[i - 1] * flipped_sizes[i];
    }
    std::reverse(strides.begin(), strides.end());
    strides.push_back(1);  // Append 1 for the last dimension

    at::Tensor d_strides = torch::tensor(strides, torch::dtype(torch::kInt32).device(device));
    auto hash_indices = torch::empty({num_indices}, torch::dtype(torch::kInt32).device(device));


    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    at::cuda::ThrustAllocator allocator;
    auto policy = thrust::cuda::par(allocator).on(stream);

    auto indices_ptr = indices.data_ptr<int32_t>();
    auto strides_ptr = d_strides.data_ptr<int32_t>();
    auto hash_indices_ptr = hash_indices.data_ptr<int32_t>();

    thrust::for_each(
        policy,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(num_indices),
        [=] __device__(int idx) {
            int32_t hash = 0;
            for (int i = 0; i < num_dims; ++i) {
                hash += indices_ptr[i * num_indices + idx] * strides_ptr[i];
            }
            hash_indices_ptr[idx] = hash;
        }
    );

    return hash_indices;
}


at::Tensor sort_indices_impl(const at::Tensor& flattened_indices) {
    auto device = flattened_indices.device();
    auto num_elements = flattened_indices.numel();

    thrust::device_vector<int32_t> indices(num_elements);
    thrust::sequence(thrust::device, indices.begin(), indices.end());
    at::Tensor sorted_indices_tensor = torch::empty({num_elements}, torch::dtype(torch::kInt32).device(device));

    thrust::device_vector<int32_t> flattened_indices_th(flattened_indices.data_ptr<int32_t>(),
                                                        flattened_indices.data_ptr<int32_t>() + num_elements);
    thrust::sort_by_key(thrust::device, flattened_indices_th.begin(), flattened_indices_th.end(), indices.begin());
    thrust::copy(indices.begin(), indices.end(), sorted_indices_tensor.data_ptr<int32_t>());
    return sorted_indices_tensor;
}



std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> qkv_sparse_indices(const at::Tensor indices, int& batch_size, 
                                                                           int& spatial_dim, int& num_heads,
                                                                           int& c) {
    int num_indices = indices.size(0);

    auto device = indices.device();

    auto _indices_qv = torch::empty({4, num_indices * num_heads * c}, torch::dtype(torch::kInt32).device(device));
    auto _indices_k = torch::empty({4, num_indices * num_heads * c}, torch::dtype(torch::kInt32).device(device));

    auto expanded_indices_0_qv = _indices_qv.select(0, 0);
    auto expanded_indices_1_qv = _indices_qv.select(0, 2);
    auto depth_indices_qv = _indices_qv.select(0, 1);
    auto channel_indices_qv = _indices_qv.select(0, 3);

    auto expanded_indices_0_k = _indices_k.select(0, 0);
    auto expanded_indices_1_k = _indices_k.select(0, 3);//transpose for k (-1,-2)
    auto depth_indices_k = _indices_k.select(0, 1);
    auto channel_indices_k = _indices_k.select(0, 2);

    int threads = 256;
    int blocks = (num_indices * num_heads * c + threads - 1) / threads;

    expand_indices_kernel<<<blocks, threads>>>(
        indices.data_ptr<int32_t>(),
        expanded_indices_0_qv.data_ptr<int32_t>(),
        expanded_indices_1_qv.data_ptr<int32_t>(),
        depth_indices_qv.data_ptr<int32_t>(),
        channel_indices_qv.data_ptr<int32_t>(),
        num_indices, num_heads, c
    );

    expanded_indices_0_k.copy_(expanded_indices_0_qv);
    expanded_indices_1_k.copy_(expanded_indices_1_qv);
    depth_indices_k.copy_(depth_indices_qv);
    channel_indices_k.copy_(channel_indices_qv);

    auto flattened_indices_qv = flatten_indices_impl(_indices_qv, {batch_size, num_heads, spatial_dim, c});
    auto sorted_indices_qv = sort_indices_impl(flattened_indices_qv);
    auto flattened_indices_k = flatten_indices_impl(_indices_k, {batch_size, num_heads, c, spatial_dim});
    auto sorted_indices_k = sort_indices_impl(flattened_indices_k);
    return {_indices_qv, sorted_indices_qv, _indices_k, sorted_indices_k};
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////



template <typename scalar_t>
__global__ void indices_kernel_q_k(const int32_t* __restrict__ _c,
                                   const int32_t* __restrict__ idx_starts,
                                   const int32_t* __restrict__ pos_starts,
                                   const int32_t* __restrict__ pos_starts_mat,
                                   int32_t* __restrict__ col_indices,
                                   int32_t* __restrict__ row_indices,
                                   scalar_t* __restrict__ val,
                                   const scalar_t* __restrict__ m1_val,
                                   const scalar_t* __restrict__ m2_val_t,
                                   int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // grid dimension for c
    int i_dim = blockIdx.y * blockDim.y + threadIdx.y; // grid dimension for i
    int j_dim = blockIdx.z * blockDim.z + threadIdx.z; // grid dimension for j

    if (idx < N) {
        int repeat_count = _c[idx];

        if (i_dim < repeat_count && j_dim < repeat_count){
            int start = idx_starts[idx];
            int pos_start = pos_starts[idx];
            int pos_start_mat = pos_starts_mat[idx];

            int pos = pos_start + i_dim * repeat_count;

            col_indices[pos + j_dim] = j_dim + start;
            row_indices[pos + j_dim] = i_dim + start;
            scalar_t sum = 0;
            for (int k = 0; k < C; ++k){
                sum += (m1_val[pos_start_mat + i_dim * C + k] * m2_val_t[pos_start_mat + j_dim * C + k]);
            }
            val[pos + j_dim] = sum;
        }
    }
}


template <typename scalar_t>
__global__ void indices_kernel_qk_v(const int32_t* __restrict__ _c,
                                   const int32_t* __restrict__ idx_starts,
                                   const int32_t* __restrict__ pos_starts_out,
                                   const int32_t* __restrict__ pos_starts_mat1,
                                   const int32_t* __restrict__ pos_starts_mat2,
                                   int32_t* __restrict__ col_indices,
                                   int32_t* __restrict__ row_indices,
                                   scalar_t* __restrict__ val,
                                   const scalar_t* __restrict__ m1_val,
                                   const scalar_t* __restrict__ m2_val_t,
                                   int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // grid dimension for c
    int i_dim = blockIdx.y * blockDim.y + threadIdx.y; // grid dimension for i
    int j_dim = blockIdx.z * blockDim.z + threadIdx.z; // grid dimension for j

    if (idx < N) {
        int repeat_count = _c[idx];

        if (i_dim < repeat_count && j_dim < C){
            int start = idx_starts[idx];
            int start_c = idx * C;
            int pos_start = pos_starts_out[idx];
            int pos_start_mat1 = pos_starts_mat1[idx];
            int pos_start_mat2 = pos_starts_mat2[idx];

            int pos = pos_start + i_dim * C;

            col_indices[pos + j_dim] = j_dim + start_c;
            row_indices[pos + j_dim] = i_dim + start;
            scalar_t sum = 0;
            for (int k = 0; k < repeat_count; ++k){
                sum += (m1_val[pos_start_mat1 + i_dim * repeat_count + k] * m2_val_t[pos_start_mat2 + j_dim * repeat_count + k]);
            }
            val[pos + j_dim] = sum;
        }
    }
}



template <typename scalar_t>
__global__ void mul_grad_rc_cr(const int32_t* __restrict__ _c,
                               const int32_t* __restrict__  pos_starts_out,
                               const int32_t* __restrict__  pos_starts_mat1,
                               const int32_t* __restrict__  pos_starts_mat2,
                               scalar_t* __restrict__  val,
                               const scalar_t* __restrict__  m1_val,
                               const scalar_t* __restrict__  m2_val_t,
                               int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // grid dimension for c
    int i_dim = blockIdx.y * blockDim.y + threadIdx.y; // grid dimension for i
    int j_dim = blockIdx.z * blockDim.z + threadIdx.z; // grid dimension for j

    if (idx < N) {
        int repeat_count = _c[idx];

        if (i_dim < repeat_count && j_dim < repeat_count){
            int pos_start_out = pos_starts_out[idx];
            int pos_start_mat1 = pos_starts_mat1[idx];
            int pos_start_mat2 = pos_starts_mat2[idx];
            int pos = pos_start_out + i_dim * repeat_count;

            scalar_t sum = 0;
            for (int k = 0; k < C; ++k){
                sum += (m1_val[pos_start_mat1 + i_dim * C + k] * m2_val_t[pos_start_mat2 + j_dim * C + k]);
            }
            val[pos + j_dim] = sum;
        }
    }
}


template <typename scalar_t>
__global__ void mul_grad_rr_rc(const int32_t* __restrict__  _c,
                               const int32_t* __restrict__  pos_starts_out,
                               const int32_t* __restrict__  pos_starts_mat1,
                               const int32_t* __restrict__  pos_starts_mat2,
                               scalar_t* __restrict__  val,
                               const scalar_t* __restrict__  m1_val,
                               const scalar_t* __restrict__  m2_val_t,
                               int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // grid dimension for c
    int i_dim = blockIdx.y * blockDim.y + threadIdx.y; // grid dimension for i
    int j_dim = blockIdx.z * blockDim.z + threadIdx.z; // grid dimension for j

    if (idx < N) {
        int repeat_count = _c[idx];

        if (i_dim < repeat_count && j_dim < C){
            int pos_start_out = pos_starts_out[idx];
            int pos_start_mat1 = pos_starts_mat1[idx];
            int pos_start_mat2 = pos_starts_mat2[idx];
            int pos = pos_start_out + i_dim * C;

            scalar_t sum = 0;
            for (int k = 0; k < repeat_count; ++k){
                sum += (m1_val[pos_start_mat1 + i_dim * repeat_count + k] * m2_val_t[pos_start_mat2 + j_dim * repeat_count + k]);
            }
            val[pos + j_dim] = sum;
        }
    }
}


template <typename scalar_t>
__global__ void mul_grad_cr_rr(const int32_t* __restrict__  _c,
                               const int32_t* __restrict__  pos_starts_out,
                               const int32_t* __restrict__  pos_starts_mat1,
                               const int32_t* __restrict__  pos_starts_mat2,
                               scalar_t* __restrict__  val,
                               const scalar_t* __restrict__  m1_val,
                               const scalar_t* __restrict__  m2_val_t,
                               int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // grid dimension for c
    int i_dim = blockIdx.y * blockDim.y + threadIdx.y; // grid dimension for i
    int j_dim = blockIdx.z * blockDim.z + threadIdx.z; // grid dimension for j

    if (idx < N) {
        int repeat_count = _c[idx];

        if (i_dim < C && j_dim < repeat_count){
            int pos_start_out = pos_starts_out[idx];
            int pos_start_mat1 = pos_starts_mat1[idx];
            int pos_start_mat2 = pos_starts_mat2[idx];
            int pos = pos_start_out + i_dim * repeat_count;

            scalar_t sum = 0;
            for (int k = 0; k < repeat_count; ++k){
                sum += (m1_val[pos_start_mat1 + i_dim * repeat_count + k] * m2_val_t[pos_start_mat2 + j_dim * repeat_count + k]);
            }
            val[pos + j_dim] = sum;
        }
    }
}


template <typename scalar_t>
__global__ void transpose_rc(const int32_t* __restrict__  _c,
                             const int32_t* __restrict__  pos_starts_out,
                             scalar_t* __restrict__  val,
                             const scalar_t* __restrict__  m1_val,
                             int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // grid dimension for c
    int i_dim = blockIdx.y * blockDim.y + threadIdx.y; // grid dimension for i
    int j_dim = blockIdx.z * blockDim.z + threadIdx.z; // grid dimension for j

    if (idx < N) {
        int repeat_count = _c[idx];

        if (i_dim < C && j_dim < repeat_count){
            int pos_start = pos_starts_out[idx];
            val[pos_start + i_dim * repeat_count + j_dim] = m1_val[pos_start + i_dim + j_dim * C];
        }
    }
}


template <typename scalar_t>
__global__ void transpose_rr(const int32_t* __restrict__  _c,
                             const int32_t* __restrict__  pos_starts_out,
                             scalar_t* __restrict__  val,
                             const scalar_t* __restrict__  m1_val,
                             int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // grid dimension for c
    int i_dim = blockIdx.y * blockDim.y + threadIdx.y; // grid dimension for i
    int j_dim = blockIdx.z * blockDim.z + threadIdx.z; // grid dimension for j

    if (idx < N) {
        int repeat_count = _c[idx];

        if (i_dim < repeat_count && j_dim < repeat_count){
            int pos_start = pos_starts_out[idx];
            val[pos_start + i_dim * repeat_count + j_dim] = m1_val[pos_start + i_dim + j_dim * repeat_count];
        }
    }
}


at::Tensor _to_csr_int_naive(const at::Tensor& rowIndices, int64_t dim, int64_t nnz) {
  at::Tensor csr = at::empty({dim + 1}, at::CUDA(at::kInt));
  at::Tensor rowIndicesInt = at::empty({rowIndices.size(0)}, at::CUDA(at::kInt));
  rowIndicesInt.copy_(rowIndices);
  at::native::sparse::cuda::Xcoo2csr(
      rowIndicesInt.data_ptr<int32_t>(), nnz, dim, csr.data_ptr<int32_t>());
  return csr;
}




// csrMatrixRefNaive is used to have a representation of a raw CSR matrix
// representation comming from `sparse_sparse_matmul_cuda_kernel` function.
// Moreover this implements a RAII guard for a cusparse descriptor
template <class scalar_t>
struct csrMatrixRefNaive {
  int* csr_indices_{nullptr};
  int* csr_pointers_{nullptr};
  scalar_t* csr_values_{nullptr};
  int nnz_{0};
  std::vector<int> size_{};

  cusparseSpMatDescr_t description_{0};

  csrMatrixRefNaive() {
  }

  csrMatrixRefNaive(
      int* csr_indices,
      int* csr_pointers,
      scalar_t* csr_values,
      int nnz,
      const std::vector<int>& size)
      : csr_indices_{csr_indices},
        csr_pointers_{csr_pointers},
        csr_values_{csr_values},
        nnz_{nnz},
        size_{size} {
    cudaDataType cuda_data_type = at::cuda::getCudaDataType<scalar_t>();
    TORCH_CUDASPARSE_CHECK(cusparseCreateCsr(
        &description_,
        this->size(0),
        this->size(1),
        this->nnz_,
        this->csr_pointers_,
        this->csr_indices_,
        this->csr_values_,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        cuda_data_type));
  }

  ~csrMatrixRefNaive() {
    cusparseDestroySpMat(description_);
  }

  int size(int index) const {
    return size_.at(index);
  }
};



template <typename scalar_t>
void get_csr_transpose_naive(const csrMatrixRefNaive<scalar_t> &csr, int* csrt_row, int* csrt_col, scalar_t* csrt_values){
    cudaDataType cuda_data_type = at::cuda::getCudaDataType<scalar_t>();
    auto handle = at::cuda::getCurrentCUDASparseHandle();
    size_t buffer_size;


    TORCH_CUDASPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
        handle,
        csr.size_[0], csr.size_[1], csr.nnz_,
        csr.csr_values_, csr.csr_pointers_, csr.csr_indices_,
        csrt_values, csrt_row, csrt_col,
        cuda_data_type,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        &buffer_size));


    auto& allocator = *c10::cuda::CUDACachingAllocator::get();
    auto dataPtr = allocator.allocate(buffer_size);
    void *buffer_temp = NULL;
    buffer_temp = dataPtr.get();

    TORCH_CUDASPARSE_CHECK(cusparseCsr2cscEx2(
        handle,
        csr.size_[0], csr.size_[1], csr.nnz_,
        csr.csr_values_, csr.csr_pointers_, csr.csr_indices_,
        csrt_values, csrt_row, csrt_col,
        cuda_data_type,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        buffer_temp));

}


template <typename scalar_t>
at::Tensor qkv_transpose_kernel(
    const at::Tensor& mat1_v,
    const at::Tensor& _c,
    int32_t& C,
    std::string& mul_type
    ) {

        int32_t N = _c.size(0);
        int _c_max = _c.max().item<int64_t>();

        int xThreads = 1;  // moderate size for the idx loop
        int yThreads = 16;    // lower size for the i loop
        int zThreads = 16;    // lower size for the j loop


        at::Tensor output_values;
        if (mul_type == "rc"){
            auto pos_starts_out = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * C).slice(0, 0, N-1), 0, at::kInt)});
            output_values = torch::empty_like(mat1_v);

            dim3 threadsPerBlock(xThreads, yThreads, zThreads);
            dim3 numBlocks((N + xThreads - 1) / xThreads,
                   (C + yThreads - 1) / yThreads,
                   (_c_max + zThreads - 1) / zThreads);
            transpose_rc<scalar_t><<<numBlocks, threadsPerBlock>>>(
                _c.data_ptr<int32_t>(),
                pos_starts_out.data_ptr<int32_t>(),
                output_values.data_ptr<scalar_t>(),
                mat1_v.data_ptr<scalar_t>(),
                N, C
            );
        }
        else if (mul_type == "rr"){
            auto pos_starts_out = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * _c).slice(0, 0, N-1), 0, at::kInt)});
            output_values = torch::empty_like(mat1_v);
            dim3 threadsPerBlock(xThreads, yThreads, zThreads);
            dim3 numBlocks((N + xThreads - 1) / xThreads,
                   (_c_max + yThreads - 1) / yThreads,
                   (_c_max + zThreads - 1) / zThreads);
            transpose_rr<scalar_t><<<numBlocks, threadsPerBlock>>>(
                _c.data_ptr<int32_t>(),
                pos_starts_out.data_ptr<int32_t>(),
                output_values.data_ptr<scalar_t>(),
                mat1_v.data_ptr<scalar_t>(),
                N, C
            );
        }
        else
            TORCH_CHECK(false, "Invalid mul_type: ", mul_type, ". Expected 'rc', or 'rr'.");

        return output_values;
}


std::tuple<at::Tensor, int64_t> return_inverse_impl(const at::Tensor& x, int64_t maxnum = -1) {
    //equivalent to unique with return inverse;
    if (maxnum == -1) {
        maxnum = x.max().item<int64_t>() + 1;
    }
    at::Tensor p = torch::zeros({maxnum}, at::TensorOptions().dtype(torch::kBool).device(x.device()));
    p.index_put_({x}, true);
    int64_t c = p.sum().item<int64_t>();
    at::Tensor p2 = torch::empty({maxnum}, x.options());
    p2.index_copy_(0,  p.nonzero().squeeze(1), torch::arange(c, x.options()));
    at::Tensor out = p2.index_select(0, x);
    return std::make_tuple(out, c);
}

template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t> reshape_sparse(const at::Tensor sparse_tensor_v,
                                                                                            const at::Tensor sparse_tensor_idx,
                                                                                            const std::vector<int32_t> sparse_tensor_shape
                                                                                            ) {
    int32_t x1 = sparse_tensor_shape[0], x2 = sparse_tensor_shape[1], x3 = sparse_tensor_shape[2], x4 = sparse_tensor_shape[3];
    auto indices = sparse_tensor_idx;

    at::Tensor new_row_indices = at::empty({sparse_tensor_idx.size(1)},
                                           at::TensorOptions().dtype(at::kInt).device(sparse_tensor_idx.device()));
    at::Tensor new_col_indices = at::empty({sparse_tensor_idx.size(1)},
                                           at::TensorOptions().dtype(at::kInt).device(sparse_tensor_idx.device()));

    thrust::device_ptr<int32_t> row_ptr(new_row_indices.data_ptr<int32_t>());
    thrust::device_ptr<int32_t> col_ptr(new_col_indices.data_ptr<int32_t>());

    thrust::device_ptr<int32_t> indices_0_ptr(sparse_tensor_idx[0].data_ptr<int32_t>());
    thrust::device_ptr<int32_t> indices_1_ptr(sparse_tensor_idx[1].data_ptr<int32_t>());
    thrust::device_ptr<int32_t> indices_2_ptr(sparse_tensor_idx[2].data_ptr<int32_t>());
    thrust::device_ptr<int32_t> indices_3_ptr(sparse_tensor_idx[3].data_ptr<int32_t>());

    // Create a zip iterator over the input and output tensors
    auto begin = thrust::make_zip_iterator(
        thrust::make_tuple(row_ptr, col_ptr, indices_0_ptr, indices_1_ptr, indices_2_ptr, indices_3_ptr)
    );
    auto end = begin + sparse_tensor_idx.size(1);

    // Run the calculation in parallel using a lambda function
    thrust::for_each(begin, end, [=] __device__ (auto t) {
        //new_row_indices = indices[0] * x2 * x3 + indices[1] * x3 + indices[2];
        int32_t row = static_cast<int32_t>(thrust::get<2>(t)) * x2 * x3 + static_cast<int32_t>(thrust::get<3>(t)) * x3 + static_cast<int32_t>(thrust::get<4>(t));
        //new_col_indices = indices[0] * x2 * x4 + indices[1] * x4 + indices[3];
        int32_t col = static_cast<int32_t>(thrust::get<2>(t)) * x2 * x4 + static_cast<int32_t>(thrust::get<3>(t)) * x4 + static_cast<int32_t>(thrust::get<5>(t));

        thrust::get<0>(t) = row;
        thrust::get<1>(t) = col;
    });


    //auto new_row_indices = indices[0] * x2 * x3 + indices[1] * x3 + indices[2];
    //auto new_col_indices = indices[0] * x2 * x4 + indices[1] * x4 + indices[3];
    at::Tensor r_indices0, r_indices1;
    int64_t len_0, len_1;
    r_indices0 = std::get<1>(at::unique_consecutive(new_row_indices, true));
    len_0 = r_indices0[-1].item<int64_t>() + 1;
    std::tie(r_indices1, len_1) = return_inverse_impl(new_col_indices);
    return std::make_tuple(new_row_indices, new_col_indices, r_indices0, r_indices1, len_0, len_1);
}

at::Tensor bincount_(at::Tensor a) {
        auto y = at::bincount(a).to(torch::kInt);
        return y.index_select(0, at::nonzero(y).squeeze(1));
}



template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor, std::vector<int32_t>, at::Tensor> matmul_cuda_naive_kernel(
    const at::Tensor& mat1_v,
    const at::Tensor& mat1_idx,
    const std::vector<int32_t>& mat1_shape,
    const at::Tensor& mat2_v,
    const at::Tensor& mat2_idx,
    const std::vector<int32_t>& mat2_shape,
    std::string mul_type
    ) {
        at::Tensor mat1_row_indices, mat1_col_indices, mat2_row_indices, mat2_col_indices, r_indices0_a, r_indices1_a, r_indices0_b, r_indices1_b;
        int32_t len_0_a, len_0_b, len_1_a, len_1_b;
        std::tie(mat1_row_indices, mat1_col_indices, r_indices0_a, r_indices1_a, len_0_a, len_1_a) = reshape_sparse<scalar_t>(mat1_v, mat1_idx, mat1_shape);
        std::tie(mat2_row_indices, mat2_col_indices, r_indices0_b, r_indices1_b, len_0_b, len_1_b) = reshape_sparse<scalar_t>(mat2_v, mat2_idx, mat2_shape);

        //ToDo: Currently converting to CSR and then transpose; use qkv_transpose_kernel to do it
        at::Tensor mat2_indptr = _to_csr_int_naive(r_indices0_b, len_0_b, r_indices0_b.size(0));
        csrMatrixRefNaive<scalar_t> csr_mat2(
          r_indices1_b.data_ptr<int>(),
          mat2_indptr.data_ptr<int>(),
          mat2_v.data_ptr<scalar_t>(),
          (int)r_indices0_b.size(0),
          {(int)len_0_b, (int)len_1_b});

        //ToDo: Write custom unique_consecutive
        at::Tensor key_rows = std::get<0>(at::unique_consecutive(mat1_row_indices));
        at::Tensor key_cols;
        if (mul_type == "q_k")
            key_cols = key_rows;
        else if (mul_type == "qk_v")
            key_cols = std::get<0>(at::_unique(mat2_col_indices, false));
        else
            TORCH_CHECK(false, "Invalid mul_type: ", mul_type, ". Expected 'q_k' or 'qk_v'.");

        at::Tensor csrt_col_a, csrt_row_a, csrt_values_a, csrt_col_b, csrt_row_b, csrt_values_b;
        {
            //ToDo: Currently converting to CSR and then transpose; use qkv_transpose_kernel to do it
            csrt_col_b =  at::empty({csr_mat2.nnz_}, at::CUDA(at::kInt));
            csrt_row_b =  at::empty({csr_mat2.size_[1] + 1}, at::CUDA(at::kInt));
            csrt_values_b = torch::empty_like(mat2_v);
            get_csr_transpose_naive(csr_mat2, csrt_row_b.data_ptr<int>(), csrt_col_b.data_ptr<int>(), csrt_values_b.data_ptr<scalar_t>());
        }


        at::Tensor output_values, result_indices, row_indices, col_indices, _c;
        if (mul_type == "q_k")
        {
            int32_t C = mat1_shape[3];
            _c = bincount_(mat1_col_indices.reshape({-1, mat1_shape[3]}).select(1, 0));
            int32_t N = _c.size(0);
            auto idx_starts = torch::cat({torch::zeros(1, _c.options()), torch::cumsum(_c.slice(0, 0, N-1), 0, at::kInt)});
            auto pos_starts = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * _c).slice(0, 0, N-1), 0, at::kInt)});
            auto pos_starts_mat = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * C).slice(0, 0, N-1), 0, at::kInt)});
            auto total_size = torch::sum(_c * _c).item<int64_t>();
            result_indices = torch::empty({2, total_size}, _c.options()).to(at::kInt);
            output_values = torch::empty({total_size}, mat2_v.options());
            row_indices =  result_indices.select(0, 0);
            col_indices =  result_indices.select(0, 1);

            int xThreads = 1;  // moderate size for the idx loop
            int yThreads = 16;    // lower size for the i loop
            int zThreads = 16;    // lower size for the j loop
            int _c_max = _c.max().item<int64_t>();
            dim3 threadsPerBlock(xThreads, yThreads, zThreads);
            dim3 numBlocks((N + xThreads - 1) / xThreads,
                   (_c_max + yThreads - 1) / yThreads,
                   (_c_max + zThreads - 1) / zThreads);
            indices_kernel_q_k<scalar_t><<<numBlocks, threadsPerBlock>>>(
                _c.data_ptr<int32_t>(),
                idx_starts.data_ptr<int32_t>(),
                pos_starts.data_ptr<int32_t>(),
                pos_starts_mat.data_ptr<int32_t>(),
                col_indices.data_ptr<int32_t>(),
                row_indices.data_ptr<int32_t>(),
                output_values.data_ptr<scalar_t>(),
                mat1_v.data_ptr<scalar_t>(),
                csrt_values_b.data_ptr<scalar_t>(),
                N, C
            );
        }
        else if (mul_type == "qk_v"){
            int32_t C = mat2_shape[3];
            _c = bincount_(mat2_col_indices.reshape({-1, mat2_shape[3]}).select(1, 0));
            int32_t N = _c.size(0);
            auto idx_starts = torch::cat({torch::zeros(1, _c.options()), torch::cumsum(_c.slice(0, 0, N-1), 0, at::kInt)});
            auto pos_starts_out = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * C).slice(0, 0, N-1), 0, at::kInt)});
            auto pos_starts_mat1 = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * _c).slice(0, 0, N-1), 0, at::kInt)});
            auto pos_starts_mat2 = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * C).slice(0, 0, N-1), 0, at::kInt)});
            auto total_size = torch::sum(_c * C).item<int64_t>();
            result_indices = torch::empty({2, total_size}, _c.options()).to(at::kInt);
            output_values = torch::empty({total_size}, mat2_v.options());
            row_indices =  result_indices.select(0, 0);
            col_indices =  result_indices.select(0, 1);

            int xThreads = 1;  // moderate size for the idx loop
            int yThreads = 16;    // lower size for the i loop
            int zThreads = 16;    // lower size for the j loop
            int _c_max = _c.max().item<int64_t>();
            dim3 threadsPerBlock(xThreads, yThreads, zThreads);
            dim3 numBlocks((N + xThreads - 1) / xThreads,
                           (_c_max + yThreads - 1) / yThreads,
                           (C + zThreads - 1) / zThreads);
            indices_kernel_qk_v<scalar_t><<<numBlocks, threadsPerBlock>>>(
                _c.data_ptr<int32_t>(),
                idx_starts.data_ptr<int32_t>(),
                pos_starts_out.data_ptr<int32_t>(),
                pos_starts_mat1.data_ptr<int32_t>(),
                pos_starts_mat2.data_ptr<int32_t>(),
                col_indices.data_ptr<int32_t>(),
                row_indices.data_ptr<int32_t>(),
                output_values.data_ptr<scalar_t>(),
                mat1_v.data_ptr<scalar_t>(),
                csrt_values_b.data_ptr<scalar_t>(),
                N, C
            );
        }
        else
            TORCH_CHECK(false, "Invalid mul_type: ", mul_type, ". Expected 'q_k', or 'qk_v'.");



        at::Tensor sparse_mul_rows = at::empty({row_indices.size(0)}, key_rows.options());
        at::Tensor sparse_mul_cols = at::empty({col_indices.size(0)}, key_cols.options());
        at::index_select_out(sparse_mul_rows, key_rows, 0, row_indices);
        at::index_select_out(sparse_mul_cols, key_cols, 0, col_indices);

        int32_t x1 = mat1_shape[0], x2 = mat1_shape[1], x3 = mat1_shape[2], x4 = mat2_shape[3];
        at::Tensor output_indices_ = at::empty({4, sparse_mul_rows.size(0)}, mat1_idx.options());

        thrust::device_ptr<int32_t> row_ptr(sparse_mul_rows.data_ptr<int32_t>());
        thrust::device_ptr<int32_t> col_ptr(sparse_mul_cols.data_ptr<int32_t>());

        thrust::device_ptr<int32_t> indices_0_ptr(output_indices_[0].data_ptr<int32_t>());
        thrust::device_ptr<int32_t> indices_1_ptr(output_indices_[1].data_ptr<int32_t>());
        thrust::device_ptr<int32_t> indices_2_ptr(output_indices_[2].data_ptr<int32_t>());
        thrust::device_ptr<int32_t> indices_3_ptr(output_indices_[3].data_ptr<int32_t>());

        auto begin = thrust::make_zip_iterator(
            thrust::make_tuple(row_ptr, col_ptr, indices_0_ptr, indices_1_ptr, indices_2_ptr, indices_3_ptr)
        );
        auto end = begin + sparse_mul_rows.size(0);

        thrust::for_each(begin, end, [=] __device__ (auto t) {
            int32_t row = static_cast<int32_t>(thrust::get<0>(t));
            int32_t col = static_cast<int32_t>(thrust::get<1>(t));

            int64_t rem_0 = row % (x2 * x3);

            thrust::get<2>(t) = row / (x2 * x3);         // indices_0
            thrust::get<3>(t) = rem_0 / x3;              // indices_1
            thrust::get<4>(t) = rem_0 % x3;              // indices_2
            thrust::get<5>(t) = col % x4;                // indices_3
        });
        std::vector<int32_t> output_shape = {x1, x2, x3, x4};
        return std::make_tuple(output_indices_, output_values, output_shape, _c);
    }

template <typename scalar_t>
at::Tensor matmul_grad_cuda_kernel(
    const at::Tensor& mat1_v,
    const at::Tensor& mat2_v,
    const at::Tensor& _c,
    int32_t& C,
    std::string& mul_type
    ) {

        int32_t N = _c.size(0);
        int _c_max = _c.max().item<int64_t>();

        int xThreads = 1;  // moderate size for the idx loop
        int yThreads = 16;    // lower size for the i loop
        int zThreads = 16;    // lower size for the j loop


        at::Tensor output_values;
        if (mul_type == "rc_cr")
        {
            auto pos_starts_out = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * _c).slice(0, 0, N-1), 0, at::kInt)});
            auto pos_starts_mat1 = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * C).slice(0, 0, N-1), 0, at::kInt)});
            auto pos_starts_mat2 = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * C).slice(0, 0, N-1), 0, at::kInt)});
            auto total_size = torch::sum(_c * _c).item<int64_t>();
            output_values = torch::empty({total_size}, mat1_v.options());
            dim3 threadsPerBlock(xThreads, yThreads, zThreads);
            dim3 numBlocks((N + xThreads - 1) / xThreads,
                   (_c_max + yThreads - 1) / yThreads,
                   (_c_max + zThreads - 1) / zThreads);
            mul_grad_rc_cr<scalar_t><<<numBlocks, threadsPerBlock>>>(
                _c.data_ptr<int32_t>(),
                pos_starts_out.data_ptr<int32_t>(),
                pos_starts_mat1.data_ptr<int32_t>(),
                pos_starts_mat2.data_ptr<int32_t>(),
                output_values.data_ptr<scalar_t>(),
                mat1_v.data_ptr<scalar_t>(),
                mat2_v.data_ptr<scalar_t>(),
                N, C
            );
        }
        else if (mul_type == "rr_rc"){
            auto pos_starts_out = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * C).slice(0, 0, N-1), 0, at::kInt)});
            auto pos_starts_mat1 = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * _c).slice(0, 0, N-1), 0, at::kInt)});
            auto pos_starts_mat2 = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * C).slice(0, 0, N-1), 0, at::kInt)});
            auto total_size = torch::sum(_c * C).item<int64_t>();
            output_values = torch::empty({total_size}, mat1_v.options());
            dim3 threadsPerBlock(xThreads, yThreads, zThreads);
            dim3 numBlocks((N + xThreads - 1) / xThreads,
                   (_c_max + yThreads - 1) / yThreads,
                   (C + zThreads - 1) / zThreads);
            mul_grad_rr_rc<scalar_t><<<numBlocks, threadsPerBlock>>>(
                _c.data_ptr<int32_t>(),
                pos_starts_out.data_ptr<int32_t>(),
                pos_starts_mat1.data_ptr<int32_t>(),
                pos_starts_mat2.data_ptr<int32_t>(),
                output_values.data_ptr<scalar_t>(),
                mat1_v.data_ptr<scalar_t>(),
                mat2_v.data_ptr<scalar_t>(),
                N, C
            );
        }
        else if (mul_type == "cr_rr"){
            auto pos_starts_out = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * C).slice(0, 0, N-1), 0, at::kInt)});
            auto pos_starts_mat1 = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * C).slice(0, 0, N-1), 0, at::kInt)});
            auto pos_starts_mat2 = torch::cat({torch::zeros(1, _c.options()), torch::cumsum((_c * _c).slice(0, 0, N-1), 0, at::kInt)});
            auto total_size = torch::sum(_c * C).item<int64_t>();
            output_values = torch::empty({total_size}, mat1_v.options());
            dim3 threadsPerBlock(xThreads, yThreads, zThreads);
            dim3 numBlocks((N + xThreads - 1) / xThreads,
                   (C + yThreads - 1) / yThreads,
                   (_c_max + zThreads - 1) / zThreads);
            mul_grad_cr_rr<scalar_t><<<numBlocks, threadsPerBlock>>>(
                _c.data_ptr<int32_t>(),
                pos_starts_out.data_ptr<int32_t>(),
                pos_starts_mat1.data_ptr<int32_t>(),
                pos_starts_mat2.data_ptr<int32_t>(),
                output_values.data_ptr<scalar_t>(),
                mat1_v.data_ptr<scalar_t>(),
                mat2_v.data_ptr<scalar_t>(),
                N, C
            );
        }
        else
            TORCH_CHECK(false, "Invalid mul_type: ", mul_type, ". Expected 'rc_cr', 'rr_rc' or 'cr_rr'.");

        return output_values;
    }


std::tuple<at::Tensor, at::Tensor, std::vector<int32_t>, at::Tensor> matmul_cuda_naive(
    const at::Tensor& mat1_v,
    const at::Tensor& mat1_idx,
    const std::vector<int32_t>& mat1_shape,
    const at::Tensor& mat2_v,
    const at::Tensor& mat2_idx,
    const std::vector<int32_t>& mat2_shape,
    std::string mul_type
    ) {
  TORCH_CHECK(mat1_idx.dim() == 2);
  TORCH_CHECK(mat2_idx.dim() == 2);
  TORCH_CHECK(mat1_v.dim() == 1);
  TORCH_CHECK(mat2_v.dim() == 1);
  TORCH_CHECK(
        mul_type == "q_k" || mul_type == "qk_v",
        "Invalid mul_type: ", mul_type,
        ". Expected 'q_k' or 'qk_v'."
    );
  TORCH_CHECK(
      mat1_v.scalar_type() == mat2_v.scalar_type(),
      "mat1 dtype ",
      mat1_v.scalar_type(),
      " does not match mat2 dtype ",
      mat2_v.scalar_type());
  TORCH_CHECK(
      mat1_idx.scalar_type() == mat2_idx.scalar_type(),
      "mat1 dtype ",
      mat1_idx.scalar_type(),
      " does not match mat2 dtype ",
      mat2_idx.scalar_type());
    TORCH_CHECK(
        mat1_idx.scalar_type() == torch::kInt32,
        "mat1 dtype is not int32, but ", mat1_idx.scalar_type()
    );
    TORCH_CHECK(
        mat1_shape.size() == 4 && mat2_shape.size() == 4 &&
        mat1_shape[0] == mat2_shape[0] &&
        mat1_shape[1] == mat2_shape[1] &&
        mat1_shape[3] == mat2_shape[2],
        "Invalid shapes: mat1_shape (size ", mat1_shape.size(), ", shape [", mat1_shape[0], ", ", mat1_shape[1], ", ", mat1_shape[2], ", ", mat1_shape[3], 
        "]) and mat2_shape (size ", mat2_shape.size(), ", shape [", mat2_shape[0], ", ", mat2_shape[1], ", ", mat2_shape[2], ", ", mat2_shape[3], 
        "]). Expected sizes: 4. Expected mat1_shape[0] == mat2_shape[0], mat1_shape[1] == mat2_shape[1], mat1_shape[3] == mat2_shape[2]."
    );




  std::tuple<at::Tensor, at::Tensor, std::vector<int32_t>, at::Tensor> result;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      mat1_v.scalar_type(), "matmul_cuda_naive_kernel", [&] {
        result = matmul_cuda_naive_kernel<scalar_t>(
            mat1_v, mat1_idx, mat1_shape, mat2_v, mat2_idx, mat2_shape, mul_type);
      });
  return result;
}



at::Tensor matmul_grad_cuda_naive(
    const at::Tensor& mat1_v,
    const at::Tensor& mat2_v,
    const at::Tensor& _c,
    int32_t& C,
    std::string& mul_type
  ) {

  TORCH_CHECK(
      mat1_v.scalar_type() == mat1_v.scalar_type(),
      "mat1 dtype ",
      mat1_v.scalar_type(),
      " does not match mat2 dtype ",
      mat1_v.scalar_type());

  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      mat1_v.scalar_type(), "matmul_grad_cuda_kernel", [&] {
        result = matmul_grad_cuda_kernel<scalar_t>(
            mat1_v, mat2_v, _c, C, mul_type);
      });
  return result;
}


at::Tensor qkv_transpose(
    const at::Tensor& mat1_v,
    const at::Tensor& _c,
    int32_t& C,
    std::string& mul_type
  ) {


  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      mat1_v.scalar_type(), "qkv_transpose_kernel", [&] {
        result = qkv_transpose_kernel<scalar_t>(
            mat1_v, _c, C, mul_type);
      });
  return result;
}
