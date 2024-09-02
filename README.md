# Sparse Swin Transformer 3D

A Swin Transformer implementation for 3D sparse volumes, following the original Swin architecture.

### Installation

Follow the [Pointcept installation guide](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#installation).  
Tested with PyTorch 2.1.0 and CUDA 12.0.

To integrate this model, copy the `sparse_swin` folder to `Pointcept/pointcept/models`. An example configuration is
provided in the `config` folder for `scannet`.

### Architecture

#### Sparse Representation

This implementation closely follows the original Swin Transformer, adapted for sparse data. Notably, native PyTorch
`sparse_coo_tensors` are avoided since their indices only work with `int64`.

Key modifications include the `WindowAttention` component, where the `qkv` matrix is derived from `qkv_sparse_cuda`,
which is coalesced and in COO format. The key matrix is already transposed. We avoid indexing with `tensor[indices]` to
prevent automatic conversion to `int64`. Instead, `tensor.index_select(dim, indices)` is used, which does not trigger
this conversion.

#### Sparse Multiplication

We experimented with [spspmm](https://github.com/karShetty/Torch-Sparse-Multiply) (cusparse-based) for attention, but it
was significantly slower than our naive approach. For example, query-key multiplication is optimized by blocking, where
we combine each window to create a dense block and perform naive matrix multiplication. This method is 10x faster than
cusparse and 100x faster than using repeated `torch.mm` on varying block sizes. Transposing the RHS matrix also improves
coalescing.

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x; // grid dimension for c
int i_dim = blockIdx.y * blockDim.y + threadIdx.y; // grid dimension for i
int j_dim = blockIdx.z * blockDim.z + threadIdx.z; // grid dimension for j

if (idx < N) {
    int repeat_count = _c[idx];

    if (i_dim < repeat_count && j_dim < C) {
        int start = idx_starts[idx];
        int start_c = idx * C;
        int pos_start = pos_starts_out[idx];
        int pos_start_mat1 = pos_starts_mat1[idx];
        int pos_start_mat2 = pos_starts_mat2[idx];

        int pos = pos_start + i_dim * C;

        col_indices[pos + j_dim] = j_dim + start_c;
        row_indices[pos + j_dim] = i_dim + start;
        scalar_t sum = 0;
        for (int k = 0; k < repeat_count; ++k) {
            sum += (m1_val[pos_start_mat1 + i_dim * repeat_count + k] * m2_val_t[pos_start_mat2 + j_dim * repeat_count + k]);
        }
        val[pos + j_dim] = sum;
    }
}
```

#### Sparse Softmax

Softmax is implemented with a custom CUDA call, as the native `torch.sparse.softmax` is slower and produces incorrect
gradients.

```python
# Python equivalent
src_max = torch.zeros(N, dtype=vals.dtype, device=vals.device)
src_max.scatter_reduce_(0, idx_row, vals, reduce='amax', include_self=False)
out = (vals - src_max.index_select(0, idx_row)).exp()
out_sum = torch.zeros(N, dtype=out.dtype, device=out.device)
out_sum.scatter_add_(0, idx_row, out)
out_sum = out_sum.index_select(0, idx_row)
out = out / out_sum
```

#### Patch Merging

Patch merging is implemented based on the approach discussed in
this [issue](https://github.com/microsoft/Swin-Transformer/issues/256).

    spconv.SparseConv3d(dim,
                        2 * dim,
                        kernel_size=2,
                        stride=2)

Similarly, Patch expansion utilizes `spconv.SparseInverseConv3d`

## To-Do

The model is comparable to [Swin3D](https://github.com/microsoft/Swin3D/tree/main/Swin3D/models) in terms of speed, but
can be increased further easily.

- [ ] Calculate indices directly in the BasicLayer to reduce computation.
- [ ] Further optimize matrix multiplication.
- [ ] Further optimize softmax.
- [ ] Release a trained model.

--- 

Parts of the code are based on the implementations
from [Swin](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py)
and [Monai](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py)