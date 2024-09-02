from __future__ import annotations

import math
from collections.abc import Sequence
from typing import List

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load

################################################################################

################################################################################


ops = load(name="custom_ops",
           sources=["pointcept/models/sparse_swin/ops.cpp", "pointcept/models/sparse_swin/ops.cu"],
           extra_cuda_cflags=['--extended-lambda', ],
           extra_include_paths=["pointcept/models/sparse_swin"],
           verbose=True)


class SparseTensor:
    def __init__(self, indices, values, shape):
        assert indices.shape[0] == 4, "Invalid shape"
        assert indices.ndim == 2, "Invalid shape"
        assert values.ndim == 1, "Invalid shape"
        assert len(shape) == 4, "Invalid shape"

        self.indices = indices
        self.values = values
        self.shape = shape

    def __repr__(self):
        return f"SparseTensor(indices={self.indices}, values={self.values}, shape={self.shape})"

    def coalesce(self):
        indices_sorted = ops.sort_indices(ops.flatten_indices(self.indices, self.shape))
        return SparseTensor(self.indices[:, indices_sorted], self.values[indices_sorted], self.shape)

    def permute(self, dims):
        if len(dims) != len(self.shape):
            raise ValueError(f"The length of dims {dims} must match the number of dimensions in shape {self.shape}.")

        if sorted(dims) != list(range(len(self.shape))):
            raise ValueError(f"dims {dims} must be a permutation of the dimensions [0, {len(self.shape) - 1}].")

        transposed_shape = [self.shape[dim] for dim in dims]
        transposed_indices = self.indices[dims, :]
        indices_sorted = ops.sort_indices(ops.flatten_indices(transposed_indices, transposed_shape))
        transposed_indices = transposed_indices.index_select(1, indices_sorted)
        transposed_values = self.values.index_select(0, indices_sorted)
        return SparseTensor(transposed_indices, transposed_values, transposed_shape)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return SparseTensor(self.indices, self.values * other, self.shape)
        else:
            raise NotImplementedError


class _spspmm_block_naive(Function):
    """
    Torch autograd Function wrapper for sparse sparse matrix (only CUDA) implementations.
    """

    @staticmethod
    def forward(
            ctx,
            mat1_v, mat1_idx, mat1_shape,
            mat2_v, mat2_idx, mat2_shape,
            mul_type
    ):
        valid_mul_values = {"q_k", "qk_v"}

        if mul_type not in valid_mul_values:
            raise ValueError(f"Invalid value for 'mul': {mul_type}. Must be 'q_k' or 'qk_v'.")

        mat3_idx, mat3_v, mat3_shape, _c = ops.matmul_cuda_naive(mat1_v, mat1_idx, mat1_shape,
                                                                 mat2_v, mat2_idx, mat2_shape,
                                                                 mul_type
                                                                 )

        ctx.save_for_backward(mat2_v, mat1_v, _c)
        ctx.mat1_shape = mat1_shape
        ctx.mat2_shape = mat2_shape
        ctx.mat3_shape = mat3_shape

        ctx.mark_non_differentiable(mat3_idx)

        ctx.mul_type = mul_type
        if mul_type == 'q_k':
            ctx.C = mat1_shape[-1]
        elif mul_type == 'qk_v':
            ctx.C = mat2_shape[-1]
        else:
            raise NotImplementedError
        return mat3_idx, mat3_v, mat3_shape

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_mat3_idx, grad_mat3_v, grad_mat3_shape):
        try:
            import pydevd
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        except:
            pass
        grad_mat3_v = grad_mat3_v.contiguous()
        mat2_v, mat1_v, _c = ctx.saved_tensors
        mul_type = ctx.mul_type

        C = ctx.C
        if mul_type == 'q_k':
            # No transpose for RHS, becasue the multiplication assumes a transposed matrix
            grad_mat1_v = ops.matmul_grad_cuda_naive(grad_mat3_v, mat2_v, _c,
                                                     C, 'rr_rc')
            mat1_t_val = ops.qkv_transpose(mat1_v, _c, C, "rc")
            grad_mat3_t_v = ops.qkv_transpose(grad_mat3_v, _c, C, "rr")
            grad_mat2_v = ops.matmul_grad_cuda_naive(mat1_t_val, grad_mat3_t_v, _c, C,
                                                     'cr_rr')
        elif mul_type == 'qk_v':
            # No transpose for RHS, becasue the multiplication assumes a transposed matrix
            grad_mat1_v = ops.matmul_grad_cuda_naive(grad_mat3_v, mat2_v, _c,
                                                     C, 'rc_cr')
            mat1_t_val = ops.qkv_transpose(mat1_v, _c, C, "rr")
            grad_mat3_t_v = ops.qkv_transpose(grad_mat3_v, _c, C, "rc")
            grad_mat2_v = ops.matmul_grad_cuda_naive(mat1_t_val, grad_mat3_t_v, _c, C,
                                                     'rr_rc')
        else:
            raise NotImplementedError

        return grad_mat1_v, None, None, grad_mat2_v, None, None, None


def spspmm_block_naive(mat1: SparseTensor, mat2: SparseTensor, mul_type='q_k') -> SparseTensor:
    valid_mul_values = {"q_k", "qk_v"}

    if mul_type not in valid_mul_values:
        raise ValueError(f"Invalid value for 'mul': {mul_type}. Must be 'q_k' or 'qk_v'.")

    mat1_v = mat1.values
    mat1_idx = mat1.indices
    mat1_shape = mat1.shape
    mat2_v = mat2.values
    mat2_idx = mat2.indices
    mat2_shape = mat2.shape

    mat3_idx, mat3_v, mat3_shape = _spspmm_block_naive.apply(mat1_v, mat1_idx, mat1_shape,
                                                             mat2_v, mat2_idx, mat2_shape,
                                                             mul_type)

    return SparseTensor(mat3_idx, mat3_v, mat3_shape)


class _softmax_last_dim(Function):
    """
    Torch autograd Function wrapper for sparse sparse matrix (only CUDA) implementations.
    """

    @staticmethod
    def forward(
            ctx,
            mat_v, mat_idx, mat_shape,
    ):

        indices = ops.unique_consecutive_indices(ops.flatten_indices(mat_idx, mat_shape))
        out = ops.softmax_last_dim_forward(indices, mat_v)
        ctx.save_for_backward(out, indices)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_mat):
        try:
            import pydevd
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        except:
            pass

        out, indices = ctx.saved_tensors
        grad_out = ops.softmax_last_dim_backward(indices, out, grad_mat)
        return grad_out, None, None,


def softmax_last_dim(mat: SparseTensor) -> SparseTensor:
    indices_ = mat.indices[:3]
    shape_ = mat.shape[:-1]
    vals_ = mat.values
    dtype = vals_.dtype
    mat_out = _softmax_last_dim.apply(vals_.float(), indices_, shape_)
    return SparseTensor(mat.indices, mat_out.to(dtype), mat.shape)


################################################################################

################################################################################


def replace_feature(out, new_features):
    return out.replace_feature(new_features)


def get_act_layer(name: str):
    """
    Create an activation layer instance based on a string input.

    For example, to create activation layers:

    .. code-block:: python

        relu_layer = get_activation_layer(name="relu")
        sigmoid_layer = get_activation_layer(name="sigmoid")

    Args:
        name: a string representing the activation type.
    """
    if not name:
        return nn.Identity()

    name = name.lower()

    # Map of activation names to their corresponding PyTorch activation layers
    activation_map = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "prelu": nn.PReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "swish": nn.SiLU  # PyTorch uses SiLU, which is equivalent to Swish
    }

    if name not in activation_map:
        raise ValueError(f"Unknown activation type '{name}'. Please use a valid activation type.")

    return activation_map[name]()


def get_norm_layer(name: str):
    """
    Create a normalization layer instance based on a string input.

    For example, to create normalization layers:

    .. code-block:: python

        batchnorm_layer = get_norm_layer(name="batchnorm")
        layernorm_layer = get_norm_layer(name="layernorm")

    Args:
        name: a string representing the normalization type.
    """
    if not name:
        raise ValueError("Normalization type must be provided.")

    name = name.lower()

    # Map of normalization names to their corresponding PyTorch normalization layers
    normalization_map = {
        "batch": nn.BatchNorm1d,
        "layer": nn.LayerNorm,
    }

    if name not in normalization_map:
        raise ValueError(f"Unknown normalization type '{name}'. Please use a valid normalization type.")

    return normalization_map


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Tensor initialization with truncated normal distribution.
    Based on:
    https://github.com/rwightman/pytorch-image-models

    Args:
       tensor: an n-dimensional `torch.Tensor`
       mean: the mean of the normal distribution
       std: the standard deviation of the normal distribution
       a: the minimum cutoff value
       b: the maximum cutoff value
    """

    if std <= 0:
        raise ValueError("the standard deviation should be greater than zero.")

    if a >= b:
        raise ValueError("minimum cutoff value (a) should be smaller than maximum cutoff value (b).")

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def window_partition_sparse(x, window_size):
    """Optimized window partition operation for spconv sparse tensors."""
    assert isinstance(
        x, spconv.SparseConvTensor
    ), "Input must be a spconv.SparseConvTensor"

    d, h, w = x.spatial_shape
    b = x.batch_size
    ws_d, ws_h, ws_w = window_size
    num_windows_d = d // ws_d
    num_windows_h = h // ws_h
    num_windows_w = w // ws_w

    indices = x.indices

    new_indices = torch.stack(
        [
            indices[:, 0] * num_windows_d * num_windows_h * num_windows_w
            + (indices[:, 1] // ws_d) * num_windows_h * num_windows_w
            + (indices[:, 2] // ws_h) * num_windows_w
            + (indices[:, 3] // ws_w),
            indices[:, 1] % ws_d * ws_h * ws_w
            + indices[:, 2] % ws_h * ws_w
            + indices[:, 3] % ws_w,
        ],
        dim=1,
    )

    new_shape = (

        ws_d * ws_h * ws_w,
    )

    x.indices = new_indices
    x.spatial_shape = new_shape
    x.batch_size = b * num_windows_d * num_windows_h * num_windows_w
    return x


def window_reverse_sparse(x, original_shape, window_size):
    """Optimized reverse of the window partition operation for spconv sparse tensors."""
    assert isinstance(
        x, spconv.SparseConvTensor
    ), "Input must be a spconv.SparseConvTensor"

    indices = x.indices
    features = x.features

    b, d, h, w, c = original_shape
    ws_d, ws_h, ws_w = window_size
    num_windows_d = d // ws_d
    num_windows_h = h // ws_h
    num_windows_w = w // ws_w

    # Decompose flat index to original indices
    batch_index = indices[:, 0] // (num_windows_d * num_windows_h * num_windows_w)
    window_flat_index = indices[:, 0] % (num_windows_d * num_windows_h * num_windows_w)
    window_d_index = window_flat_index // (num_windows_h * num_windows_w)
    window_h_index = (window_flat_index % (num_windows_h * num_windows_w)) // num_windows_w
    window_w_index = window_flat_index % num_windows_w

    # Compute original indices within each window
    depth_in_window = indices[:, 1] // (ws_h * ws_w)
    height_in_window = (indices[:, 1] % (ws_h * ws_w)) // ws_w
    width_in_window = indices[:, 1] % ws_w

    # Combine window and in-window indices to form original full indices
    orig_depth_index = window_d_index * ws_d + depth_in_window
    orig_height_index = window_h_index * ws_h + height_in_window
    orig_width_index = window_w_index * ws_w + width_in_window

    original_indices = torch.stack([batch_index, orig_depth_index, orig_height_index, orig_width_index], dim=1)

    x.indices = original_indices
    x.spatial_shape = [d, h, w]
    x.batch_size = b

    return x


@torch.compile
def window_partition(x: torch.Tensor, window_size: List[int]) -> torch.Tensor:
    x_shape = x.size()
    b, d, h, w, c = x_shape
    x = x.view(
        b,
        d // window_size[0],
        window_size[0],
        h // window_size[1],
        window_size[1],
        w // window_size[2],
        window_size[2],
        c,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
    )
    return windows


@torch.compile
def compute_mask(dims: List[int], window_size: List[int], shift_size: List[int], device: torch.device) -> torch.Tensor:
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device, dtype=torch.int8)

    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    # attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    # attn_mask.masked_fill_(attn_mask != 0, -100)
    # return attn_mask
    return mask_windows


@torch.compile
def softmax_last_dim_(sparse_tensor):
    # Equivalent the custom ops, but slower
    # torch.sparse.softmax() gives out wrong gradients and is slower
    indices_ = sparse_tensor.indices[:3]
    idx_row = torch.unique_consecutive(ops.flatten_indices(indices_, sparse_tensor.shape[:-1]), return_inverse=True)[1]
    N = idx_row[-1] + 1

    dtype = sparse_tensor.values.dtype
    if dtype == torch.float16:
        vals = sparse_tensor.values.float()
    else:
        vals = sparse_tensor.values

    src_max = torch.zeros(N, dtype=vals.dtype, device=vals.device)
    src_max.scatter_reduce_(0, idx_row, vals, reduce='amax', include_self=False)
    out = (vals - src_max.index_select(0, idx_row)).exp()
    out_sum = torch.zeros(N, dtype=out.dtype, device=out.device)
    out_sum.scatter_add_(0, idx_row, out)
    out_sum = out_sum.index_select(0, idx_row)
    out = out / out_sum

    if dtype == torch.float16:
        out = out.half()

    return SparseTensor(sparse_tensor.indices, out, sparse_tensor.shape)


@torch.compile
def qkv_sparse_cuda(qkv: torch.Tensor, indices: torch.Tensor, batch_size: int, spatial_shape):
    num_heads, c = qkv.shape[2:]
    spatial_dim = spatial_shape
    indices_qv, sorted_qv, indices_k, sorted_k = ops.qkv_sparse_indices(indices, batch_size, spatial_dim,
                                                                        num_heads, c)
    indices_qv_sorted = indices_qv.index_select(1, sorted_qv)
    indices_k_sorted = indices_k.index_select(1, sorted_k)

    q_tensor = SparseTensor(indices_qv_sorted, qkv[:, 0].reshape(-1).index_select(0, sorted_qv),
                            [batch_size, num_heads, spatial_dim, c])
    k_tensor_transpose = SparseTensor(indices_k_sorted, qkv[:, 1].reshape(-1).index_select(0, sorted_k),
                                      [batch_size, num_heads, c, spatial_dim])
    v_tensor = SparseTensor(indices_qv_sorted, qkv[:, 2].reshape(-1).index_select(0, sorted_qv),
                            [batch_size, num_heads, spatial_dim, c])
    return q_tensor, k_tensor_transpose, v_tensor


def pad_sparse_tensor(x, pad_l, pad_r, pad_t, pad_b, pad_d0=0, pad_d1=0):
    """
    Manually pads a SparseConvTensor.

    Parameters:
    x (SparseConvTensor): The input sparse tensor.
    pad_l, pad_r (int): Padding on the left and right of the width dimension.
    pad_t, pad_b (int): Padding on the top and bottom of the height dimension.
    pad_d0, pad_d1 (int): Padding for depth, used in 3D tensors.

    Returns:
    SparseConvTensor: A new SparseConvTensor with adjusted indices and spatial shape.
    """
    new_indices = x.indices.clone()
    new_indices[:, 1] += pad_d0  # Depth front padding
    new_indices[:, 2] += pad_t  # Height top padding
    new_indices[:, 3] += pad_l  # Width left padding
    # Calculate new spatial shape
    new_spatial_shape = (
        x.spatial_shape[0] + pad_d0 + pad_d1,
        x.spatial_shape[1] + pad_t + pad_b,
        x.spatial_shape[2] + pad_l + pad_r,
    )
    x.indices = new_indices
    x.spatial_shape = new_spatial_shape
    return x


def roll_sparse_tensor(x, shifts, dims):
    """
    Cyclically shifts a SparseConvTensor along specified dimensions.

    Parameters:
    x (SparseConvTensor): The input sparse tensor.
    shifts (tuple): Number of places to shift along each specified dimension.
    dims (tuple): Dimensions along which to apply the shifts.

    Returns:
    SparseConvTensor: A new SparseConvTensor with cyclically shifted indices.
    """
    new_indices = x.indices.clone()

    for shift, dim in zip(shifts, dims):
        # Apply cyclic shift for each specified dimension
        if dim < 1 or dim > 3:
            raise ValueError(
                "Dimension out of bounds for spatial shift. Must be 1, 2, or 3."
            )

        # Calculate the new index considering cyclic conditions
        new_indices[:, dim] = (new_indices[:, dim] + shift) % x.spatial_shape[dim - 1]
    x.indices = new_indices

    return x


@torch.compile
def sp_add_rpb(sparse, dense):
    nonzero_indices = sparse.indices
    nonzero_values = sparse.values
    shape = sparse.shape
    dense_non_zero = dense.view(-1).index_select(0,
                                                 nonzero_indices[1] * dense.shape[1] * dense.shape[2] + nonzero_indices[
                                                     2] * dense.shape[2] + nonzero_indices[3])
    nonzero_values = nonzero_values + dense_non_zero
    return SparseTensor(nonzero_indices, nonzero_values, shape)


@torch.compile
def sp_add_mask(sparse, dense):
    nonzero_indices = sparse.indices
    nonzero_values = sparse.values
    shape = sparse.shape
    idx_0 = nonzero_indices[0] % dense.shape[0]
    window1_values = dense.view(-1).index_select(0, idx_0 * dense.shape[1] + nonzero_indices[2])
    window2_values = dense.view(-1).index_select(0, idx_0 * dense.shape[1] + nonzero_indices[3])
    attn_mask = window2_values - window1_values
    attn_mask.masked_fill_(attn_mask != 0, -100)
    nonzero_values = nonzero_values + attn_mask

    return SparseTensor(nonzero_indices, nonzero_values, shape)


################################################################################

################################################################################


class WindowAttentionSparse(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Sequence[int],
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        mesh_args = torch.meshgrid.__kwdefaults__
        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask):
        n = x.spatial_shape[0]
        b = x.batch_size
        c = x.features.shape[-1]
        qkv = self.qkv(x.features).reshape(-1, 3, self.num_heads, c // self.num_heads)
        q, k_t, v = qkv_sparse_cuda(qkv, x.indices, b, n)
        q = q * self.scale
        attn = spspmm_block_naive(q, k_t, 'q_k')
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().to(attn.values.dtype)
        attn = sp_add_rpb(attn, relative_position_bias)
        if mask is not None:
            attn = sp_add_mask(attn, mask)
            attn = softmax_last_dim(attn)
        else:
            attn = softmax_last_dim(attn)
        x_ = spspmm_block_naive(attn, v, 'qk_v')
        x_ = x_.permute((0, 2, 1, 3))
        arg_sort = ops.return_inverse(ops.flatten_indices(x.indices.T.contiguous(), [b, n]), -1)[0]
        x = replace_feature(x, x_.values.reshape(x.features.shape).index_select(0, arg_sort))
        x = replace_feature(x, self.proj_drop(self.proj(x.features)))
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Sequence[int],
            shift_size: Sequence[int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            act_layer: str = "GELU",
            norm_layer: type[nn.LayerNorm | nn.BatchNorm1d] = nn.LayerNorm,
            use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionSparse(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop)

    def forward_part1(self, x, mask_matrix):
        x = replace_feature(x, self.norm1(x.features))
        b, d, h, w, c = x.batch_size, *x.spatial_shape, x.features.shape[-1]
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = pad_sparse_tensor(x, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1)
        dp, hp, wp, = *x.spatial_shape,
        dims = [b, dp, hp, wp, c]

        if any(i > 0 for i in shift_size):
            shifted_x = roll_sparse_tensor(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition_sparse(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = window_reverse_sparse(attn_windows, dims, window_size)
        if any(i > 0 for i in shift_size):
            x = roll_sparse_tensor(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            new_spatial_shape = [d, h, w]
            x.spatial_shape = new_spatial_shape
        return x

    def forward_part2(self, x):
        x = replace_feature(x, self.norm2(x.features))
        x = replace_feature(x, self.drop_path(self.mlp(x.features)))
        return x

    def forward(self, x, mask_matrix):
        shortcut = x
        x = self.forward_part1(x, mask_matrix)
        x = replace_feature(x, shortcut.features + self.drop_path(x.features))
        x = replace_feature(x, x.features + self.forward_part2(x).features)
        return x


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            depth: int,
            num_heads: int,
            window_size: Sequence[int],
            drop_path: list,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            norm_layer: type[nn.LayerNorm | nn.BatchNorm1d] = nn.LayerNorm,
            downsample: nn.Module | None = None,
            use_checkpoint: bool = False,
            indice_key=None,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size), indice_key=indice_key
            )

    def forward(self, x):
        b, d, h, w, c = x.batch_size, *x.spatial_shape, x.features.shape[-1]
        window_size, shift_size = get_window_size(
            (d, h, w), self.window_size, self.shift_size
        )
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        if self.shift_size == 0:
            attn_mask = None
        else:
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.features.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(spconv.SparseModule):
    """
    Patch embedding block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Unlike ViT patch embedding block: (1) input is padded to satisfy window size requirements (2) normalized if
    specified (3) position embedding is not used.

    Example::
    """

    def __init__(
            self,
            patch_size: Sequence[int] | int = 2,
            in_chans: int = 1,
            embed_dim: int = 48,
            norm_layer: type[nn.LayerNorm | nn.BatchNorm1d] = nn.LayerNorm,
            spatial_dims: int = 3,
            indice_key=None,
    ) -> None:
        """
        Args:
            patch_size: dimension of patch size.
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            norm_layer: normalization layer.
            spatial_dims: spatial dimension.
        """

        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.conv_encoder = spconv.SparseSequential(*[
            spconv.SubMConv3d(
                in_channels=in_chans,
                out_channels=embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
                stride=1,
                indice_key='encoder_conv',
            ),
            SegResBlockSparse(
                spatial_dims=spatial_dims,
                in_channels=embed_dim,
                kernel_size=3,
                norm="batch",
                indice_key='encoder_res'
            )
        ])

        self.proj = spconv.SparseConv3d(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size,
            indice_key=indice_key
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.conv_encoder(x)
        x = self.proj(x)
        if self.norm is not None:
            x = replace_feature(x, self.norm(x.features))
        return x


class PatchMerging(spconv.SparseModule):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def __init__(self, dim: int, norm_layer: type[nn.LayerNorm | nn.BatchNorm1d] = nn.LayerNorm, spatial_dims: int = 3,
                 indice_key=None) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        print(indice_key)
        self.reduction = spconv.SparseConv3d(dim,
                                             2 * dim,
                                             kernel_size=2,
                                             stride=2,
                                             bias=False,
                                             indice_key=indice_key)
        self.norm = norm_layer(dim)

    def forward(self, x):
        x = replace_feature(x, self.norm(x.features))
        x = self.reduction(x)
        return x


class PatchExpand(spconv.SparseModule):
    def __init__(
            self,
            dim: int,
            norm_layer: type[nn.LayerNorm | nn.BatchNorm1d] = nn.LayerNorm,
            spatial_dims=3,
            indice_key=None
    ):
        print(indice_key)
        super().__init__()
        self.expansion = spconv.SparseInverseConv3d(dim,
                                                    dim // 2,
                                                    kernel_size=2,
                                                    bias=False,
                                                    indice_key=indice_key
                                                    )
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        x = self.expansion(x)
        x = replace_feature(x, self.norm(x.features))
        return x


################################################################################

################################################################################

class SegResBlockSparse(spconv.SparseModule):
    """
    Residual network block used SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm: tuple | str,
            kernel_size: tuple | int = 3,
            act: tuple | str = "relu",
            indice_key=None,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """
        super().__init__()

        if isinstance(kernel_size, (tuple, list)):
            padding = tuple(k // 2 for k in kernel_size)
        else:
            padding = kernel_size // 2  # type: ignore

        self.bn1 = get_norm_layer(name=norm, spatial_dims=1, channels=in_channels)
        self.act1 = get_act_layer(act)
        self.conv1 = spconv.SubMConv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
            indice_key=indice_key,
        )

        self.bn2 = get_norm_layer(name=norm, spatial_dims=1, channels=in_channels)
        self.act2 = get_act_layer(act)
        self.conv2 = spconv.SubMConv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
            indice_key=indice_key,
        )

    def forward(self, x):
        identity = x.features

        x = replace_feature(x, self.bn1(x.features))
        x = replace_feature(x, self.act1(x.features))
        x = self.conv1(x)

        x = replace_feature(x, self.bn2(x.features))
        x = replace_feature(x, self.act2(x.features))
        x = self.conv2(x)

        x = replace_feature(x, x.features + identity)
        return x


class DropPath(nn.Module):
    """Stochastic drop paths per sample for residual blocks.
    Based on:
    https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        """
        Args:
            drop_prob: drop path probability.
            scale_by_keep: scaling by non-dropped probability.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

        if not (0 <= drop_prob <= 1):
            raise ValueError("Drop path prob should be between 0 and 1.")

    def drop_path(self, x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class MLP(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
            self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0, act: tuple | str = "GELU",
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
            dropout_rate: fraction of the input units to drop.
            act: activation type and arguments. Defaults to GELU. Also supports "GEGLU" and others.
            dropout_mode: dropout mode, can be "vit" or "swin".
                "vit" mode uses two dropout instances as implemented in
                https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py#L87
                "swin" corresponds to one instance as implemented in
                https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_mlp.py#L23


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim) if act != "GEGLU" else nn.Linear(hidden_size, mlp_dim * 2)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = get_act_layer(act)
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = self.drop1

    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x


################################################################################

################################################################################


@MODELS.register_module("SS-v1m1")
class SwinUNETSparse(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            depths: Sequence[int] = (2, 2, 2, 2),
            num_heads: Sequence[int] = (3, 6, 12, 24),
            feature_size: int = 12,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            dropout_path_rate: float = 0.0,
            normalize: bool = True,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
            patch_norm=False,
            patch_size=2,
            window_size=7,
            norm_layer: type[nn.LayerNorm | nn.BatchNorm1d] = nn.LayerNorm,
            mlp_ratio=4.0,
            qkv_bias=True,
            *args,
            **kwargs,
    ) -> None:

        super().__init__()

        self.patch_size = patch_size
        self.window_size = window_size

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        embed_dim = feature_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            spatial_dims=spatial_dims,
            indice_key='patch_embedding'
        )
        self.pos_drop = spconv.SparseSequential(*[nn.Dropout(p=drop_rate)])
        dpr = [x.item() for x in torch.linspace(0, dropout_path_rate, sum(depths))]
        self.layers_down = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                indice_key=f"layer{i_layer}",
            )
            self.layers_down.append(layer)
        if self.normalize:
            self.norm_down = norm_layer(int(embed_dim * 2 ** (self.num_layers - 1)))

        ##########################################

        self.concat_back_dim = nn.ModuleList()
        self.layers_up = nn.ModuleList()

        for i_layer in range(self.num_layers):
            concat_linear = (
                nn.Linear(
                    2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                )
                if i_layer > 0
                else nn.Identity()
            )
            self.concat_back_dim.append(concat_linear)
            if i_layer == 0:
                layer = PatchExpand(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    norm_layer=norm_layer, indice_key=f"layer{self.num_layers - 2 - i_layer}",
                )
                self.layers_up.append(layer)
            else:
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=self.window_size,
                    drop_path=dpr[
                              sum(depths[: (self.num_layers - 1 - i_layer)]): sum(
                                  depths[: (self.num_layers - 1 - i_layer) + 1]
                              )
                              ],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    downsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    indice_key=f"layer{self.num_layers - 2 - i_layer}",
                )
                self.layers_up.append(layer)
        if self.normalize:
            self.norm_up = norm_layer(int(embed_dim * 2 ** (0)))

        self.final_up = PatchExpand(
            dim=feature_size,
            norm_layer=norm_layer, indice_key='patch_embedding')

        self.decoder1 = SegResBlockSparse(
            spatial_dims=spatial_dims,
            in_channels=feature_size // 2,
            kernel_size=3,
            norm="batch",
            indice_key='decoder1'
        )
        self.out = spconv.SubMConv3d(
            in_channels=feature_size // 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            indice_key='out',
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):

        def _trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
            with torch.no_grad():
                # Cut & paste from PyTorch official master until it's in a few official releases - RW
                # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
                def norm_cdf(x):
                    # Computes standard normal cumulative distribution function
                    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

                # Values are generated by using a truncated uniform distribution and
                # then using the inverse CDF for the normal distribution.
                # Get upper and lower cdf values
                l = norm_cdf((a - mean) / std)
                u = norm_cdf((b - mean) / std)

                # Uniformly fill tensor with values from [l, u], then translate to
                # [2l-1, 2u-1].
                tensor.uniform_(2 * l - 1, 2 * u - 1)

                # Use inverse cdf transform for normal distribution to get truncated
                # standard normal
                tensor.erfinv_()

                # Transform to proper mean, std
                tensor.mul_(std * math.sqrt(2.0))
                tensor.add_(mean)

                # Clamp to ensure it's in the proper range
                tensor.clamp_(min=a, max=b)
                return tensor

        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_swin(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x_downsample = []
        for layer in self.layers_down:
            x_downsample.append(x)
            x = layer(x)
        if self.normalize:
            x = replace_feature(x, self.norm_down(x.features))
        x_downsample.append(x)
        return x_downsample

    def forward_swin_up(self, x, x_downsample):
        for _i, layer in enumerate(self.layers_up):
            if _i == 0:
                x = layer(x)
            else:
                x = replace_feature(x, torch.cat([x.features, x_downsample[-_i - 1].features], dim=1))
                x = replace_feature(x, self.concat_back_dim[_i](x.features))
                x = layer(x)
        if self.normalize:
            x = replace_feature(x, self.norm_up(x.features))
        x = self.final_up(x)
        return x

    def forward(self, x):
        o2b = offset2batch(x['offset'])
        x_in = spconv.SparseConvTensor(x['feat'],
                                       torch.cat([o2b[:, None], x['grid_coord']], -1).to(torch.int32),
                                       [math.ceil(dim / 64) * 64
                                        for dim in torch.max(x['grid_coord'], dim=0).values.tolist()],
                                       o2b.max().item() + 1
                                       )

        hidden_states_out = self.forward_swin(x_in)
        dec0 = self.forward_swin_up(hidden_states_out[-1], hidden_states_out[:-1])

        out = self.decoder1(dec0)
        logits = self.out(out)
        return logits.features
