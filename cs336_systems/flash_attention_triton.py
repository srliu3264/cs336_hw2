import math

import torch
import triton
import triton.language as tl

from cs336_systems.flash_attention import _flash_backward_comp


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_i = tl.load(Q_block_ptr)
    m_i = tl.full((Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(T_k):
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_idx[:, None] >= k_idx[None, :]
            S_ij = S_ij + tl.where(mask, 0.0, -1.0e6)
        m_new = tl.maximum(m_i, tl.max(S_ij, axis=1))
        alpha = tl.exp(m_i - m_new)
        P_tilde = tl.exp(S_ij - m_new[:, None])
        l_i = alpha * l_i + tl.sum(P_tilde, axis=1)
        P_cast = P_tilde.to(V_block_ptr.type.element_ty)
        o_i = tl.dot(P_cast, V_j, acc=alpha[:, None] * o_i)
        m_i = m_new
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
    o_i = o_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)
    tl.store(O_block_ptr, o_i.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L_i)


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

        B, N_q, D = Q.shape
        N_k = K.shape[-2]
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        O = torch.empty_like(Q)
        L = torch.empty((B, N_q), dtype=torch.float32, device=Q.device)
        scale = 1.0 / math.sqrt(D)
        T_q = triton.cdiv(N_q, Q_TILE_SIZE)
        grid = (T_q, B)
        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            N_q,
            N_k,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = _flash_backward_comp(Q, K, V, O, L, dO, ctx.is_causal)
        return dQ, dK, dV, None
