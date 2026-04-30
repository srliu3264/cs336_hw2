from __future__ import annotations

import math

import torch

B_Q = 16
B_K = 16


def _flash_backward(Q, K, V, O, L, dO, is_causal: bool):
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if is_causal:
        n_q, n_k = Q.shape[-2], K.shape[-2]
        q_idx = torch.arrange(n_q, device=Q.device)
        k_inx = torch.arrange(n_k, device=Q.device)
        mask = q_idx[:, None] >= k_idx[None, :]
        S = S + torch.where(mask, 0.0, -1.0e6)
    P = torch.exp(S - L.unsqueeze(-1))
    D = (O * dO).sum(dim=-1)
    dV = torch.matmul(P.transpose(-2, -1), dO)
    dP = torch.matmul(dO, V.transpose(-2, -1))
    dS = P * (dP - D.unsqueeze(-1)) * scale
    dQ = torch.matmul(dS, K)
    dK = torch.matmul(dS.transpose(-2, -1), Q)
    return dQ, dK, dV


_flash_backward_comp = torch.compile(_flash_backward)


class FlashAttention2PyTorch(torch.autograd.Function):
    """FA2 (algorithm1)

    Inputs:
        Q: (..., n_q, d)
        K: (..., n_k, d)
        V: (..., n_k, d)
        is_causal: igonored rn

    Returns:
        O: (..., n_q, d)
    """

    @staticmethod
    def forward(
        ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False
    ) -> torch.Tensor:
        # Q: (..., n_q, d); K, V: (..., n_k, d)
        # return: O, logsumexp L

        n_q = Q.shape[-2]
        n_k = K.shape[-2]
        T_q = n_q // B_Q
        T_k = n_k // B_K
        d = Q.shape[-1]

        scale = 1.0 / math.sqrt(d)  # used for S_ij

        # my understanding of FA2: O=softmax(QK^t)V: rows of O match rows of Q, softmax is on line
        O = torch.zeros_like(Q)
        L = torch.empty(Q.shape[:-1], dtype=Q.dtype, device=Q.device)  # (..., n_q)

        for i in range(T_q):
            Q_i = Q[..., i * B_Q : (i + 1) * B_Q, :]  # (..., B_q, d)

            m_i = torch.full(
                Q_i.shape[:-1], float("-inf"), dtype=Q.dtype, device=Q.device
            )
            l_i = torch.zeros(Q_i.shape[:-1], dtype=Q.dtype, device=Q.device)
            O_i = torch.zeros_like(Q_i)

            for j in range(T_k):
                K_j = K[..., j * B_K : (j + 1) * B_K, :]  # (..., B_k, d)
                V_j = V[..., j * B_K : (j + 1) * B_K, :]  # (..., B_k, d)

                S_ij = (
                    torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale
                )  # (..., B_q, B_k)

                # m_new = m_i^{(j)} in pdf, m_i is m_i^{(j-1) the old one}
                m_new = torch.maximum(m_i, S_ij.amax(dim=-1))  # (..., B_q)
                P_tilde = torch.exp(S_ij - m_new.unsqueeze(-1))  # (..., B_q, B_k)
                alpha = torch.exp(m_i - m_new)  # (..., B_q)
                l_i = alpha * l_i + P_tilde.sum(dim=-1)
                O_i = alpha.unsqueeze(-1) * O_i + torch.matmul(
                    P_tilde, V_j
                )  # ( ..,B_q,d)
                m_i = m_new

            # Final normalization for this query tile, and write outputs.
            O[..., i * B_Q : (i + 1) * B_Q, :] = O_i / l_i.unsqueeze(-1)
            L[..., i * B_Q : (i + 1) * B_Q] = m_i + torch.log(l_i)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        # only return O, L is saved in ctx.
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = _flash_backward_comp(Q, K, V, O, L, dO, ctx.is_causal)
        return dQ, dK, dV, None
