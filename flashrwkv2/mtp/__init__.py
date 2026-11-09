# SPDX-License-Identifier: MIT
"""Static-buffer light-MTP draft loop.

The draft loop is a small gated-MLP block followed by the shared LM head,
repeated once per drafted token.  Every iteration is bandwidth bound: it reads
the block weights plus the full ``C x V`` head, so the loop cost is dominated by
the head projection.

``DraftLoop`` keeps every intermediate in preallocated buffers and never syncs
to the host, so the whole loop can be captured once into a CUDA graph and then
replayed with a single graph launch.  That removes the per-iteration kernel
launch and allocation overhead; it cannot remove the head weight reads, which
set the bandwidth floor of ``depth * C * V * 2`` bytes.

Example::

    loop = DraftLoop.from_state_dict(state, lm_head_weight, depth=2)
    loop.capture(embed)
    ids = loop.replay(hidden, token_ids)   # [1, depth] on device, no sync
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .._internal.linear import adapter_linear

_BLOCK_PREFIX = "block."
_NORM_KEYS = ("norm_e", "norm_h", "norm_mlp", "norm_out")


def _bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.bfloat16).contiguous()


class DraftLoop(nn.Module):
    """Light-MTP draft loop over preallocated buffers.

    Args:
        state: mapping with the sidecar ``block.*`` weight names.
        lm_head_weight: ``[V, C]`` BF16 shared LM head weight.
        depth: number of drafted tokens per call.
        eps: layer-norm epsilon used by the block.
    """

    def __init__(
        self,
        state: dict[str, torch.Tensor],
        lm_head_weight: torch.Tensor,
        *,
        depth: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be a positive int")
        self.depth = int(depth)
        self.eps = float(eps)
        self._norm = nn.ModuleDict()
        for name in _NORM_KEYS:
            key = f"{_BLOCK_PREFIX}{name}."
            if f"{key}weight" not in state:
                raise ValueError(f"missing {key}weight in the draft state")
            size = state[f"{key}weight"].shape[0]
            layer = nn.LayerNorm(size, eps=self.eps).to(dtype=torch.bfloat16)
            with torch.no_grad():
                layer.weight.copy_(_bf16(state[f"{key}weight"]))
                layer.bias.copy_(_bf16(state[f"{key}bias"]))
            self._norm[name] = layer
        for name in ("proj", "gate", "up", "down"):
            if f"{_BLOCK_PREFIX}{name}.weight" not in state:
                raise ValueError(f"missing {name}.weight in the draft state")
        self.proj = nn.Parameter(_bf16(state["block.proj.weight"]), requires_grad=False)
        self.gate = nn.Parameter(_bf16(state["block.gate.weight"]), requires_grad=False)
        self.up = nn.Parameter(_bf16(state["block.up.weight"]), requires_grad=False)
        self.down = nn.Parameter(_bf16(state["block.down.weight"]), requires_grad=False)
        hidden_size = int(state["block.proj.weight"].shape[1]) // 2
        self.hidden_size = hidden_size
        self.intermediate_size = int(state["block.gate.weight"].shape[0])
        if lm_head_weight.shape[1] != hidden_size:
            raise ValueError("lm_head weight must be [vocab, hidden]")
        self.lm_head = nn.Parameter(_bf16(lm_head_weight), requires_grad=False)
        self.vocab_size = int(lm_head_weight.shape[0])
        self._graph: torch.cuda.CUDAGraph | None = None
        self._embed: nn.Module | None = None
        self._buffers_ready = False

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, torch.Tensor],
        lm_head_weight: torch.Tensor,
        *,
        depth: int,
        eps: float = 1e-5,
    ) -> DraftLoop:
        """Build a loop from a sidecar ``block.*`` state dict."""
        return cls(state, lm_head_weight, depth=depth, eps=eps)

    # -- buffers ----------------------------------------------------------- #
    def _make_buffers(self, device: torch.device) -> None:
        hidden = self.hidden_size
        intermediate = self.intermediate_size
        bf16 = torch.bfloat16
        self._hidden = torch.zeros((1, hidden), device=device, dtype=bf16)
        self._token = torch.zeros((1,), device=device, dtype=torch.long)
        self._emb = torch.zeros((1, hidden), device=device, dtype=bf16)
        self._cat = torch.zeros((1, 2 * hidden), device=device, dtype=bf16)
        self._x = torch.zeros((1, hidden), device=device, dtype=bf16)
        self._mlp_in = torch.zeros((1, hidden), device=device, dtype=bf16)
        self._gate = torch.zeros((1, intermediate), device=device, dtype=bf16)
        self._up = torch.zeros((1, intermediate), device=device, dtype=bf16)
        self._hidden_next = torch.zeros((1, hidden), device=device, dtype=bf16)
        self._norm_out = torch.zeros((1, hidden), device=device, dtype=bf16)
        self._logits = torch.zeros(
            (1, self.vocab_size), device=device, dtype=torch.float32
        )
        self._drafts = torch.zeros((1, self.depth), device=device, dtype=torch.long)
        self._buffers_ready = True

    def _step(self) -> None:
        hidden = self._hidden
        token = self._token
        assert self._embed is not None
        for index in range(self.depth):
            self._emb.copy_(self._embed(token))
            self._cat[:, : self.hidden_size].copy_(self._norm["norm_e"](self._emb))
            self._cat[:, self.hidden_size :].copy_(self._norm["norm_h"](hidden))
            fused = adapter_linear(self._cat, self.proj, output_dtype=torch.bfloat16)
            self._x.copy_(hidden + fused)
            self._mlp_in.copy_(self._norm["norm_mlp"](self._x))
            self._gate.copy_(
                adapter_linear(self._mlp_in, self.gate, output_dtype=torch.bfloat16)
            )
            self._up.copy_(
                adapter_linear(self._mlp_in, self.up, output_dtype=torch.bfloat16)
            )
            activated = F.silu(self._gate) * self._up
            down = adapter_linear(activated, self.down, output_dtype=torch.bfloat16)
            self._hidden_next.copy_(self._x + down)
            hidden = self._hidden_next
            self._norm_out.copy_(self._norm["norm_out"](hidden))
            self._logits.copy_(
                adapter_linear(self._norm_out, self.lm_head, output_dtype=torch.float32)
            )
            token = self._logits.argmax(dim=-1)
            self._drafts[:, index].copy_(token)
        self._hidden.copy_(hidden)

    # -- execution ---------------------------------------------------------- #
    @torch.no_grad()
    def forward(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        embed: nn.Module,
    ) -> torch.Tensor:
        """Run one draft pass and return ``[1, depth]`` token ids."""
        self._prepare(embed)
        self._hidden.copy_(hidden.to(torch.bfloat16).view(1, -1))
        self._token.copy_(token_ids.to(torch.long).view(1))
        self._step()
        return self._drafts

    @torch.no_grad()
    def capture(self, embed: nn.Module) -> None:
        """Capture the draft loop into a CUDA graph for later replays."""
        if self._graph is not None:
            return
        self._prepare(embed)
        for _ in range(3):
            self._step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step()
        self._graph = graph

    @torch.no_grad()
    def replay(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Replay the captured graph and return ``[1, depth]`` token ids."""
        if self._graph is None:
            raise RuntimeError("capture() must run before replay()")
        self._hidden.copy_(hidden.to(torch.bfloat16).view(1, -1))
        self._token.copy_(token_ids.to(torch.long).view(1))
        self._graph.replay()
        return self._drafts

    def _prepare(self, embed: nn.Module) -> None:
        device = self.lm_head.device
        if not self._buffers_ready:
            self._make_buffers(device)
        if self._embed is None or self._embed is not embed:
            self._embed = embed.to(device)
