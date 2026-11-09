# SPDX-License-Identifier: MIT
"""Benchmark the light-MTP draft loop: torch reference vs static vs graphed."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from torch import nn

from flashrwkv2.mtp import DraftLoop


def _make_weights(args, device):
    hidden = args.hidden
    intermediate = args.intermediate
    vocab = args.vocab

    def norm_scale():
        return torch.randn(hidden, device=device) * 0.1 + 1.0

    state = {
        "block.norm_e.weight": norm_scale(),
        "block.norm_e.bias": torch.zeros(hidden, device=device),
        "block.norm_h.weight": norm_scale(),
        "block.norm_h.bias": torch.zeros(hidden, device=device),
        "block.norm_mlp.weight": norm_scale(),
        "block.norm_mlp.bias": torch.zeros(hidden, device=device),
        "block.norm_out.weight": norm_scale(),
        "block.norm_out.bias": torch.zeros(hidden, device=device),
        "block.proj.weight": torch.randn(hidden, 2 * hidden, device=device) * 0.02,
        "block.gate.weight": torch.randn(intermediate, hidden, device=device) * 0.02,
        "block.up.weight": torch.randn(intermediate, hidden, device=device) * 0.02,
        "block.down.weight": torch.randn(hidden, intermediate, device=device) * 0.02,
    }
    state = {key: value.bfloat16() for key, value in state.items()}
    lm_head = (torch.randn(vocab, hidden, device=device) * 0.02).bfloat16()
    return state, lm_head


def _torch_reference(loop, embed, hidden, token, depth):
    """Allocating torch loop with the same weights."""
    h = hidden.view(1, 1, -1)
    drafted = []
    for _ in range(depth):
        emb = embed(token).view(1, 1, -1)
        fused = F.linear(
            torch.cat([loop._norm["norm_e"](emb), loop._norm["norm_h"](h)], -1),
            loop.proj,
        )
        x = h + fused
        mlp_in = loop._norm["norm_mlp"](x)
        act = F.silu(F.linear(mlp_in, loop.gate)) * F.linear(mlp_in, loop.up)
        h = x + F.linear(act, loop.down)
        logits = torch.mm(
            loop._norm["norm_out"](h)[:, -1],
            loop.lm_head.t(),
            out_dtype=torch.float32,
        )
        token = logits.argmax(dim=-1)
        drafted.append(token)
    return torch.stack(drafted, dim=1)


def _time(fn, iterations, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=14336)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    state, lm_head = _make_weights(args, device)
    embed = nn.Embedding(args.vocab, args.hidden, device=device).to(torch.bfloat16)
    loop = DraftLoop.from_state_dict(state, lm_head, depth=args.depth).to(device)
    loop.capture(embed)

    hidden = torch.randn(1, args.hidden, device=device, dtype=torch.bfloat16)
    token = torch.randint(0, args.vocab, (1,), device=device)

    reference = _torch_reference(loop, embed, hidden, token, args.depth).clone()
    eager_ids = loop.forward(hidden, token, embed).clone()
    graph_ids = loop.replay(hidden, token).clone()

    eager_ms = _time(lambda: loop.forward(hidden, token, embed), args.iterations)
    graph_ms = _time(lambda: loop.replay(hidden, token), args.iterations)
    print(
        json.dumps(
            {
                "depth": args.depth,
                "hidden": args.hidden,
                "intermediate": args.intermediate,
                "vocab": args.vocab,
                "eager_matches_graph": torch.equal(eager_ids, graph_ids),
                "eager_matches_reference": torch.equal(eager_ids, reference),
                "reference": reference.tolist(),
                "eager": eager_ids.tolist(),
                "eager_ms": round(eager_ms, 4),
                "graph_ms": round(graph_ms, 4),
                "speedup": round(eager_ms / graph_ms, 3) if graph_ms else None,
                "head_gb": round(args.depth * args.hidden * args.vocab * 2 / 1e9, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
