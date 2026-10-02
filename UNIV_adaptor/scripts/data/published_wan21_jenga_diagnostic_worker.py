"""Diagnostic-only Jenga interventions; never used by the published pilot.

The original worker/entrypoint/source checkouts stay unchanged. Counters do not
synchronize CUDA or replace attention kernels. Only explicitly labelled controls
disable the cache branch or replace the token permutation with identity.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from UNIV_adaptor.scripts.data import published_wan21_worker as worker
from UNIV_adaptor.scripts.data.published_wan21_jenga_diagnostic import load_plan


def install_diagnostic_hooks(module, settings, torch_module=None):
    import sys
    if torch_module is None:
        import torch as torch_module
    if settings not in (
        {"cache": "upstream_zero", "order": "gilbert"},
        {"cache": "hard_off", "order": "gilbert"},
        {"cache": "hard_off", "order": "identity"},
    ):
        raise ValueError("Unsupported diagnostic intervention")
    original_forward, original_generate = module.teacache_forward, module.generate
    attention = sys.modules["wan.modules.model_mul"]
    dense, sparse = attention.flash_attention, attention.block_sparse_attention
    counters = {}
    identity_order = None

    def counted_dense(*args, **kwargs):
        counters["dense_attention_calls"] += 1
        return dense(*args, **kwargs)

    def counted_sparse(*args, **kwargs):
        counters["sparse_attention_calls"] += 1
        return sparse(*args, **kwargs)

    def forward(model, *args, **kwargs):
        nonlocal identity_order
        counters["forward_calls"] += 1
        enabled = model.enable_teacache
        if settings["cache"] == "hard_off":
            model.enable_teacache = False
            model.use_cache = False
            counters["hard_off_calls"] += 1
        elif enabled:
            counters["cache_enabled_calls"] += 1
        old_order = old_inverse = None
        if settings["order"] == "identity":
            old_order, old_inverse = model.hilbert_order, model.linear_to_hilbert
            if identity_order is None:
                identity_order = torch_module.arange(old_order.numel(), dtype=old_order.dtype, device=old_order.device)
            if identity_order.numel() != old_order.numel():
                raise RuntimeError("Token geometry changed in a fixed-shape diagnostic")
            model.hilbert_order = model.linear_to_hilbert = identity_order
            counters["identity_order_calls"] += 1
        try:
            result = original_forward(model, *args, **kwargs)
            if model.use_cache:
                counters["cache_hits"] += 1
            return result
        finally:
            model.enable_teacache = enabled
            if old_order is not None:
                model.hilbert_order, model.linear_to_hilbert = old_order, old_inverse

    def generate(options):
        counters.clear()
        counters.update({key: 0 for key in (
            "forward_calls", "hard_off_calls", "cache_enabled_calls", "cache_hits",
            "identity_order_calls", "dense_attention_calls", "sparse_attention_calls",
        )})
        return original_generate(options)

    module.teacache_forward, module.generate = forward, generate
    attention.flash_attention, attention.block_sparse_attention = counted_dense, counted_sparse
    return counters


def run(args):
    plan = load_plan(args.out)
    jobs = [j for j in plan["jobs"] if j["gpu"] == args.gpu and j["arm"]["id"] == args.arm]
    if not jobs:
        raise ValueError("No matching diagnostic jobs")
    settings = jobs[0]["diagnostic"]
    if any(j["diagnostic"] != settings for j in jobs):
        raise ValueError("Mixed interventions in one worker")
    original_load = worker.load_entrypoint
    original_immutable = worker.immutable
    counters = None

    def load_entrypoint(source):
        nonlocal counters
        if source != "jenga":
            raise ValueError("Diagnostic worker is Jenga-only")
        module = original_load(source)
        counters = install_diagnostic_hooks(module, settings)
        print(f"Jenga diagnostic-only intervention: {settings}; official pilot unchanged", flush=True)
        return module

    def save_receipt(path, row):
        result = row | {"jenga_diagnostics": {"settings": settings, "counters": dict(counters),
            "scope": "diagnostic control only; integer hooks add CPU overhead, no per-call CUDA timing"}}
        original_immutable(path, result)
        print(f"Diagnostic counts: {result['jenga_diagnostics']['counters']}", flush=True)

    # The existing worker orchestrates weights, warmup, noise capture and receipts.
    # Its original file hash and behaviour are not changed for the frozen v3 pilot.
    worker.load_plan = load_plan
    worker.load_entrypoint = load_entrypoint
    worker.immutable = save_receipt
    args.calibration = True
    worker.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--probe", action="store_true")
    run(parser.parse_args())
