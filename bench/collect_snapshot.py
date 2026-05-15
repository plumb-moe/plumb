#!/usr/bin/env python3
"""
Collect expert activation snapshot via transformers (bypasses autoattach).

Loads OLMoE-1B-7B on GPU, attaches ProfilingHooks directly, runs prompts,
writes snapshot JSON to --snapshot-dir for use by benchmark_eplb_concurrent.py.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from plumb.hook import ProfilingHooks, ActivationCounter

PROMPTS = [
    "The capital of France is",
    "Explain how transformers work in machine learning:",
    "What are the main causes of climate change?",
    "Describe the transformer architecture in detail:",
    "How does reinforcement learning from human feedback work?",
    "What is mixture-of-experts and why does it improve efficiency?",
    "Explain how expert load imbalance affects MoE inference latency:",
    "Write a detailed explanation of attention mechanisms:",
    "Describe protein synthesis from DNA to functional protein:",
    "Explain quantum computing vs classical computing:",
    "What is backpropagation and why does it work?",
    "How does HTTPS protect data transmission?",
    "Explain gradient descent in one paragraph:",
    "What causes the northern lights?",
    "How do GPUs accelerate deep learning?",
    "Describe supervised vs unsupervised learning:",
    "What is the significance of the Turing test?",
    "Explain large language model pre-training and fine-tuning:",
    "What is the difference between RAM and ROM?",
    "How does the human immune system fight viruses?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--snapshot-dir", default="/tmp/plumb")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    snap_dir = Path(args.snapshot_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Use device_map="auto" so large models (e.g. DeepSeek-V2-Lite) spread
    # across all available GPUs rather than OOMing on a single device.
    print(f"Loading {args.model} on available GPUs ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    first_device = next(model.parameters()).device
    print(f"Model loaded (first param on {first_device}).", flush=True)

    counter = ActivationCounter(window_size=50000)
    hooks = ProfilingHooks(counter)
    n_layers = hooks.attach(model, top_k=args.top_k)
    print(f"Attached hooks to {n_layers} layers.", flush=True)

    prompts_pool = (PROMPTS * (args.n_prompts // len(PROMPTS) + 2))[:args.n_prompts]

    print(f"Running {args.n_prompts} prompts ...", flush=True)
    t0 = time.time()
    for i, prompt in enumerate(prompts_pool):
        inputs = tokenizer(prompt, return_tensors="pt").to(first_device)
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_prompts} prompts done ({elapsed:.1f}s)", flush=True)

    # Drain the counter queue
    time.sleep(2)

    snap = counter.snapshot()
    print(f"Snapshot: {len(snap)} expert slots, pass_count={counter.pass_count}", flush=True)

    # Check imbalance
    try:
        from plumb.analysis.imbalance import compute_imbalance
        imbalance = compute_imbalance(counter)
        imbalance_data = [
            {"layer_id": r.layer_id, "ratio": round(r.imbalance_ratio, 4), "max_expert": r.max_expert_id}
            for r in sorted(imbalance, key=lambda x: x.layer_id)
        ]
        if imbalance_data:
            top = sorted(imbalance_data, key=lambda x: x["ratio"], reverse=True)[:3]
            for r in top:
                print(f"  Layer {r['layer_id']}: imbalance {r['ratio']:.2f}x (max expert {r['max_expert']})", flush=True)
    except Exception as e:
        print(f"  imbalance check skipped: {e}", flush=True)
        imbalance_data = []

    pid = os.getpid()
    payload = {
        "pid": pid,
        "model_name": args.model,
        "n_layers": n_layers,
        "pass_count": counter.pass_count,
        "updated_at": time.time(),
        "started_at": t0,
        "imbalance": imbalance_data,
        "expert_counts": {
            f"{lid}:{eid}": count for (lid, eid), count in snap.items()
        },
        "expert_loads": {
            f"{lid}:{eid}": count for (lid, eid), count in snap.items()
        },
        "gpu_to_numa": {},
    }

    out = snap_dir / f"{pid}_snapshot.json"
    out.write_text(json.dumps(payload))
    print(f"\nSnapshot written to {out}", flush=True)
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
