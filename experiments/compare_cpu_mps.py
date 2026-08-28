"""
compare_cpu_mps.py

Runs identical sequences through the same model loaded on CPU and on MPS,
and reports how much the two backends diverge on the quantities the main
experiment actually cares about: raw logits, attention patterns, the
induction logit score, mean-ablation reference activations, and the causal
ablation effect itself (zero- and mean-ablation).

Usage:
    python compare_cpu_mps.py [path/to/full_induction_experiment.py]
"""

import importlib.util
import random
import sys
from pathlib import Path

import torch

DEFAULT_EXPERIMENT_PATH = Path(
    "/Users/nicke/CircuitLens/experiments/full_induction_experiment.py"
)
N_COMPARE_SAMPLES = 20
N_ABLATION_HEADS = 3
N_MEAN_ACTIVATION_SAMPLES = 20


def load_experiment_module(path):
    spec = importlib.util.spec_from_file_location("induction_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_tensors(cpu_tensor, mps_tensor):
    diff = (cpu_tensor - mps_tensor.to("cpu")).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
    }


def compare_logits(cpu_logits, mps_logits):
    stats = compare_tensors(cpu_logits, mps_logits)
    stats["argmax_mismatches"] = (
        cpu_logits.argmax(-1) != mps_logits.to("cpu").argmax(-1)
    ).sum().item()
    stats["total_positions"] = cpu_logits.shape[0] * cpu_logits.shape[1]
    return stats


def avg(stats, key):
    return sum(s[key] for s in stats) / len(stats)


def worst(stats, key):
    return max(s[key] for s in stats)


def main():
    exp_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EXPERIMENT_PATH
    exp = load_experiment_module(exp_path)

    rng = random.Random(exp.SEED)

    print("Loading CPU model...")
    model_cpu = exp.HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model_cpu.eval()

    print("Loading MPS model...")
    model_mps = exp.HookedTransformer.from_pretrained("gpt2-small", device="mps")
    model_mps.eval()

    token_pool = exp.build_token_pool(model_cpu, rng)

    repeated_samples = [
        exp.make_repeated_sequence(token_pool, rng) for _ in range(N_COMPARE_SAMPLES)
    ]
    shuffled_samples = [
        exp.make_shuffled_sequence(token_pool, rng)
        for _ in range(N_MEAN_ACTIVATION_SAMPLES)
    ]

    logit_stats, attn_stats, induction_pairs = [], [], []
    attention_totals_cpu = torch.zeros(
        model_cpu.cfg.n_layers, model_cpu.cfg.n_heads
    )

    print(f"Comparing forward passes on {N_COMPARE_SAMPLES} sequences...")

    with torch.no_grad():
        for sequence in repeated_samples:
            tokens_cpu = torch.tensor([sequence], dtype=torch.long, device="cpu")
            tokens_mps = torch.tensor([sequence], dtype=torch.long, device="mps")

            logits_cpu, attn_cpu = exp.run_with_attention(model_cpu, tokens_cpu)
            logits_mps, attn_mps = exp.run_with_attention(model_mps, tokens_mps)

            logit_stats.append(compare_logits(logits_cpu, logits_mps))
            attn_stats.append(compare_tensors(attn_cpu, attn_mps))

            attention_totals_cpu += exp.attention_copy_scores(attn_cpu, sequence)

            targets = exp.get_induction_targets(sequence)
            score_cpu = exp.induction_logit_score(logits_cpu, targets, exp.SEQUENCE_LENGTH)
            score_mps = exp.induction_logit_score(
                logits_mps.to("cpu"), targets, exp.SEQUENCE_LENGTH
            )
            induction_pairs.append((score_cpu, score_mps))

    print()
    print(f"Compared {N_COMPARE_SAMPLES} sequences, CPU vs MPS:")
    print()
    print(f"Logits  - mean abs diff: {avg(logit_stats, 'mean_abs_diff'):.6f}, "
          f"worst-case max abs diff: {worst(logit_stats, 'max_abs_diff'):.6f}")
    total_mismatches = sum(s["argmax_mismatches"] for s in logit_stats)
    total_positions = sum(s["total_positions"] for s in logit_stats)
    print(f"Logits  - argmax mismatches: {total_mismatches}/{total_positions}")
    print(f"Attn    - mean abs diff: {avg(attn_stats, 'mean_abs_diff'):.6f}, "
          f"worst-case max abs diff: {worst(attn_stats, 'max_abs_diff'):.6f}")

    induction_diffs = [abs(c - m) for c, m in induction_pairs]
    print(f"Induction score - mean abs diff: {sum(induction_diffs)/len(induction_diffs):.6f}, "
          f"max: {max(induction_diffs):.6f}")

    # --- Mean-activation reference (used for mean-ablation) ---
    print()
    print(f"Comparing mean-activation reference on {N_MEAN_ACTIVATION_SAMPLES} sequences...")

    mean_act_cpu = exp.compute_mean_activations(model_cpu, shuffled_samples)
    mean_act_mps = exp.compute_mean_activations(model_mps, shuffled_samples)
    mean_act_stats = compare_tensors(mean_act_cpu, mean_act_mps)

    print(f"Mean activations - mean abs diff: {mean_act_stats['mean_abs_diff']:.6f}, "
          f"max abs diff: {mean_act_stats['max_abs_diff']:.6f}")

    # --- Causal ablation (zero and mean) on a few candidate heads ---
    flattened = attention_totals_cpu.flatten() / N_COMPARE_SAMPLES
    _, indices = torch.topk(flattened, N_ABLATION_HEADS)
    candidate_heads = [
        (idx.item() // model_cpu.cfg.n_heads, idx.item() % model_cpu.cfg.n_heads)
        for idx in indices
    ]

    print()
    print(f"Comparing ablation effects on candidate heads {candidate_heads}...")

    ablation_diffs = {"zero": [], "mean": []}

    with torch.no_grad():
        for layer, head in candidate_heads:
            for sequence in repeated_samples[:10]:
                tokens_cpu = torch.tensor([sequence], dtype=torch.long, device="cpu")
                tokens_mps = torch.tensor([sequence], dtype=torch.long, device="mps")
                targets = exp.get_induction_targets(sequence)

                for mode, mean_cpu, mean_mps in (
                    ("zero", None, None),
                    ("mean", mean_act_cpu, mean_act_mps),
                ):
                    logits_cpu = exp.ablate_head(
                        model_cpu, tokens_cpu, layer, head,
                        mode=mode, mean_activations=mean_cpu,
                    )
                    logits_mps = exp.ablate_head(
                        model_mps, tokens_mps, layer, head,
                        mode=mode, mean_activations=mean_mps,
                    )

                    score_cpu = exp.induction_logit_score(
                        logits_cpu, targets, exp.SEQUENCE_LENGTH
                    )
                    score_mps = exp.induction_logit_score(
                        logits_mps.to("cpu"), targets, exp.SEQUENCE_LENGTH
                    )

                    ablation_diffs[mode].append(abs(score_cpu - score_mps))

    for mode in ("zero", "mean"):
        diffs = ablation_diffs[mode]
        print(f"{mode.capitalize()}-ablation score - mean abs diff: "
              f"{sum(diffs)/len(diffs):.6f}, max: {max(diffs):.6f}")

    print()
    all_ablation_diffs = ablation_diffs["zero"] + ablation_diffs["mean"]
    if (
        worst(logit_stats, "max_abs_diff") < 1e-3
        and total_mismatches == 0
        and mean_act_stats["max_abs_diff"] < 1e-3
        and max(all_ablation_diffs) < 1e-3
    ):
        print("MPS looks numerically safe for this experiment.")
    else:
        print("MPS shows non-trivial divergence from CPU - recommend running on CPU.")


if __name__ == "__main__":
    main()