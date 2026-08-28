import csv
import random
from pathlib import Path

import torch
from transformer_lens import HookedTransformer


SEED = 42
N_SAMPLES = 200
SEQUENCE_LENGTH = 8
TOP_K_HEADS = 20
TOKEN_POOL_SIZE = 200
MEAN_ABLATION_REFERENCE_SAMPLES = 100

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mean_and_std(values):
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, variance ** 0.5


def build_token_pool(model, rng):
    """
    Sample random token ids directly from the vocabulary, rather than
    English words. This avoids the semantic-similarity confound you get
    from natural-language candidates (e.g. " apple" / " orange" having
    elevated baseline attention/logit relationships unrelated to induction).
    """
    vocab_size = model.cfg.d_vocab

    special_ids = {
        token_id
        for token_id in (
            getattr(model.tokenizer, "eos_token_id", None),
            getattr(model.tokenizer, "bos_token_id", None),
            getattr(model.tokenizer, "pad_token_id", None),
        )
        if token_id is not None
    }

    pool = set()

    while len(pool) < TOKEN_POOL_SIZE:
        candidate = rng.randrange(vocab_size)

        if candidate in special_ids:
            continue

        pool.add(candidate)

    return list(pool)


def make_repeated_sequence(token_pool, rng):
    first_half = rng.sample(token_pool, SEQUENCE_LENGTH)
    second_half = first_half.copy()

    return first_half + second_half


def make_shuffled_sequence(token_pool, rng):
    first_half = rng.sample(token_pool, SEQUENCE_LENGTH)
    second_half = first_half.copy()

    while True:
        rng.shuffle(second_half)

        if all(
            second_half[i] != first_half[i]
            for i in range(SEQUENCE_LENGTH)
        ):
            break

    return first_half + second_half


def get_induction_targets(sequence):
    midpoint = len(sequence) // 2
    first_half = sequence[:midpoint]

    successor = {}

    for i in range(midpoint - 1):
        successor[first_half[i]] = first_half[i + 1]

    successor[first_half[-1]] = first_half[0]

    targets = []

    for token in sequence[midpoint:]:
        targets.append(successor[token])

    return targets


def run_with_attention(model, tokens):
    logits, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: name.endswith("attn.hook_pattern"),
    )

    patterns = []

    for layer in range(model.cfg.n_layers):
        patterns.append(
            cache[f"blocks.{layer}.attn.hook_pattern"]
        )

    attention = torch.stack(patterns)

    return logits, attention


def induction_logit_score(logits, targets, start_position):
    scores = []

    for offset, target in enumerate(targets):
        position = start_position + offset

        correct = logits[0, position, target]

        distractors = logits[0, position].clone()

        distractors[target] = -torch.inf

        best_distractor = distractors.max()

        scores.append(
            (correct - best_distractor).item()
        )

    return sum(scores) / len(scores)


def attention_copy_scores(attention, sequence):
    midpoint = len(sequence) // 2

    n_layers = attention.shape[0]
    n_heads = attention.shape[2]

    scores = torch.zeros(n_layers, n_heads, device=attention.device)
    counts = torch.zeros_like(scores)

    for second_position in range(midpoint, len(sequence)):
        token = sequence[second_position]
        first_position = (sequence[:midpoint].index(token) + 1) % midpoint

        for layer in range(n_layers):
            for head in range(n_heads):
                scores[layer, head] += attention[
                    layer,
                    0,          # batch index (batch size is always 1 here)
                    head,
                    second_position,
                    first_position,
                ]
                counts[layer, head] += 1

    return scores / counts


@torch.no_grad()
def compute_mean_activations(model, samples):
    """
    Average attn.hook_z activation per head over a reference distribution,
    used for mean-ablation. Using the shuffled (non-induction) sequences as
    the reference gives an "expected output absent the repeat structure"
    baseline, rather than the off-distribution zero vector.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    totals = torch.zeros(n_layers, n_heads, d_head, device=model.cfg.device)
    count = 0

    for sequence in samples:
        tokens = torch.tensor(
            [sequence],
            dtype=torch.long,
            device=model.cfg.device,
        )

        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name.endswith("attn.hook_z"),
        )

        for layer in range(n_layers):
            z = cache[f"blocks.{layer}.attn.hook_z"]  # (batch, pos, head, d_head)
            totals[layer] += z.sum(dim=(0, 1))

        count += tokens.shape[1]

    return totals / count


def ablate_head(model, tokens, layer, head, mode="zero", mean_activations=None):
    def hook(value, hook):
        value = value.clone()

        if mode == "zero":
            value[:, :, head, :] = 0
        elif mode == "mean":
            value[:, :, head, :] = mean_activations[layer, head]
        else:
            raise ValueError(f"Unknown ablation mode: {mode}")

        return value

    hook_name = f"blocks.{layer}.attn.hook_z"

    return model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, hook)],
    )


@torch.no_grad()
def evaluate_ablation(model, samples, candidate_heads, mean_activations):
    results = []

    for layer, head in candidate_heads:
        normal_scores = []
        zero_scores = []
        mean_scores = []

        for sequence in samples:
            tokens = torch.tensor(
                [sequence],
                dtype=torch.long,
                device=model.cfg.device,
            )

            targets = get_induction_targets(sequence)

            normal_logits = model(tokens)
            normal_scores.append(
                induction_logit_score(normal_logits, targets, SEQUENCE_LENGTH)
            )

            zero_logits = ablate_head(model, tokens, layer, head, mode="zero")
            zero_scores.append(
                induction_logit_score(zero_logits, targets, SEQUENCE_LENGTH)
            )

            mean_logits = ablate_head(
                model,
                tokens,
                layer,
                head,
                mode="mean",
                mean_activations=mean_activations,
            )
            mean_scores.append(
                induction_logit_score(mean_logits, targets, SEQUENCE_LENGTH)
            )

        normal_mean, _ = mean_and_std(normal_scores)
        zero_mean, _ = mean_and_std(zero_scores)
        mean_mean, _ = mean_and_std(mean_scores)

        zero_effects = [n - z for n, z in zip(normal_scores, zero_scores)]
        mean_effects = [n - m for n, m in zip(normal_scores, mean_scores)]

        zero_effect_mean, zero_effect_std = mean_and_std(zero_effects)
        mean_effect_mean, mean_effect_std = mean_and_std(mean_effects)

        results.append(
            {
                "layer": layer,
                "head": head,
                "normal_score": normal_mean,
                "zero_ablated_score": zero_mean,
                "zero_effect": zero_effect_mean,
                "zero_effect_std": zero_effect_std,
                "mean_ablated_score": mean_mean,
                "mean_effect": mean_effect_mean,
                "mean_effect_std": mean_effect_std,
            }
        )

    return results


def save_results(results, filename):
    path = RESULTS_DIR / filename

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    return path


def main():
    device = get_device()

    print(f"Using device: {device}")

    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device=device,
    )

    model.eval()

    rng = random.Random(SEED)

    token_pool = build_token_pool(model, rng)

    repeated_samples = [
        make_repeated_sequence(token_pool, rng)
        for _ in range(N_SAMPLES)
    ]

    shuffled_samples = [
        make_shuffled_sequence(token_pool, rng)
        for _ in range(N_SAMPLES)
    ]

    attention_totals = torch.zeros(
        model.cfg.n_layers,
        model.cfg.n_heads,
        device=device,
    )

    repeated_induction_scores = []
    shuffled_induction_scores = []

    with torch.no_grad():
        for sample_index, sequence in enumerate(repeated_samples):
            tokens = torch.tensor(
                [sequence],
                dtype=torch.long,
                device=device,
            )

            logits, attention = run_with_attention(
                model,
                tokens,
            )

            copy_scores = attention_copy_scores(
                attention,
                sequence,
            )

            attention_totals += copy_scores

            targets = get_induction_targets(sequence)

            score = induction_logit_score(
                logits,
                targets,
                SEQUENCE_LENGTH,
            )

            repeated_induction_scores.append(score)

            if sample_index % 25 == 0:
                print(
                    f"Repeated samples: "
                    f"{sample_index + 1}/{N_SAMPLES}"
                )

        for sequence in shuffled_samples:
            tokens = torch.tensor(
                [sequence],
                dtype=torch.long,
                device=device,
            )

            logits = model(tokens)

            targets = get_induction_targets(sequence)

            score = induction_logit_score(
                logits,
                targets,
                SEQUENCE_LENGTH,
            )

            shuffled_induction_scores.append(score)

    attention_average = attention_totals / N_SAMPLES

    flattened = attention_average.flatten()
    values, indices = torch.topk(
        flattened,
        TOP_K_HEADS,
    )

    candidates = []

    print()
    print("Average attention to token following earlier occurrence:")
    print()

    for rank, (value, index) in enumerate(
        zip(values, indices),
        start=1,
    ):
        layer = index.item() // model.cfg.n_heads
        head = index.item() % model.cfg.n_heads

        candidates.append((layer, head))

        print(
            f"{rank:2}. "
            f"L{layer}H{head}: "
            f"{value.item():.4f}"
        )

    repeated_mean, repeated_std = mean_and_std(repeated_induction_scores)
    shuffled_mean, shuffled_std = mean_and_std(shuffled_induction_scores)

    print()
    print("Induction score:")
    print(
        f"Repeated: {repeated_mean:.4f} (±{repeated_std:.4f})"
    )
    print(
        f"Shuffled: {shuffled_mean:.4f} (±{shuffled_std:.4f})"
    )
    print(
        f"Difference: "
        f"{repeated_mean - shuffled_mean:.4f}"
    )

    attention_results = []

    for index in range(
        model.cfg.n_layers * model.cfg.n_heads
    ):
        layer = index // model.cfg.n_heads
        head = index % model.cfg.n_heads

        attention_results.append(
            {
                "layer": layer,
                "head": head,
                "attention_to_copy": (
                    attention_average[layer, head].item()
                ),
            }
        )

    attention_results.sort(
        key=lambda x: x["attention_to_copy"],
        reverse=True,
    )

    attention_path = save_results(
        attention_results,
        "attention_head_scores.csv",
    )

    print()
    print(f"Saved: {attention_path}")

    print()
    print("Computing mean-ablation reference activations...")

    mean_activations = compute_mean_activations(
        model,
        shuffled_samples[:MEAN_ABLATION_REFERENCE_SAMPLES],
    )

    print(
        f"Running causal ablations (zero and mean) on top "
        f"{TOP_K_HEADS} heads..."
    )

    ablation_results = evaluate_ablation(
        model,
        repeated_samples[:50],
        candidates,
        mean_activations,
    )

    ablation_results.sort(
        key=lambda x: x["zero_effect"],
        reverse=True,
    )

    print()
    print("Ablation results:")
    print()

    for rank, result in enumerate(
        ablation_results,
        start=1,
    ):
        print(
            f"{rank:2}. "
            f"L{result['layer']}H{result['head']}: "
            f"zero_effect={result['zero_effect']:.4f} "
            f"(±{result['zero_effect_std']:.4f}), "
            f"mean_effect={result['mean_effect']:.4f} "
            f"(±{result['mean_effect_std']:.4f})"
        )

    ablation_path = save_results(
        ablation_results,
        "head_ablation_results.csv",
    )

    print()
    print(f"Saved: {ablation_path}")


if __name__ == "__main__":
    main()