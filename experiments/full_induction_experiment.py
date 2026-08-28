import csv
import random
from pathlib import Path

import torch
from transformer_lens import HookedTransformer


SEED = 42
N_SAMPLES = 200
SEQUENCE_LENGTH = 8
TOP_K_HEADS = 20

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_token_pool(model):
    candidates = [
        " apple", " banana", " cherry", " dog", " eagle",
        " forest", " garden", " house", " island", " jacket",
        " kitten", " lemon", " mountain", " notebook", " orange",
        " piano", " rabbit", " river", " summer", " table",
        " umbrella", " violin", " window", " yellow", " zebra",
        " alpha", " beta", " gamma", " delta", " epsilon",
        " football", " guitar", " hammer", " island", " jungle",
        " kitchen", " library", " morning", " ocean", " planet",
        " queen", " rocket", " school", " train", " valley",
    ]

    token_pool = []

    for text in candidates:
        tokens = model.to_tokens(
            text,
            prepend_bos=False,
        )[0]

        if len(tokens) == 1:
            token_pool.append(tokens[0].item())

    token_pool = list(dict.fromkeys(token_pool))

    if len(token_pool) < SEQUENCE_LENGTH * 2:
        raise RuntimeError(
            f"Only found {len(token_pool)} usable single-token strings."
        )

    return token_pool


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


def ablate_head(model, tokens, layer, head):
    def hook(value, hook):
        value = value.clone()
        value[:, :, head, :] = 0
        return value

    hook_name = f"blocks.{layer}.attn.hook_z"

    return model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, hook)],
    )


def evaluate_ablation(
    model,
    samples,
    candidate_heads,
):
    results = []

    for layer, head in candidate_heads:
        normal_scores = []
        ablated_scores = []

        for sequence in samples:
            tokens = torch.tensor(
                [sequence],
                dtype=torch.long,
                device=model.cfg.device,
            )

            targets = get_induction_targets(sequence)

            normal_logits = model(tokens)

            normal_score = induction_logit_score(
                normal_logits,
                targets,
                SEQUENCE_LENGTH,
            )

            ablated_logits = ablate_head(
                model,
                tokens,
                layer,
                head,
            )

            ablated_score = induction_logit_score(
                ablated_logits,
                targets,
                SEQUENCE_LENGTH,
            )

            normal_scores.append(normal_score)
            ablated_scores.append(ablated_score)

        normal = sum(normal_scores) / len(normal_scores)
        ablated = sum(ablated_scores) / len(ablated_scores)

        results.append(
            {
                "layer": layer,
                "head": head,
                "normal_score": normal,
                "ablated_score": ablated,
                "effect": normal - ablated,
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
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()

    print(f"Using device: {device}")

    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device=device,
    )

    model.eval()

    token_pool = build_token_pool(model)

    rng = random.Random(SEED)

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
    print("Average attention to earlier corresponding token:")
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

    repeated_mean = sum(
        repeated_induction_scores
    ) / len(repeated_induction_scores)

    shuffled_mean = sum(
        shuffled_induction_scores
    ) / len(shuffled_induction_scores)

    print()
    print("Induction score:")
    print(
        f"Repeated: {repeated_mean:.4f}"
    )
    print(
        f"Shuffled: {shuffled_mean:.4f}"
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
    print(
        f"Running causal ablations on top "
        f"{TOP_K_HEADS} heads..."
    )

    ablation_results = evaluate_ablation(
        model,
        repeated_samples[:50],
        candidates,
    )

    ablation_results.sort(
        key=lambda x: x["effect"],
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
            f"effect={result['effect']:.4f} "
            f"("
            f"{result['normal_score']:.4f} -> "
            f"{result['ablated_score']:.4f}"
            f")"
        )

    ablation_path = save_results(
        ablation_results,
        "head_ablation_results.csv",
    )

    print()
    print(f"Saved: {ablation_path}")


if __name__ == "__main__":
    main()