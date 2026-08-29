import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Figure 1: attention scores, all 144 heads --------------------------

attn_df = pd.read_csv('results/attention_head_scores.csv')
attn_df = attn_df.sort_values('attention_to_copy', ascending=False).reset_index(drop=True)
attn_labels = 'L' + attn_df['layer'].astype(str) + 'H' + attn_df['head'].astype(str)

N_LABELED = 20  # labeling all 144 is unreadable

plt.figure(figsize=(12, 4))
plt.bar(range(len(attn_df)), attn_df['attention_to_copy'], width=1.0)
plt.xticks(range(N_LABELED), attn_labels[:N_LABELED], rotation=90, fontsize=7)
plt.xlim(-1, len(attn_df))
plt.ylabel('Attention to induction target')
plt.xlabel(f'Rank (top {N_LABELED} labelled of {len(attn_df)} heads)')
plt.tight_layout()
plt.savefig('figures/attention_scores_full.pdf')
plt.close()

# --- Figure 2: zero- vs mean-ablation effect, top-20 candidate heads ----

abl_df = pd.read_csv('results/head_ablation_results.csv')
abl_df = abl_df.sort_values('zero_effect', ascending=False)
abl_labels = 'L' + abl_df['layer'].astype(str) + 'H' + abl_df['head'].astype(str)

x = np.arange(len(abl_df))
w = 0.35

plt.figure(figsize=(10, 4))
plt.bar(x - w / 2, abl_df['zero_effect'], w, yerr=abl_df['zero_effect_std'], label='Zero-ablation')
plt.bar(x + w / 2, abl_df['mean_effect'], w, yerr=abl_df['mean_effect_std'], label='Mean-ablation')
plt.xticks(x, abl_labels, rotation=90, fontsize=7)
plt.axhline(0, color='black', linewidth=0.5)
plt.ylabel('Effect on induction score')
plt.legend()
plt.tight_layout()
plt.savefig('figures/ablation_effects.pdf')
plt.close()

# --- Figure 3: attention score vs. zero-ablation effect -----------------

merged = abl_df.merge(attn_df, on=['layer', 'head'])

plt.figure(figsize=(5, 5))
plt.scatter(merged['attention_to_copy'], merged['zero_effect'])
for _, row in merged.iterrows():
    plt.annotate(f"L{int(row['layer'])}H{int(row['head'])}",
                 (row['attention_to_copy'], row['zero_effect']), fontsize=7)
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('Attention to induction target')
plt.ylabel('Zero-ablation effect')
plt.tight_layout()
plt.savefig('figures/attention_vs_effect.pdf')
plt.close()