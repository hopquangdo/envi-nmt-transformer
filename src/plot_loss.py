import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── 1. Load Data ──────────────────────────────────────────────
def load_data(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path, names=['epoch', 'batch', 'loss', 'lr', 'extra'], engine='python', on_bad_lines='skip')
    
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df['loss']  = pd.to_numeric(df['loss'],  errors='coerce')
    
    df = df.dropna(subset=['epoch', 'loss'])
    df['epoch'] = df['epoch'].astype(int)
    
    return df

p1_path = 'D:/02_Workspace/nlp/envi-nmt-transformer/loss/loss_log_phase1.csv'
p2_path = 'D:/02_Workspace/nlp/envi-nmt-transformer/loss/loss_log_phase2.csv'

df1 = load_data(p1_path)
df2 = load_data(p2_path)

# Normalize epochs to start from 1
min_epoch_1 = df1['epoch'].min()
min_epoch_2 = df2['epoch'].min()

df1['epoch'] = df1['epoch'] - min_epoch_1 + 1
df2['epoch'] = df2['epoch'] - min_epoch_2 + 1

stats1 = df1.groupby('epoch')['loss'].mean()
stats2 = df2.groupby('epoch')['loss'].mean()

print("\n── Phase 1 per epoch average loss (Normalized) ──")
print(stats1.head())
print("\n── Phase 2 per epoch average loss (Normalized) ──")
print(stats2.head())

# ── 2. Plot ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# ax1: Phase 1
ax1.plot(stats1.index, stats1.values, marker='o', markersize=6, linewidth=2.5, color='#4C72B0')
# Điểm xuyết text nếu nhiều epoch quá
for e, m in stats1.items():
    if e % 3 == 0 or e <= 5:  
        ax1.annotate(f'{m:.2f}', (e, m), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9)
ax1.set_title(f'Phase 1 Loss (OPUS-100)', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Mean Cross-Entropy Loss', fontsize=12)
ax1.grid(True, alpha=0.3)

# ax2: Phase 2
ax2.plot(stats2.index, stats2.values, marker='s', markersize=6, linewidth=2.5, color='#DD8452')
for e, m in stats2.items():
    if e % 5 == 0 or e <= 5:
        ax2.annotate(f'{m:.2f}', (e, m), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9)
ax2.set_title(f'Phase 2 Loss (PhoMT)', fontsize=14, fontweight='bold', pad=12)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Mean Cross-Entropy Loss', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout(pad=2.0)

# Save & Show
output_path = 'D:/02_Workspace/nlp/envi-nmt-transformer/outputs/loss_plot.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"\n✅ Plot saved to {output_path}")

plt.show()
