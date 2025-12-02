"""
Generate figures for poster presentation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-poster')
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18

# ============================================================
# Figure 1: Threshold Optimization
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
f1_scores = [0.753, 0.778, 0.793, 0.785, 0.780, 0.780, 0.769, 0.749]
precisions = [0.698, 0.752, 0.778, 0.800, 0.811, 0.826, 0.830, 0.834]
recalls = [0.865, 0.846, 0.839, 0.803, 0.784, 0.770, 0.749, 0.715]

ax.plot(thresholds, f1_scores, 'o-', linewidth=3, markersize=10, label='F1 Score', color='purple')
ax.plot(thresholds, precisions, 's-', linewidth=2, markersize=8, label='Precision', color='green', alpha=0.7)
ax.plot(thresholds, recalls, '^-', linewidth=2, markersize=8, label='Recall', color='orange', alpha=0.7)

# Mark optimal
ax.axvline(0.20, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Optimal (0.20)')
ax.axhline(0.793, color='red', linestyle='--', linewidth=1, alpha=0.3)

ax.set_xlabel('Threshold', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Threshold Optimization Results', fontweight='bold', pad=20)
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.65, 0.90])

plt.tight_layout()
plt.savefig('threshold_optimization.png', dpi=300, bbox_inches='tight')
plt.savefig('threshold_optimization.pdf', bbox_inches='tight')
print("✓ threshold_optimization.png")

# ============================================================
# Figure 2: Feature Ablation Study
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

features = ['Base\n(B0)', 'B3', '+MFB', '+Threshold\nOpt', '+Co-occur']
f1_values = [0.777, 0.783, 0.786, 0.793, 0.796]
improvements = [0, 0.006, 0.003, 0.007, 0.003]

colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
bars = ax.bar(features, f1_values, color=colors, edgecolor='black', linewidth=2, alpha=0.8)

# Add value labels
for i, (bar, val, imp) in enumerate(zip(bars, f1_values, improvements)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{val:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    if i > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.02,
                f'+{imp:.1%}',
                ha='center', va='top', fontsize=11, color='white', fontweight='bold')

ax.set_ylabel('F1 Score', fontweight='bold')
ax.set_title('Feature Ablation Study', fontweight='bold', pad=20)
ax.set_ylim([0.75, 0.82])
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('ablation_study.png', dpi=300, bbox_inches='tight')
plt.savefig('ablation_study.pdf', bbox_inches='tight')
print("✓ ablation_study.png")

# ============================================================
# Figure 3: Performance Metrics Table (as image)
# ============================================================

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')

metrics_data = [
    ['Metric', 'Value'],
    ['F1 Score', '0.793'],
    ['Precision', '0.778'],
    ['Recall', '0.839'],
    ['Exact Match', '0.364'],
    ['Hamming Accuracy', '0.988'],
]

table = ax.table(cellText=metrics_data, cellLoc='center', loc='center',
                colWidths=[0.6, 0.4])

table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(1, 3)

# Style header
for i in range(2):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, 6):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        table[(i, j)].set_edgecolor('black')
        table[(i, j)].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('metrics_table.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ metrics_table.png")

# ============================================================
# Figure 4: Model Architecture Diagram (Simple)
# ============================================================

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Boxes
boxes = [
    (5, 11, 'RGB Image\n512×512', '#3498db'),
    (5, 9.5, 'EfficientNet-B3\nBackbone\n(ImageNet pretrained)', '#2ecc71'),
    (5, 7.5, '1536 Features', '#95a5a6'),
    (5, 6, 'Classifier\nDropout + MLP', '#f39c12'),
    (5, 4, 'BCEWithLogitsLoss\n+ MFB Weights', '#e74c3c'),
    (5, 2, '249-dim Predictions', '#9b59b6'),
]

for x, y, text, color in boxes:
    bbox = dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.text(x, y, text, ha='center', va='center', fontsize=14, fontweight='bold',
            color='white', bbox=bbox)
    
    if y > 2:
        ax.annotate('', xy=(x, y-0.8), xytext=(x, y-0.3),
                   arrowprops=dict(arrowstyle='->', lw=3, color='black'))

ax.text(5, 0.5, '16.3M Parameters | F1: 0.793', ha='center', fontsize=12, style='italic')

plt.tight_layout()
plt.savefig('architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ architecture.png")

print("\n" + "="*50)
print("All figures generated!")
print("="*50)
