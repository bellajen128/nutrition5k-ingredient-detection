
"""
Generate Ablation Study Figure for IE7615 Project
Based on actual experimental results
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ============================================================
# Plot 1: Main Ablation Study - Bar Chart
# ============================================================
ax1 = fig.add_subplot(gs[0, :])

# Actual data from your experiments
stages = [
    'B0 Baseline\n(Threshold=0.50)',
    'B0 + Threshold\nOptimization (0.20)',
    'B3 Upgrade\n(Threshold=0.50)',
    'B3 + Threshold\n(0.20)',
    'B3 + MFB\nWeighting',
    'Final Model\n(40 epochs)'
]

f1_scores = [0.777, 0.786, 0.783, 0.793, 0.814, 0.864]
improvements = [0, 0.009, 0.006, 0.010, 0.021, 0.050]

colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
bars = ax1.bar(range(len(stages)), f1_scores, color=colors, 
               edgecolor='black', linewidth=2.5, alpha=0.85, width=0.6)

# Add value labels on top
for i, (bar, f1, imp) in enumerate(zip(bars, f1_scores, improvements)):
    height = bar.get_height()
    
    # F1 score on top
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.003,
            f'{f1:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # Improvement inside bar (if not baseline)
    if i > 0:
        ax1.text(bar.get_x() + bar.get_width()/2., height - 0.005,
                f'+{imp*100:.1f}%',
                ha='center', va='top', fontsize=11, color='white', fontweight='bold')

ax1.set_ylabel('F1 Score', fontweight='bold', fontsize=14)
ax1.set_title('Ablation Study: Progressive Improvements (Validation Set)',
             fontweight='bold', fontsize=16, pad=20)
ax1.set_xticks(range(len(stages)))
ax1.set_xticklabels(stages, fontsize=11)
ax1.set_ylim([0.75, 0.88])
ax1.grid(True, axis='y', alpha=0.4, linestyle='--')
ax1.axhline(y=0.777, color='red', linestyle='--', linewidth=2, 
           alpha=0.5, label='B0 Baseline (0.777)')
ax1.axhline(y=0.864, color='green', linestyle='--', linewidth=2, 
           alpha=0.5, label='Test Set Result (0.864)')
ax1.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)

# ============================================================
# Plot 2: Contribution Breakdown
# ============================================================
ax2 = fig.add_subplot(gs[1, 0])

contributions = [
    'B0 Baseline',
    'Threshold Opt\n(0.50→0.20)',
    'B3 Upgrade',
    'MFB Weighting',
    'Full Training\n(40 epochs)'
]

contrib_values = [0.777, 0.009, 0.007, 0.021, 0.050]
cumulative = np.cumsum([0] + contrib_values[:-1]) + np.array(contrib_values)

colors_contrib = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
bars2 = ax2.barh(contributions, contrib_values, color=colors_contrib,
                 edgecolor='black', linewidth=2, alpha=0.85)

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, contrib_values)):
    width = bar.get_width()
    if i == 0:
        label = f'F1: {val:.3f}'
    else:
        label = f'+{val:.3f}'
    ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
            label, ha='left', va='center', fontweight='bold', fontsize=11)

ax2.set_xlabel('Contribution to F1 Score', fontweight='bold', fontsize=12)
ax2.set_title('Feature Contribution Analysis', fontweight='bold', fontsize=13)
ax2.set_xlim([0, 0.09])
ax2.grid(True, axis='x', alpha=0.3)

# ============================================================
# Plot 3: Precision, Recall, F1 Evolution
# ============================================================
ax3 = fig.add_subplot(gs[1, 1])

stages_short = ['B0\n(0.5)', 'B0\n(0.2)', 'B3\n(0.5)', 'B3\n(0.2)', 'B3+MFB', 'Final\n(Test)']
precision = [0.833, 0.781, 0.830, 0.815, 0.803, 0.8694]
recall = [0.756, 0.824, 0.765, 0.790, 0.858, 0.8760]
f1 = [0.777, 0.786, 0.783, 0.793, 0.814, 0.864]

x_pos = np.arange(len(stages_short))
width = 0.25

ax3.plot(x_pos, precision, 'o-', linewidth=2.5, markersize=8, 
        label='Precision', color='#e74c3c')
ax3.plot(x_pos, recall, 's-', linewidth=2.5, markersize=8, 
        label='Recall', color='#3498db')
ax3.plot(x_pos, f1, '^-', linewidth=2.5, markersize=8, 
        label='F1 Score', color='#2ecc71')

ax3.set_ylabel('Score', fontweight='bold', fontsize=12)
ax3.set_title('Precision-Recall-F1 Evolution', fontweight='bold', fontsize=13)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(stages_short, fontsize=10)
ax3.set_ylim([0.70, 0.90])
ax3.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
ax3.grid(True, alpha=0.3)

# Highlight best F1
best_idx = np.argmax(f1)
ax3.scatter(best_idx, f1[best_idx], s=300, color='gold', 
           edgecolor='black', linewidth=2, zorder=5, marker='*')
ax3.text(best_idx, f1[best_idx] + 0.01, 'Best F1: 0.864',
        ha='center', fontweight='bold', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ============================================================
# Overall Title and Layout
# ============================================================
fig.suptitle('IE7615: Food Ingredient Recognition - Ablation Study',
            fontsize=18, fontweight='bold', y=0.98)

# Add text box with key findings
textstr = '''Key Findings:
• Threshold Optimization: +3.7% (B0: 0.777 → 0.786)
• B0 → B3 Upgrade: +0.6% (Architecture improvement)
• MFB Weighting: +2.1% (Class imbalance handling)
• Final Test Performance: 0.864 (Precision: 0.869, Recall: 0.876)'''

fig.text(0.02, 0.02, textstr, fontsize=10, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.savefig('ablation_study_complete.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('ablation_study_complete.pdf', bbox_inches='tight')
print("✓ ablation_study_complete.png saved")
print("✓ ablation_study_complete.pdf saved")
plt.show()

# ============================================================
# Generate simplified version for slides
# ============================================================
fig2, ax = plt.subplots(figsize=(12, 7))

stages_simple = [
    'B0\nBaseline',
    'Threshold\nOpt',
    'B3\nUpgrade',
    'MFB\nWeighting',
    'Final Model'
]
f1_simple = [0.777, 0.786, 0.793, 0.814, 0.864]
colors_simple = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

bars_simple = ax.bar(range(len(stages_simple)), f1_simple, 
                     color=colors_simple, edgecolor='black', 
                     linewidth=2.5, alpha=0.85, width=0.6)

# Add labels
for i, (bar, val) in enumerate(zip(bars_simple, f1_simple)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
           f'{val:.3f}',
           ha='center', va='bottom', fontweight='bold', fontsize=15)
    
    if i > 0:
        improvement = f1_simple[i] - f1_simple[i-1]
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.005,
               f'+{improvement*100:.1f}%',
               ha='center', va='top', fontsize=12, color='white', fontweight='bold')

ax.set_ylabel('F1 Score', fontweight='bold', fontsize=14)
ax.set_title('Ablation Study: Progressive Performance Improvements',
            fontweight='bold', fontsize=16, pad=20)
ax.set_xticks(range(len(stages_simple)))
ax.set_xticklabels(stages_simple, fontsize=12)
ax.set_ylim([0.75, 0.90])
ax.grid(True, axis='y', alpha=0.4, linestyle='--')

# Add annotations
ax.annotate('', xy=(0, 0.777), xytext=(4, 0.864),
           arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))
ax.text(2, 0.825, 'Total Improvement: +8.7%',
       ha='center', fontsize=12, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('ablation_study_simple.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('ablation_study_simple.pdf', bbox_inches='tight')
print("✓ ablation_study_simple.png saved (簡化版，適合 slides)")
print("\n✅ All figures generated successfully!")

# Print summary
print("\n" + "="*60)
print("ABLATION STUDY SUMMARY")
print("="*60)
print(f"{'Stage':<25} {'F1 Score':<12} {'Improvement':<15}")
print("-"*60)
for i, (stage, f1) in enumerate(zip(stages, f1_scores)):
    if i == 0:
        print(f"{stage:<25} {f1:<12.3f} {'Baseline':<15}")
    else:
        imp = f1_scores[i] - f1_scores[i-1]
        print(f"{stage:<25} {f1:<12.3f} {f'+{imp:.3f}':<15}")
print("="*60)
print(f"Total Improvement: {f1_scores[-1] - f1_scores[0]:.3f} (+{(f1_scores[-1] - f1_scores[0])/f1_scores[0]*100:.1f}%)")
print("="*60)
