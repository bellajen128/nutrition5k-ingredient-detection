#!/usr/bin/env python3
"""
IE7615 Ablation Study Figure Generator
No external dependencies beyond matplotlib, numpy, seaborn
Run: python3 generate_ablation.py
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# FIGURE 1: MAIN ABLATION STUDY
# ============================================================
print("Generating Figure 1: Main Ablation Study...")

fig1 = plt.figure(figsize=(14, 8))

stages = [
    'B0\n(t=0.50)',
    'B0\n(t=0.20)',
    'B3\n(t=0.50)',
    'B3\n(t=0.20)',
    'B3+MFB',
    'Final\nTest'
]

f1_scores = [0.777, 0.786, 0.783, 0.793, 0.814, 0.864]
improvements = [0, 0.009, 0.006, 0.010, 0.021, 0.050]

colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']

ax = fig1.add_subplot(111)
bars = ax.bar(range(len(stages)), f1_scores, color=colors, 
              edgecolor='black', linewidth=2.5, alpha=0.85, width=0.6)

# Add value labels
for i, (bar, f1, imp) in enumerate(zip(bars, f1_scores, improvements)):
    height = bar.get_height()
    
    # F1 on top
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
            f'{f1:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=13)
    
    # Improvement inside
    if i > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.005,
                f'+{imp*100:.1f}%',
                ha='center', va='top', fontsize=10, color='white', fontweight='bold')

ax.set_ylabel('F1 Score', fontweight='bold', fontsize=12)
ax.set_title('Ablation Study: Progressive Improvements', fontweight='bold', fontsize=14)
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, fontsize=11)
ax.set_ylim([0.75, 0.90])
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0.777, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axhline(y=0.864, color='green', linestyle='--', linewidth=1.5, alpha=0.5)

plt.tight_layout()
plt.savefig('ablation_study_main.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: ablation_study_main.png")
plt.close()

import matplotlib.pyplot as plt

# ============================================================
# FIGURE 2: SIMPLE VERSION FOR SLIDES
# ============================================================
print("Generating Figure 2: Simple Version (Slides)...")

fig2 = plt.figure(figsize=(12, 7))

# 修正：MFB Weighting 20 epochs 之後缺少逗號，已補上
stages_simple = [
    'B0\nBaseline',
    'Threshold\nOpt',
    'B3\nUpgrade',
    'MFB\nWeighting',
    '40 epochs',  # <-- 這裡缺少逗號，已修正
    'Final Model'
]
f1_simple = [0.777, 0.786, 0.793, 0.814, 0.859, 0.864]

# 修正：f1_simple 有 6 個元素，所以 colors_simple 也要有 6 個元素，已新增一個顏色
colors_simple = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e74c3c'] 

ax2 = fig2.add_subplot(111)
bars2 = ax2.bar(range(len(stages_simple)), f1_simple, 
                color=colors_simple, edgecolor='black', 
                linewidth=2.5, alpha=0.85, width=0.6)

for i, (bar, val) in enumerate(zip(bars2, f1_simple)):
    height = bar.get_height()
    # 顯示 F1 Score 數值
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.003,
            f'{val:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # 顯示相對於前一階段的提升百分比
    if i > 0:
        improvement = f1_simple[i] - f1_simple[i-1]
        # 確保 improvement 是正數才顯示 '+' 符號
        improvement_str = f'+{improvement*100:.1f}%' if improvement >= 0 else f'{improvement*100:.1f}%'
        
        ax2.text(bar.get_x() + bar.get_width()/2., height - 0.005,
                improvement_str,
                ha='center', va='top', fontsize=11, color='white', fontweight='bold')

ax2.set_ylabel('F1 Score', fontweight='bold', fontsize=13)
ax2.set_title('Feature Ablation Study', fontweight='bold', fontsize=15)
ax2.set_xticks(range(len(stages_simple)))
ax2.set_xticklabels(stages_simple, fontsize=12)
ax2.set_ylim([0.75, 0.90])
ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('ablation_study_slides.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: ablation_study_slides.png")
plt.close()

# ============================================================
# FIGURE 3: PRECISION-RECALL-F1 COMPARISON
# ============================================================
print("Generating Figure 3: Precision-Recall-F1 Comparison...")

fig3 = plt.figure(figsize=(12, 7))

stages_short = ['B0 (0.5)', 'B0 (0.2)', 'B3 (0.5)', 'B3 (0.2)', 'B3+MFB', 'Final']
precision = [0.833, 0.781, 0.830, 0.815, 0.803, 0.8694]
recall = [0.756, 0.824, 0.765, 0.790, 0.858, 0.8760]
f1 = [0.777, 0.786, 0.783, 0.793, 0.814, 0.864]

x_pos = np.arange(len(stages_short))

ax3 = fig3.add_subplot(111)
ax3.plot(x_pos, precision, 'o-', linewidth=2.5, markersize=8, 
         label='Precision', color='#e74c3c')
ax3.plot(x_pos, recall, 's-', linewidth=2.5, markersize=8, 
         label='Recall', color='#3498db')
ax3.plot(x_pos, f1, '^-', linewidth=2.5, markersize=8, 
         label='F1 Score', color='#2ecc71')

ax3.set_ylabel('Score', fontweight='bold', fontsize=12)
ax3.set_title('Metric Evolution: Precision, Recall, F1', fontweight='bold', fontsize=14)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(stages_short, fontsize=10, rotation=15)
ax3.set_ylim([0.70, 0.90])
ax3.legend(loc='lower right', fontsize=11)
ax3.grid(True, alpha=0.3)

# Highlight best F1
best_idx = np.argmax(f1)
ax3.scatter(best_idx, f1[best_idx], s=300, color='gold', 
           edgecolor='black', linewidth=2, zorder=5, marker='*')

plt.tight_layout()
plt.savefig('ablation_study_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: ablation_study_metrics.png")
plt.close()

# ============================================================
# FIGURE 4: CONTRIBUTION BREAKDOWN (PIE CHART)
# ============================================================
print("Generating Figure 4: Contribution Analysis...")

fig4 = plt.figure(figsize=(10, 8))

# Calculate contributions
b0_baseline = 0.777
threshold_contrib = 0.786 - 0.777  # 0.009
b3_contrib = 0.793 - 0.786  # 0.007
mfb_contrib = 0.814 - 0.793  # 0.021
full_train_contrib = 0.864 - 0.814  # 0.050

contribution_labels = [
    f'B0 Baseline\n{b0_baseline:.3f}',
    f'Threshold Opt\n+{threshold_contrib:.3f}',
    f'B3 Upgrade\n+{b3_contrib:.3f}',
    f'MFB Weighting\n+{mfb_contrib:.3f}',
    f'Full Training\n+{full_train_contrib:.3f}'
]

contribution_values = [
    0.777,
    threshold_contrib,
    b3_contrib,
    mfb_contrib,
    full_train_contrib
]

colors_pie = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

ax4 = fig4.add_subplot(111)
wedges, texts, autotexts = ax4.pie(contribution_values, labels=contribution_labels, 
                                     colors=colors_pie, autopct='%1.1f%%',
                                     startangle=90, textprops={'fontsize': 11},
                                     explode=(0.05, 0.05, 0.05, 0.05, 0.05))

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

ax4.set_title('Contribution Distribution to Final F1 Score', 
              fontweight='bold', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig('ablation_study_contribution.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: ablation_study_contribution.png")
plt.close()

# ============================================================
# FIGURE 5: DETAILED TABLE
# ============================================================
print("Generating Figure 5: Summary Table...")

fig5 = plt.figure(figsize=(12, 6))
ax5 = fig5.add_subplot(111)
ax5.axis('tight')
ax5.axis('off')

table_data = [
    ['Stage', 'Model', 'Threshold', 'F1 Score', 'Δ F1', 'Δ %'],
    ['1', 'B0', '0.50', '0.777', '-', 'Baseline'],
    ['2', 'B0', '0.20', '0.786', '+0.009', '+1.2%'],
    ['3', 'B3', '0.50', '0.783', '+0.006', '+0.8%'],
    ['4', 'B3', '0.20', '0.793', '+0.010', '+1.3%'],
    ['5', 'B3+MFB', '0.20', '0.814', '+0.021', '+2.7%'],
    ['6', 'B3+MFB\n(40 epochs)', '0.20', '0.864*', '+0.050', '+6.4%*'],
]

table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.08, 0.15, 0.12, 0.12, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(6):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, 7):
    color = '#ecf0f1' if i % 2 == 0 else 'white'
    for j in range(6):
        table[(i, j)].set_facecolor(color)
        if i == 6:  # Highlight last row
            table[(i, j)].set_facecolor('#fff9e6')
            table[(i, j)].set_text_props(weight='bold')

ax5.text(0.5, 0.95, 'Ablation Study Summary', 
         ha='center', fontsize=14, fontweight='bold', transform=ax5.transAxes)
ax5.text(0.5, 0.05, '* Test Set Result (最終模型在測試集的性能)', 
         ha='center', fontsize=10, style='italic', transform=ax5.transAxes)

plt.tight_layout()
plt.savefig('ablation_study_table.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: ablation_study_table.png")
plt.close()

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\n" + "="*70)
print("ABLATION STUDY SUMMARY")
print("="*70)
print(f"{'Stage':<20} {'F1 Score':<15} {'Improvement':<20}")
print("-"*70)

stages_full = [
    'B0 (t=0.50)',
    'B0 (t=0.20)',
    'B3 (t=0.50)',
    'B3 (t=0.20)',
    'B3 + MFB',
    'Final (Test)'
]
f1_all = [0.777, 0.786, 0.783, 0.793, 0.814, 0.864]

for i, (stage, f1) in enumerate(zip(stages_full, f1_all)):
    if i == 0:
        print(f"{stage:<20} {f1:<15.3f} {'Baseline':<20}")
    else:
        imp = f1_all[i] - f1_all[i-1]
        pct = (imp / f1_all[i-1]) * 100
        print(f"{stage:<20} {f1:<15.3f} {f'+{imp:.3f} ({pct:+.1f}%)':<20}")

print("-"*70)
total_imp = f1_all[-1] - f1_all[0]
total_pct = (total_imp / f1_all[0]) * 100
print(f"{'TOTAL':<20} {f1_all[-1]:<15.3f} {f'+{total_imp:.3f} ({total_pct:+.1f}%)':<20}")
print("="*70)

print("\n✅ All figures generated successfully!")
print("\nGenerated files:")
print("  1. ablation_study_main.png - Main ablation study")
print("  2. ablation_study_slides.png - Simple version for slides")
print("  3. ablation_study_metrics.png - Precision/Recall/F1 comparison")
print("  4. ablation_study_contribution.png - Contribution pie chart")
print("  5. ablation_study_table.png - Summary table")
print("\n")
