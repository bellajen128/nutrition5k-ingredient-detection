"""
Generate correct Feature Ablation Study figure
"""

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 22

fig, ax = plt.subplots(figsize=(12, 7))

# Correct data (without co-occurrence)
features = ['Base\n(B0)', 'B3', '+MFB', '+Threshold\nOpt (0.20)']
f1_values = [0.777, 0.783, 0.786, 0.793]
improvements = [0, 0.006, 0.003, 0.007]

colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
bars = ax.bar(features, f1_values, color=colors, edgecolor='black', linewidth=2.5, alpha=0.85, width=0.6)

# Add value labels on top
for i, (bar, val, imp) in enumerate(zip(bars, f1_values, improvements)):
    height = bar.get_height()
    
    # F1 score on top
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
            f'{val:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=18)
    
    # Improvement percentage inside bar
    if i > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.004,
                f'+{imp*100:.1f}%',
                ha='center', va='top', fontsize=15, color='white', fontweight='bold')

ax.set_ylabel('F1 Score', fontweight='bold')
ax.set_title('Feature Ablation Study', fontweight='bold', pad=20)
ax.set_ylim([0.75, 0.82])
ax.grid(True, axis='y', alpha=0.4, linestyle='--')

# Add horizontal line at final performance
ax.axhline(y=0.793, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Final Performance')
ax.legend(loc='lower right', fontsize=14, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig('ablation_study_correct.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('ablation_study_correct.pdf', bbox_inches='tight')
print("✓ ablation_study_correct.png saved")
plt.close()

print("\nDone!")
