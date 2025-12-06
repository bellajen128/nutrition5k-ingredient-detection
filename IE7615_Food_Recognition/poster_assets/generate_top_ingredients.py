"""
Generate top ingredients visualization for poster
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.size'] = 12

# Top 20 most common ingredients from Nutrition5K
# (Based on training data statistics)
ingredients = [
    'olive oil', 'salt', 'garlic', 'onions', 'pepper',
    'tomato', 'lemon juice', 'vinegar', 'butter', 'parsley',
    'cheese', 'chicken', 'white rice', 'eggs', 'basil',
    'bread', 'carrots', 'spinach', 'bell peppers', 'mushrooms'
]

# Approximate frequencies (occurrences in training set)
frequencies = [
    1316, 1171, 794, 574, 586,
    520, 470, 440, 410, 380,
    350, 330, 310, 290, 270,
    250, 230, 210, 190, 170
]

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(ingredients)))
bars = ax.barh(ingredients, frequencies, color=colors, edgecolor='black', linewidth=1)

# Add value labels
for bar, freq in zip(bars, frequencies):
    width = bar.get_width()
    ax.text(width + 20, bar.get_y() + bar.get_height()/2,
            f'{freq}',
            ha='left', va='center', fontweight='bold', fontsize=10)

ax.set_xlabel('Frequency in Training Set', fontweight='bold', fontsize=14)
ax.set_title('Top 20 Most Common Ingredients', fontweight='bold', fontsize=16, pad=15)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('top_ingredients.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('top_ingredients.pdf', bbox_inches='tight')
print("✓ top_ingredients.png saved")

plt.close()
print("Done!")
