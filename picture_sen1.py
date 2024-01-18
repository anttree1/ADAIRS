import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
classification_accuracy = np.array([65.71, 66.26, 68.35, 69.78, 68.57, 69.89, 70.00, 69.89, 69.77, 70.00])
classification_accuracy_std = np.array([0.31, 0.55, 0.31, 0.60, 1.13, 2.23, 1.52, 1.86, 2.65, 1.28])

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('The upper bound of "a"', fontsize=14)
ax1.set_ylabel('Classification Accuracy (%)', color=color, fontsize=14)
ax1.errorbar(x, classification_accuracy, yerr=classification_accuracy_std, color=color, marker='o', markersize=8,
             linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(65, 73)
ax1.set_xticks(x)
lines = ax1.get_lines()
ax1.axhline(y=65.6, color='r', linestyle='--')
ax1.text(0.5, 65.7, 'baseline', color='r',fontsize=12)


ax1.legend(lines, ['Classification Accuracy'], loc='upper left',fontsize=14)
fig.tight_layout()
plt.show()
plt.savefig('plot.png')
