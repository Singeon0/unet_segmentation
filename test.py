import matplotlib.pyplot as plt
import numpy as np

# Model names and their respective metrics
models = ["MULTIRES_ATT_MODEL", "ATT_UNET_MODEL", "MULTIRES_MODEL", "UNET"]
f1_scores = [91.68, 90.11, 89.55, 88.23]
iou_scores = [84.06, 82.22, 81.30, 79.68]

x = np.arange(len(models))  # Label locations
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Bar plots
rects1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score')
rects2 = ax.bar(x + width/2, iou_scores, width, label='IOU')

# Adding text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Comparison: F1 Score and IOU')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Set y-axis limits to zoom in around 85%
ax.set_ylim([75, 100])

# Annotate bars with their values
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()
