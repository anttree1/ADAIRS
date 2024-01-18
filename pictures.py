import matplotlib.pyplot as plt
import numpy as np

def hot_pic(input_matrix):
    data_iter = iter(input_matrix).next()
    matrix_batch = data_iter.view(16, 90, 90)  # 调整为二维数组
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()  # 将axes展平成一维数组
    for i in range(16):
        heatmap = axes[i].imshow(matrix_batch[i], cmap='Blues', interpolation='nearest')
        axes[i].set_title(f"Image {i + 1}")  # 设置标题
    fig.colorbar(heatmap, ax=axes)  # 添加颜色条

    plt.tight_layout()  # 调整子图的布局，避免重叠
    plt.show()

def hot_pic_nobatch(input_matrix):
    fig, ax = plt.subplots()
    heatmap = ax.imshow(input_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.show()