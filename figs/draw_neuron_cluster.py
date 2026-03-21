import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.patheffects as path_effects

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(-2, 10)
ax.set_ylim(-1, 9)
ax.set_aspect('equal')
ax.axis('off')

# 标题
ax.text(4, 8.5, 'DelfNet 神经元簇内部网络示意图',
        ha='center', va='center', fontsize=16, fontweight='bold')

# 定义神经元位置 - 形成一个簇
np.random.seed(42)
n_neurons = 8
center = np.array([4, 4])
radius = 2.5

# 在圆形区域内分布神经元
angles = np.linspace(0, 2*np.pi, n_neurons, endpoint=False) + np.random.uniform(-0.2, 0.2, n_neurons)
radii = radius * (0.6 + 0.4 * np.random.random(n_neurons))
neuron_positions = center + np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

# 神经元状态（激活/抑制）- 用不同颜色表示
neuron_states = np.random.choice(['active', 'inhibit', 'neutral'], n_neurons, p=[0.4, 0.3, 0.3])

# 神经元颜色映射
state_colors = {
    'active': '#FF6B6B',      # 红色 - 激活
    'inhibit': '#4ECDC4',     # 青色 - 抑制
    'neutral': '#FFE66D'      # 黄色 - 中性
}

# 画神经元之间的连接（内部网络）
# 生成连接矩阵 - 表示神经元之间的互相影响
connections = []
for i in range(n_neurons):
    for j in range(n_neurons):
        if i != j:
            # 随机决定是否有连接，距离越近概率越大
            dist = np.linalg.norm(neuron_positions[i] - neuron_positions[j])
            prob = max(0.1, 1 - dist/5)
            if np.random.random() < prob * 0.6:
                connections.append((i, j))

# 连接类型：兴奋性或抑制性
connection_types = np.random.choice(['excite', 'inhibit'], len(connections), p=[0.6, 0.4])

# 画连接线
for idx, (i, j) in enumerate(connections):
    start = neuron_positions[i]
    end = neuron_positions[j]

    # 计算箭头起止点（从圆边缘开始）
    vec = end - start
    dist = np.linalg.norm(vec)
    vec_norm = vec / dist

    # 从神经元边缘开始
    start_adj = start + vec_norm * 0.5
    end_adj = end - vec_norm * 0.5

    # 连接类型决定颜色和线型
    if connection_types[idx] == 'excite':
        color = '#FF6B6B'
        linestyle = '-'
        alpha = 0.6
    else:
        color = '#4ECDC4'
        linestyle = '--'
        alpha = 0.6

    # 画箭头
    ax.annotate('', xy=end_adj, xytext=start_adj,
                arrowprops=dict(arrowstyle='->', color=color,
                               lw=1.5, linestyle=linestyle,
                               alpha=alpha,
                               connectionstyle='arc3,rad=0.1'))

# 画神经元（圆形）
for i, (x, y) in enumerate(neuron_positions):
    state = neuron_states[i]
    color = state_colors[state]

    # 神经元主体
    circle = Circle((x, y), 0.45, color=color, ec='#2C3E50', lw=2, alpha=0.9)
    ax.add_patch(circle)

    # 神经元标签
    ax.text(x, y, f'n{i}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#2C3E50')

    # 状态标签
    state_label = {'active': '激活', 'inhibit': '抑制', 'neutral': '中性'}[state]
    ax.text(x, y + 0.7, state_label, ha='center', va='bottom',
            fontsize=8, color=color, fontweight='bold')

# 画簇的边界（虚线椭圆）
cluster_ellipse = mpatches.Ellipse(center, radius*2.5, radius*2.5,
                                    fill=False, ec='#3498DB',
                                    lw=2, linestyle='--', alpha=0.8)
ax.add_patch(cluster_ellipse)
ax.text(center[0], center[1] - radius - 0.8, 'Cluster (神经元簇)在某个时刻t下的状态',
        ha='center', va='top', fontsize=12, color='#3498DB', fontweight='bold')

# 图例
legend_elements = [
    mpatches.Patch(facecolor='#FF6B6B', ec='#2C3E50', label='激活神经元'),
    mpatches.Patch(facecolor='#4ECDC4', ec='#2C3E50', label='抑制神经元'),
    mpatches.Patch(facecolor='#FFE66D', ec='#2C3E50', label='中性神经元'),
    plt.Line2D([0], [0], color='#FF6B6B', lw=2, label='兴奋性连接'),
    plt.Line2D([0], [0], color='#4ECDC4', lw=2, linestyle='--', label='抑制性连接'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
          framealpha=0.9, title='图例', title_fontsize=10)

# 说明文字
explanation = (
    "DelfNet 核心思想:\n"
    "- 神经元簇内形成局部网络\n"
    "- 神经元之间存在信息交换\n"
    "- 激活与抑制形成竞争机制\n"
    "- 实现自组织特性"
)
ax.text(-1.5, 2.5, explanation, ha='left', va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECF0F1', alpha=0.8),
        linespacing=1.5)

# 时间步标注
time_steps = ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't9']
ax.text(8.5, 6.5, '时间演化:', ha='left', va='center', fontsize=11, fontweight='bold')
for i, t in enumerate(time_steps):
    ax.text(8.5, 5.8 - i*0.5, f'{t}: 状态更新', ha='left', va='center', fontsize=9, color='#7F8C8D')

plt.tight_layout()
plt.savefig('neuron_cluster_network.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("[OK] 神经元簇内部网络示意图已生成: neuron_cluster_network.png")