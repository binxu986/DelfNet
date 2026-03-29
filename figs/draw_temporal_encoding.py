import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布 - 横向展示多个时间步（增加高度以容纳中间说明区域）
fig = plt.figure(figsize=(18, 11))

# 主标题
fig.suptitle('DelfNeuron Cluster 时间编码示意图\n神经元激活顺序形成对输入数据的时间编码',
             fontsize=14, fontweight='bold', y=0.96)

# 定义神经元位置 - 固定位置用于所有时间步
n_neurons = 6
center = np.array([0, 0])
radius = 1.2

# 在圆形区域内分布神经元（固定位置）
np.random.seed(42)
angles = np.linspace(0, 2*np.pi, n_neurons, endpoint=False)
radii = radius * (0.7 + 0.3 * np.random.random(n_neurons))
base_positions = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

# 神经元颜色映射
state_colors = {
    'active': '#FF6B6B',      # 红色 - 激活
    'inhibit': '#4ECDC4',     # 青色 - 抑制
    'neutral': '#AAAAAA'      # 灰色 - 未激活/中性
}

# 定义4个时间步的神经元激活模式（模拟时间编码）
# 每个时间步，不同的神经元被激活，形成特定的激活序列
time_steps = ['t0', 't1', 't2', 't3', 't4', 't5']

# 神经元激活模式 - 设计一个有意义的激活序列
# 展示神经元依次被激活的过程，体现时间编码
activation_patterns = [
    ['neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral'],  # t0: 初始状态
    ['active', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral'],   # t1: n0激活
    ['active', 'active', 'neutral', 'inhibit', 'neutral', 'neutral'],    # t2: n0,n1激活, n3抑制
    ['active', 'active', 'active', 'inhibit', 'neutral', 'inhibit'],     # t3: n0,n1,n2激活, n3,n5抑制
    ['active', 'active', 'active', 'inhibit', 'active', 'inhibit'],      # t4: n0,n1,n2,n4激活
    ['neutral', 'active', 'neutral', 'neutral', 'active', 'neutral'],    # t5: 最终稳定状态
]

# 创建6个子图区域（横向排列）- 上半部分
n_time_steps = 6  # 展示全部6个时间步
axes = []
for i in range(n_time_steps):
    ax = fig.add_axes([0.04 + i * 0.16, 0.50, 0.14, 0.38])  # 6个子图横向排列
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    axes.append(ax)

# 画时间步之间的箭头连接
for i in range(n_time_steps - 1):
    # 画从当前时间步到下一个时间步的箭头
    arrow_x_start = 0.04 + i * 0.16 + 0.14 - 0.01
    arrow_x_end = 0.04 + (i + 1) * 0.16 + 0.01

    # 用annotate画箭头（在figure坐标系）
    ax_arrow = fig.add_axes([arrow_x_start, 0.68, 0.02, 0.06])
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)
    ax_arrow.axis('off')
    ax_arrow.annotate('', xy=(1, 0.5), xytext=(0, 0.5),
                      arrowprops=dict(arrowstyle='->', color='#3498DB', lw=2))
    ax_arrow.set_zorder(0)

# 在每个时间步画神经元簇
for t_idx, ax in enumerate(axes):
    # 时间步标签
    time_label = time_steps[t_idx]
    ax.text(0, 2.2, f'时刻 {time_label}', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#2C3E50')

    # 当前时间步的激活模式
    current_pattern = activation_patterns[t_idx]

    # 画簇的边界（虚线椭圆）
    cluster_ellipse = mpatches.Ellipse((0, 0), radius*2.3, radius*2.3,
                                        fill=True, facecolor='#ECF0F1',
                                        ec='#3498DB', lw=1.5, linestyle='--', alpha=0.5)
    ax.add_patch(cluster_ellipse)

    # 画神经元
    for n_idx, (x, y) in enumerate(base_positions):
        state = current_pattern[n_idx]
        color = state_colors[state]

        # 神经元主体
        circle = Circle((x, y), 0.35, color=color, ec='#2C3E50', lw=1.5, alpha=0.9)
        ax.add_patch(circle)

        # 神经元标签
        ax.text(x, y, f'n{n_idx}', ha='center', va='center',
                fontsize=9, fontweight='bold', color='#2C3E50')

        # 如果是激活状态，添加一个发光效果（更大的浅色圆）
        if state == 'active':
            glow = Circle((x, y), 0.45, color='#FF6B6B', alpha=0.3)
            ax.add_patch(glow)

# ===================== 中间空白区域：说明文字 =====================
ax_middle = fig.add_axes([0.04, 0.43, 0.92, 0.05])
ax_middle.set_xlim(0, 1)
ax_middle.set_ylim(0, 1)
ax_middle.axis('off')

# 状态矩阵说明
ax_middle.text(0.5, 0.7, '状态值: 1=激活(红), -1=抑制(青), 0=中性(灰)',
                ha='center', va='center', fontsize=12, color='#34495E')

# 说明文字
explanation = "核心思想: 神经元激活顺序形成时间编码，不同输入触发不同的激活序列"
ax_middle.text(0.5, 0.25, explanation, ha='center', va='center',
                fontsize=12, color='#E74C3C', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ECF0F1', alpha=0.8))

# ===================== 底部：时间编码序列状态矩阵 =====================
# 画时间编码序列示意（底部）- 每一时刻神经元状态列表
ax_encoding = fig.add_axes([0.04, 0.05, 0.92, 0.32])
ax_encoding.set_xlim(0, 12)
ax_encoding.set_ylim(0, 3.5)
ax_encoding.axis('off')

ax_encoding.text(0.3, 3.2, '时间编码序列 - 每时刻神经元状态变化',
                 ha='left', va='top', fontsize=11, fontweight='bold', color='#2C3E50')

# 时间步标签和神经元状态列表
n_time_steps_show = 6
time_labels = ['t0', 't1', 't2', 't3', 't4', 't5']
neuron_labels = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5']

# 每个时间步的位置
time_positions = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5]

# 状态符号映射
state_symbols = {
    'active': ('1', '#FF6B6B'),    # 激活：红色，值为1
    'inhibit': ('-1', '#4ECDC4'),  # 抑制：青色，值为-1
    'neutral': ('0', '#AAAAAA')    # 中性：灰色，值为0
}

# 画时间步标题
for t_idx, (t_label, t_pos) in enumerate(zip(time_labels, time_positions)):
    # 时间步标签
    ax_encoding.text(t_pos, 2.6, t_label, ha='center', va='center',
                    fontsize=11, fontweight='bold', color='#3498DB')

    # 当前时间步的激活模式
    current_pattern = activation_patterns[t_idx]

    # 画神经元状态列表（垂直排列）
    for n_idx, state in enumerate(current_pattern):
        y_pos = 2.2 - n_idx * 0.35  # 垂直排列

        # 状态颜色和符号
        symbol, color = state_symbols[state]

        # 画小方块表示状态
        box_width = 0.5
        box_height = 0.25
        rect = mpatches.Rectangle((t_pos - box_width/2, y_pos - box_height/2),
                                   box_width, box_height,
                                   facecolor=color, alpha=0.8,
                                   edgecolor='#2C3E50', lw=1)
        ax_encoding.add_patch(rect)

        # 状态值
        ax_encoding.text(t_pos, y_pos, symbol, ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')

        # 神经元标签（只在第一列显示）
        if t_idx == 0:
            ax_encoding.text(t_pos - 1, y_pos, neuron_labels[n_idx],
                            ha='right', va='center', fontsize=9, color='#7F8C8D')

# 画时间演进箭头
for i in range(n_time_steps_show - 1):
    ax_encoding.annotate('', xy=(time_positions[i+1] - 0.7, 1.3),
                         xytext=(time_positions[i] + 0.7, 1.3),
                         arrowprops=dict(arrowstyle='->', color='#3498DB', lw=1.5))

# 图例（右上角）
legend_elements = [
    mpatches.Patch(facecolor='#FF6B6B', ec='#2C3E50', label='激活神经元'),
    mpatches.Patch(facecolor='#4ECDC4', ec='#2C3E50', label='抑制神经元'),
    mpatches.Patch(facecolor='#AAAAAA', ec='#2C3E50', label='未激活'),
]

# 把图例放在figure上
fig.legend(handles=legend_elements, loc='upper right', fontsize=9,
          framealpha=0.9, title='神经元状态', title_fontsize=10,
          bbox_to_anchor=(0.95, 0.88))

plt.savefig('temporal_encoding.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("[OK] 时间编码示意图已生成: temporal_encoding.png")