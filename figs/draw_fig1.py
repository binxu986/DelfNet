import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
fig.suptitle('L=5 深度神经网络 传统结构 vs 增量参数结构 对比图',
             fontsize=16, fontweight='bold', y=0.95)

# 层位置
layer_positions = np.array([
    [0, 4],   # layer 0
    [2, 4],   # layer 1
    [4, 4],   # layer 2
    [6, 4],   # layer 3
    [8, 4]    # layer 4
])

# 权重位置（圆柱）- 左图用
weight_positions_left = np.array([
    [0, 2.5],  # w0
    [2, 2.5],  # w1
    [4, 2.5],  # w2
    [6, 2.5],  # w3
    [8, 2.5]   # w4
])

# ---------------------- 左图：传统神经网络 ----------------------
ax1.set_title('传统神经网络', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlim(-1, 9)
ax1.set_ylim(0, 6)
ax1.set_aspect('equal')

# 画层（圆圈）
for i, (x, y) in enumerate(layer_positions):
    circle = plt.Circle((x, y), 0.5, color='white', ec='#2E86AB', lw=2.5)
    ax1.add_patch(circle)
    ax1.text(x, y, f'${i}$', ha='center', va='center',
             fontsize=12, fontweight='bold', color='#2E86AB')

# 画层间箭头
for i in range(4):
    x_start, y_start = layer_positions[i]
    x_end, y_end = layer_positions[i+1]
    ax1.arrow(x_start+0.5, y_start, x_end-x_start-1, 0,
              head_width=0.15, head_length=0.15, fc='#A23B72', ec='#A23B72', lw=2)

# 画权重（圆柱）w0~w4
for i, (x, y) in enumerate(weight_positions_left):
    # 圆柱主体
    cylinder = plt.Rectangle((x-0.15, y-0.3), 0.3, 0.6,
                           color='#F18F01', alpha=0.8, ec='#C73E1D')
    ax1.add_patch(cylinder)
    # 圆柱顶部椭圆
    ellipse = mpatches.Ellipse((x, y+0.3), 0.3, 0.15,
                         color='#F18F01', ec='#C73E1D')
    ax1.add_patch(ellipse)
    # 标签
    ax1.text(x, y-0.7, f'$w_{i}$', ha='center', va='center',
             fontsize=11, fontweight='bold', color='#C73E1D')

    # 权重连接线：相同索引的层和权重相连
    ax1.plot([x, layer_positions[i][0]], [y, layer_positions[i][1]-0.5],
             'k--', lw=1, alpha=0.6)

ax1.text(4, 1, '独立权重参数\n$w_0,w_1,w_2,w_3,w_4$',
         ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax1.axis('off')

# ---------------------- 右图：增量参数网络 ----------------------
ax2.set_title('增量参数神经网络', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlim(-1, 9)
ax2.set_ylim(0, 6)
ax2.set_aspect('equal')

# 画层（圆圈）
for i, (x, y) in enumerate(layer_positions):
    circle = plt.Circle((x, y), 0.5, color='white', ec='#2E86AB', lw=2.5)
    ax2.add_patch(circle)
    ax2.text(x, y, f'${i}$', ha='center', va='center',
             fontsize=12, fontweight='bold', color='#2E86AB')

# 画层间箭头
for i in range(4):
    x_start, y_start = layer_positions[i]
    x_end, y_end = layer_positions[i+1]
    ax2.arrow(x_start+0.5, y_start, x_end-x_start-1, 0,
              head_width=0.15, head_length=0.15, fc='#A23B72', ec='#A23B72', lw=2)

# 增量权重位置（w0 + Δ1~Δ4）
delta_positions = np.array([
    [0, 2.5],   # w0
    [2, 2.5],   # Δ1
    [4, 2.5],   # Δ2
    [6, 2.5],   # Δ3
    [8, 2.5]    # Δ4
])

# 颜色：w0用深绿色，Δ1~Δ4用蓝色系
delta_colors = ['#228B22', '#00A896', '#028090', '#05668D', '#0353A0']
delta_labels = ['$W_0$', r'$\Delta_1$', r'$\Delta_2$', r'$\Delta_3$', r'$\Delta_4$']

# 画增量权重（圆柱）
for i, (x, y) in enumerate(delta_positions):
    cylinder = plt.Rectangle((x-0.15, y-0.3), 0.3, 0.6,
                           color=delta_colors[i], alpha=0.8, ec='#003366')
    ax2.add_patch(cylinder)
    ellipse = mpatches.Ellipse((x, y+0.3), 0.3, 0.15,
                         color=delta_colors[i], ec='#003366')
    ax2.add_patch(ellipse)
    ax2.text(x, y-0.7, delta_labels[i], ha='center', va='center',
             fontsize=11, fontweight='bold', color='#003366')

    # 权重连接线：圆柱连接到所有序号>=它的层
    for j in range(i, len(layer_positions)):
        ax2.plot([x, layer_positions[j][0]], [y, layer_positions[j][1]-0.5],
                 'k--', lw=1, alpha=0.6)

# 增量公式
formula_text = r'增量权重参数' + '\n' + \
               r'$W_1=W_0+\Delta_1$' + '\n' + \
               r'$W_2=W_0+\Delta_1+\Delta_2$' + '\n' + \
               r'$W_3=W_0+\Delta_1+\Delta_2+\Delta_3$' + '\n' + \
               r'$W_4=W_0+\Delta_1+\Delta_2+\Delta_3+\Delta_4$'
ax2.text(4, 0.8, formula_text, ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax2.axis('off')

plt.tight_layout()
plt.savefig('neural_network_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("[OK] 传统网络 vs 增量参数网络 对比图已生成完成！")