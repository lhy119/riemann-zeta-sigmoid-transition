import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# 0. 直接嵌入你的计算结果
# ============================================================
N_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
          25, 30, 35, 40, 50, 60, 80, 100, 150, 200, 300, 500]
lnN = np.log(N_list)

var_list = np.array([
    0.1356, 0.1680, 0.1701, 0.1787, 0.1778, 0.1784, 0.1778,
    0.1748, 0.1796, 0.1785, 0.1798, 0.1791, 0.1782, 0.1794,
    0.1804, 0.1809, 0.1802, 0.1809, 0.1807, 0.1801, 0.1805,
    0.1816, 0.1797, 0.1799, 0.1799, 0.1805
])

# 最新 AICc 结果（以 Power-law 为基准）
aic_pow = -338.92   # Power-law AICc
aic_sig = -330.21   # Logistic AICc
aic_log = -286.35   # Logarithmic decay AICc
r2_sig = 0.9733
r2_pow = 0.9787
r2_log = 0.8226

# ============================================================
# 1. 模型函数（用于绘制拟合曲线）
# ============================================================
def sigmoid(x, v0, v1, k, x0):
    return v0 + (v1 - v0) / (1 + np.exp(-k * (x - x0)))

def power_law(x, a, b, c):
    return a + b * np.power(x, c)

def log_decay(x, a, b):
    return a + b / x

# 拟合得到曲线参数（使用你已算出的数据，确保曲线一致）
p0_sig = [var_list[0], var_list[-1], 1.0, np.median(lnN)]
popt_sig, _ = curve_fit(sigmoid, lnN, var_list, p0=p0_sig, maxfev=10000)

p0_pow = [0.1, 0.5, -1.0]
popt_pow, _ = curve_fit(power_law, lnN, var_list, p0=p0_pow, maxfev=10000)

# ============================================================
# 2. 绘制图2a —— CUE 方差弛豫缺失
# ============================================================
plt.figure(figsize=(7, 5))

# 细密的 x 用于画平滑曲线
x_smooth = np.linspace(lnN[0], lnN[-1], 200)

# ------ 灰色半透明矩形：有效饱和窗口 (N=2 ~ N=10) ------
rect_start = np.log(2)
rect_end   = np.log(10)
plt.axvspan(rect_start, rect_end, color='gray', alpha=0.12, zorder=0)
mid_rect = (rect_start + rect_end) / 2
plt.text(mid_rect, 0.194, 'Effective\nSaturation\nWindow',
         ha='center', va='center', fontsize=9, color='gray',
         fontstyle='italic')

# ------ 数据散点 ------
plt.scatter(lnN, var_list, c='black', s=28, zorder=5, label='CUE data')

# ------ 拟合曲线 ------
plt.plot(x_smooth, sigmoid(x_smooth, *popt_sig), 'r--', lw=1.5,
         label='Sigmoid (AICC={:.0f})'.format(aic_sig))
plt.plot(x_smooth, power_law(x_smooth, *popt_pow), 'b-.', lw=1.5,
         label='Power-law (AICC={:.0f})'.format(aic_pow))
# 对数衰减 AIC 太高，不画出曲线，但信息框里会列出

# ------ GUE 极限水平虚线 ------
plt.axhline(y=0.178, color='gray', linestyle=':', lw=1)
plt.text(lnN[-1]+0.1, 0.178, 'GUE limit', fontsize=8, color='gray', va='center')

# ------ 箭头：指向 N>10 的统计涨落区域 ------
arrow_x = 3.6
arrow_y = 0.185
plt.annotate('Statistical\nfluctuations\nonly',
             xy=(5.5, 0.180),           # 箭头指向 (ln250≈5.52)
             xytext=(arrow_x, arrow_y), # 文字位置
             arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
             fontsize=9, color='darkred', fontweight='bold',
             ha='center', va='center')

# ------ 信息框：模型比较 ------
text_str = (r'$\bf{Model\ Comparison}$' +
            '\nSigmoid: AICC = {:.1f}'.format(aic_sig) +
            '\nPower-law: AICC = {:.1f}'.format(aic_pow) +
            '\nLogarithmic: AICC = {:.1f}'.format(aic_log) +
            '\n\nSigmoid offers no\nsignificant advantage')
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9)
plt.text(0.97, 0.55, text_str, transform=plt.gca().transAxes,
         fontsize=8, verticalalignment='top', horizontalalignment='right',
         bbox=props)

# ------ 坐标轴与标题 ------
plt.xlabel(r'$\ln N$', fontsize=12)
plt.ylabel('Variance of normalized spacings', fontsize=12)
plt.title('CUE: Absence of a macroscopic relaxation window', fontsize=13, fontweight='bold')
plt.legend(fontsize=8, loc='upper left')
plt.grid(True, alpha=0.25)
plt.xlim(lnN[0]-0.2, lnN[-1]+0.5)
plt.ylim(0.12, 0.21)

plt.tight_layout()
plt.savefig('Figure2a_CUE_relaxation_absence.pdf', format='pdf', bbox_inches='tight')
plt.show()

print("图2a已保存为 Figure2a_CUE_relaxation_absence.pdf")
