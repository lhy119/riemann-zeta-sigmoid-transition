import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ============================================================
# 1. 读取JSON格式的零点数据
# ============================================================
filename = '1-2800-2800.1621-r0-0-0.zeros.txt'  # 改成您的实际文件名

with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()

# 跳过注释行，找到JSON数据
# 文件第1行是空行，第2行是注释，第3行是JSON
for line in content.split('\n'):
    line = line.strip()
    if line and not line.startswith('#'):
        # 这就是JSON数据
        data = json.loads(line)
        break

positive_zeros = np.array([float(z) for z in data['positive_zeros']])
positive_zeros.sort()

print(f"读取到 {len(positive_zeros)} 个正虚部零点")
print(f"范围: {positive_zeros[0]:.2f} ~ {positive_zeros[-1]:.2f}")

# ============================================================
# 2. 定义间距展开与统计量计算函数
# ============================================================
def compute_stats(gammas, segment_ratio=50):
    n = len(gammas)
    if n < 50:
        return np.nan, np.nan
    spacings = np.diff(gammas)
    seg_size = max(20, n // segment_ratio)
    n_seg = len(spacings) // seg_size
    if n_seg < 2:
        mean_sp = np.mean(spacings)
        if mean_sp == 0:
            return np.nan, np.nan
        norm_sp = spacings / mean_sp
    else:
        norm_list = []
        for i in range(n_seg):
            seg = spacings[i*seg_size:(i+1)*seg_size]
            m = np.mean(seg)
            if m > 0:
                norm_list.append(seg / m)
        norm_sp = np.concatenate(norm_list)
    var = np.var(norm_sp)
    hist, _ = np.histogram(norm_sp, bins=50, density=True)
    prob = hist / (hist.sum() + 1e-12)
    prob = prob[prob > 0]
    ent = -np.sum(prob * np.log(prob))
    return var, ent

# ============================================================
# 3. 选择 T 值，计算方差和熵
# ============================================================
T_min = max(50, positive_zeros[0] * 2)
T_max = positive_zeros[-1] * 0.95
num_T = 20  # 采样点数量
T_array = np.logspace(np.log10(T_min), np.log10(T_max), num_T)

lnT_list, var_list, ent_list = [], [], []

print("\n正在计算统计量...")
for T in T_array:
    idx = np.searchsorted(positive_zeros, T)
    if idx < 50:
        continue
    sub = positive_zeros[:idx]
    var, ent = compute_stats(sub, segment_ratio=50)
    if not np.isnan(var):
        lnT_list.append(np.log(T))
        var_list.append(var)
        ent_list.append(ent)
        print(f"  T={T:7.1f}, lnT={np.log(T):.3f}, zeros={idx:4d}, var={var:.4f}")

lnT_arr = np.array(lnT_list)
var_arr = np.array(var_list)

# ============================================================
# 4. 逻辑斯谛拟合
# ============================================================
def sigmoid(x, v0, v1, k, x0):
    return v0 + (v1 - v0) / (1 + np.exp(-k * (x - x0)))

fit_ok = False
if len(lnT_arr) > 8:
    p0 = [var_arr[0], var_arr[-1], 1.0, np.median(lnT_arr)]
    try:
        popt, _ = curve_fit(sigmoid, lnT_arr, var_arr, p0=p0, maxfev=10000)
        v0, v1, k, x0 = popt
        res = var_arr - sigmoid(lnT_arr, *popt)
        r2 = 1 - np.sum(res**2) / np.sum((var_arr - np.mean(var_arr))**2)
        print(f"\n=== 逻辑斯谛拟合结果 ===")
        print(f"  拐点 ℓ₀ = {x0:.3f}")
        print(f"  陡度 k = {k:.3f}")
        print(f"  R² = {r2:.4f}")
        fit_ok = True
    except Exception as e:
        print(f"拟合失败: {e}")

# ============================================================
# 5. 与黎曼ζ结果的对比
# ============================================================
print(f"\n=== 与黎曼ζ零点对比 ===")
print(f"  黎曼ζ: 拐点 ℓ₀ = 7.60,  陡度 k = 0.542,  R² = 0.9968")
if fit_ok:
    print(f"  L函数:  拐点 ℓ₀ = {x0:.3f}, 陡度 k = {k:.3f}, R² = {r2:.4f}")
    print(f"  拐点差异: Δℓ₀ = {abs(x0 - 7.60):.2f}")
    print(f"  陡度差异: Δk = {abs(k - 0.542):.3f}")
else:
    print("  L函数无法拟合，说明弛豫行为与黎曼ζ存在显著差异")

# ============================================================
# 6. 画图
# ============================================================
plt.figure(figsize=(10, 6))
plt.scatter(lnT_arr, var_arr, c='blue', s=40, label='Dirichlet L-function (mod 5)')

if fit_ok:
    x_fit = np.linspace(lnT_arr.min(), lnT_arr.max(), 200)
    y_fit = sigmoid(x_fit, *popt)
    plt.plot(x_fit, y_fit, 'r-', lw=1.5, label=f'Sigmoid fit (k={k:.3f}, R²={r2:.4f})')

plt.axhline(y=0.178, color='gray', linestyle='--', label='GUE limit (0.178)')
plt.xlabel(r'$\ln T$', fontsize=14)
plt.ylabel('Variance of normalized spacings', fontsize=14)
plt.title('Dirichlet L-function (mod 2800) spacing variance', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lfunc_mod2800_variance.pdf', dpi=150)
plt.show()

print("\n图片已保存为 lfunc_mod2800_variance.pdf")