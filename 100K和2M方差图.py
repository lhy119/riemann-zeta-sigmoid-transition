import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import entropy

# ========== 读取零点数据（200万） ==========
filename = "zeros200.txt"
zeros = np.loadtxt(filename).ravel()
zeros.sort()
print(f"Successfully read {len(zeros)} zeros\n")

# ========== 参数设置 ==========
T_min = 20.0
T_max = zeros[-1] * 0.95
num_T = 30
T_array = np.logspace(np.log10(T_min), np.log10(T_max), num_T)

def sigmoid(x, v0, v1, k, x0):
    return v0 + (v1 - v0) / (1 + np.exp(-k * (x - x0)))

def spacing_variance(gammas, segment_ratio=50):
    n = len(gammas)
    if n < 50:
        return np.nan, 0
    spacings = np.diff(gammas)
    seg_size = max(20, n // segment_ratio)
    n_seg = len(spacings) // seg_size
    if n_seg < 2:
        mean_sp = np.mean(spacings)
        if mean_sp == 0:
            return np.nan, 0
        norm_sp = spacings / mean_sp
        return np.var(norm_sp), n
    norm_list = []
    for i in range(n_seg):
        seg = spacings[i*seg_size:(i+1)*seg_size]
        m = np.mean(seg)
        if m > 0:
            norm_list.append(seg / m)
    norm_all = np.concatenate(norm_list)
    return np.var(norm_all), n

def spacing_entropy(gammas, nbins=50):
    n = len(gammas)
    if n < 100:
        return np.nan, 0
    spacings = np.diff(gammas)
    seg_size = max(20, n // 50)
    n_seg = len(spacings) // seg_size
    if n_seg < 2:
        mean_sp = np.mean(spacings)
        if mean_sp == 0:
            return np.nan, 0
        norm_sp = spacings / mean_sp
    else:
        norm_list = []
        for i in range(n_seg):
            seg = spacings[i*seg_size:(i+1)*seg_size]
            m = np.mean(seg)
            if m > 0:
                norm_list.append(seg / m)
        norm_sp = np.concatenate(norm_list)
    hist, _ = np.histogram(norm_sp, bins=nbins, density=True)
    prob = hist / (hist.sum() + 1e-12)
    prob = prob[prob > 0]
    return entropy(prob), n

# ============================================================
# 1. 计算不同分段比率下的方差（含详细打印）
# ============================================================
ratios = [25, 50, 100]
results_var = {r: {'lnT': [], 'var': [], 'zeros': []} for r in ratios}
print("=== 2M variance calculation (ratio=50 shown) ===")
for T in T_array:
    idx = np.searchsorted(zeros, T)
    if idx < 100:
        continue
    sub = zeros[:idx]
    lnT = np.log(T)
    # 只对 ratio=50 进行逐点打印，其他 ratio 仍计算但不打印
    for r in ratios:
        var, n = spacing_variance(sub, segment_ratio=r)
        if not np.isnan(var):
            results_var[r]['lnT'].append(lnT)
            results_var[r]['var'].append(var)
            results_var[r]['zeros'].append(n)
            if r == 50:   # 只打印 ratio=50 的点
                print(f"T={T:.2e}, lnT={lnT:.2f}, zeros={n}, var={var:.4f}")

# 存储每个 ratio 的拟合参数
fit_params = {}
print("\n=== Variance fits ===")
for r in ratios:
    lnT_arr = np.array(results_var[r]['lnT'])
    var_arr = np.array(results_var[r]['var'])
    if len(lnT_arr) < 10:
        continue
    p0 = [var_arr[0], var_arr[-1], 1.0, np.median(lnT_arr)]
    try:
        popt, _ = curve_fit(sigmoid, lnT_arr, var_arr, p0=p0, maxfev=5000)
        v0, v1, k, x0 = popt
        res = var_arr - sigmoid(lnT_arr, *popt)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((var_arr - np.mean(var_arr))**2)
        r2 = 1 - ss_res/ss_tot
        fit_params[r] = (popt, r2, lnT_arr, var_arr)
        print(f"ratio={r}: inflection={x0:.3f}, k={k:.3f}, R²={r2:.4f}")
    except Exception as e:
        print(f"ratio={r} failed: {e}")

# ============================================================
# 2. 计算熵（使用 ratio=50，含打印）
# ============================================================
lnT_ent = []
ent_vals = []
print("\n=== Entropy calculation ===")
for T in T_array:
    idx = np.searchsorted(zeros, T)
    if idx < 100:
        continue
    sub = zeros[:idx]
    lnT = np.log(T)
    ent, n = spacing_entropy(sub, nbins=50)
    if not np.isnan(ent):
        lnT_ent.append(lnT)
        ent_vals.append(ent)
        print(f"T={T:.2e}, lnT={lnT:.2f}, zeros={n}, entropy={ent:.4f}")

lnT_ent = np.array(lnT_ent)
ent_vals = np.array(ent_vals)
ent_fit_params = None
if len(lnT_ent) > 10:
    p0 = [ent_vals[0], ent_vals[-1], 1.0, np.median(lnT_ent)]
    try:
        popt_ent, _ = curve_fit(sigmoid, lnT_ent, ent_vals, p0=p0, maxfev=5000)
        v0_e, v1_e, k_e, x0_e = popt_ent
        res_e = ent_vals - sigmoid(lnT_ent, *popt_ent)
        ss_res_e = np.sum(res_e**2)
        ss_tot_e = np.sum((ent_vals - np.mean(ent_vals))**2)
        r2_e = 1 - ss_res_e/ss_tot_e
        ent_fit_params = (popt_ent, r2_e)
        print(f"\nEntropy fit: inflection={x0_e:.3f}, k={k_e:.3f}, R²={r2_e:.4f}")
    except Exception as e:
        print(f"Entropy fit failed: {e}")

# ============================================================
# 图1：方差拟合（三个 ratio 在同一张图上）
# ============================================================
plt.figure(figsize=(8,6))
colors = ['blue', 'red', 'green']
markers = ['o', 's', '^']
for idx, r in enumerate(ratios):
    if r not in fit_params:
        continue
    popt, r2, lnT_arr, var_arr = fit_params[r]
    x0, k = popt[3], popt[2]
    plt.scatter(lnT_arr, var_arr, color=colors[idx], marker=markers[idx], alpha=0.6, s=30, label=f'ratio={r} data')
    x_fit = np.linspace(min(lnT_arr), max(lnT_arr), 200)
    y_fit = sigmoid(x_fit, *popt)
    plt.plot(x_fit, y_fit, color=colors[idx], linestyle='-', linewidth=1.5,
             label=f'ratio={r}: ℓ₀={x0:.2f}, k={k:.2f}, R²={r2:.4f}')
plt.axhline(y=0.178, color='gray', linestyle='--', label='GUE value (0.178)')
plt.xlabel(r'$\ln T$', fontsize=12)
plt.ylabel('Variance of normalized spacings', fontsize=12)
plt.title('Variance transition with different segmentation ratios', fontsize=14)
plt.legend(fontsize=9, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('variance_2M.pdf', dpi=300)
plt.show()

# ============================================================
# 图2：熵拟合
# ============================================================
if ent_fit_params is not None:
    popt_ent, r2_e = ent_fit_params
    x0_e, k_e = popt_ent[3], popt_ent[2]
    plt.figure(figsize=(8,6))
    plt.scatter(lnT_ent, ent_vals, color='green', alpha=0.7, s=30, label='Data (2M zeros)')
    x_fit = np.linspace(min(lnT_ent), max(lnT_ent), 200)
    y_fit = sigmoid(x_fit, *popt_ent)
    plt.plot(x_fit, y_fit, 'r-', linewidth=2,
             label=f'Sigmoid fit: ℓ₀={x0_e:.2f}, k={k_e:.3f}, R²={r2_e:.4f}')
    plt.xlabel(r'$\ln T$', fontsize=12)
    plt.ylabel('Shannon entropy of normalized spacings', fontsize=12)
    plt.title('Entropy transition (2M zeros, r=50)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('entropy_2M.pdf', dpi=300)
    plt.show()

# ============================================================
# ★ 新增：合并 100k 与 2M 方差图（论文用 Figure 1）★
# ============================================================
# 硬编码 100k 数据（来自之前运行的输出）
lnT_100k = [4.41, 4.69, 4.97, 5.25, 5.53, 5.82, 6.10, 6.38, 6.66, 6.94,
            7.23, 7.51, 7.79, 8.07, 8.35, 8.64, 8.92, 9.20, 9.48, 9.76,
            10.05, 10.33, 10.61, 10.89, 11.17]
var_100k = [0.1296, 0.1213, 0.1292, 0.1258, 0.1322, 0.1374, 0.1388, 0.1417,
            0.1476, 0.1508, 0.1527, 0.1556, 0.1567, 0.1563, 0.1554, 0.1555,
            0.1557, 0.1560, 0.1567, 0.1573, 0.1581, 0.1588, 0.1594, 0.1601,
            0.1608]

# 2M 数据取 ratio=50 的结果
lnT_2M = np.array(results_var[50]['lnT'])
var_2M = np.array(results_var[50]['var'])

# 合并图
plt.figure(figsize=(10, 6))
plt.scatter(lnT_100k, var_100k, c='royalblue', s=25, alpha=0.7, label='100k zeros')
plt.scatter(lnT_2M, var_2M, c='firebrick', s=15, alpha=0.7, label='2M zeros (r=50)')

# 画下降支 sigmoid 拟合（使用 ratio=50 的拟合参数）
ell = np.linspace(6.0, 14.0, 200)
if 50 in fit_params:
    popt_50, _, _, _ = fit_params[50]
    v0, v1, k, ell0 = popt_50
    sigmoid_curve = v1 + (v0 - v1) / (1 + np.exp(-k * (ell - ell0)))
    mask = ell >= np.log(500)
    plt.plot(ell[mask], sigmoid_curve[mask], 'r-', linewidth=2.5,
             label='Sigmoid fit ($T \geq 500$)')

plt.axvline(x=np.log(500), color='gray', linestyle='--', linewidth=1.2)
plt.text(np.log(500)+0.05, 0.185, '$T=500$', color='gray', fontsize=11)
plt.annotate('weak overshoot', xy=(6.8, 0.165), xytext=(5.2, 0.15),
             arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.2),
             fontsize=11, color='dimgray')
plt.axhline(y=0.178, color='black', linestyle=':', linewidth=1.5)
plt.text(13.5, 0.179, 'GUE limit 0.178', fontsize=11, va='bottom', ha='right')

plt.xlabel(r'$\ln T$', fontsize=14)
plt.ylabel('Variance of normalized spacings', fontsize=14)
plt.title('Combined variance data and sigmoid fit', fontsize=15)
plt.legend(loc='lower left', fontsize=11)
plt.xlim(3.5, 14.5)
plt.ylim(0.10, 0.24)
plt.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()
plt.savefig('combined_variance.pdf', dpi=300)
plt.show()

print("\nAll figures saved: variance_2M.pdf, entropy_2M.pdf, combined_variance.pdf")
