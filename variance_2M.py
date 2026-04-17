import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import entropy
import time

# 读取零点数据
filename = "zeros200.txt"
zeros = np.loadtxt(filename).ravel()
zeros.sort()
print(f"Successfully read {len(zeros)} zeros")

# 参数设置
T_min = 20.0
T_max = zeros[-1] * 0.95
num_T = 30
T_array = np.logspace(np.log10(T_min), np.log10(T_max), num_T)

def sigmoid(x, v0, v1, k, x0):
    return v0 + (v1 - v0) / (1 + np.exp(-k * (x - x0)))

def spacing_variance(gammas, segment_ratio=50):
    n = len(gammas)
    if n < 50:
        return np.nan
    spacings = np.diff(gammas)
    seg_size = max(20, n // segment_ratio)
    n_seg = len(spacings) // seg_size
    if n_seg < 2:
        mean_sp = np.mean(spacings)
        if mean_sp == 0:
            return np.nan
        norm_sp = spacings / mean_sp
        return np.var(norm_sp)
    norm_list = []
    for i in range(n_seg):
        seg = spacings[i*seg_size:(i+1)*seg_size]
        m = np.mean(seg)
        if m > 0:
            norm_list.append(seg / m)
    norm_all = np.concatenate(norm_list)
    return np.var(norm_all)

def spacing_entropy(gammas, nbins=50):
    n = len(gammas)
    if n < 100:
        return np.nan
    spacings = np.diff(gammas)
    seg_size = max(20, n // 50)
    n_seg = len(spacings) // seg_size
    if n_seg < 2:
        mean_sp = np.mean(spacings)
        if mean_sp == 0:
            return np.nan
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
    return entropy(prob)

# ============================================================
# 1. Different segment ratios for variance (robustness check)
# ============================================================
ratios = [25, 50, 100]
results_var = {r: {'lnT': [], 'var': []} for r in ratios}
for T in T_array:
    idx = np.searchsorted(zeros, T)
    if idx < 100:
        continue
    sub = zeros[:idx]
    lnT = np.log(T)
    for r in ratios:
        var = spacing_variance(sub, segment_ratio=r)
        if not np.isnan(var):
            results_var[r]['lnT'].append(lnT)
            results_var[r]['var'].append(var)

print("\n=== Variance with different segment ratios ===")
fit_var = {}
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
        fit_var[r] = (x0, k, r2, popt)
        print(f"ratio={r}: inflection={x0:.3f}, k={k:.3f}, R²={r2:.4f}")
    except:
        print(f"ratio={r} failed")

# ============================================================
# 2. Entropy of spacings (using segment_ratio=50)
# ============================================================
lnT_ent = []
ent_vals = []
for T in T_array:
    idx = np.searchsorted(zeros, T)
    if idx < 100:
        continue
    sub = zeros[:idx]
    lnT = np.log(T)
    ent = spacing_entropy(sub, nbins=50)
    if not np.isnan(ent):
        lnT_ent.append(lnT)
        ent_vals.append(ent)

lnT_ent = np.array(lnT_ent)
ent_vals = np.array(ent_vals)
try:
    p0_ent = [ent_vals[0], ent_vals[-1], 1.0, np.median(lnT_ent)]
    popt_ent, _ = curve_fit(sigmoid, lnT_ent, ent_vals, p0=p0_ent, maxfev=5000)
    v0_e, v1_e, k_e, x0_e = popt_ent
    res_e = ent_vals - sigmoid(lnT_ent, *popt_ent)
    ss_res_e = np.sum(res_e**2)
    ss_tot_e = np.sum((ent_vals - np.mean(ent_vals))**2)
    r2_e = 1 - ss_res_e/ss_tot_e
    print(f"\n=== Spacing entropy sigmoid fit ===")
    print(f"inflection={x0_e:.3f}, k={k_e:.3f}, R²={r2_e:.4f}")
except:
    print("Entropy fit failed")

# ============================================================
# 3. Plotting (all labels in English)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: variance with different segment ratios
for r, (x0, k, r2, popt) in fit_var.items():
    lnT_arr = np.array(results_var[r]['lnT'])
    var_arr = np.array(results_var[r]['var'])
    ax1.scatter(lnT_arr, var_arr, alpha=0.5, label=f'ratio={r} data')
    x_fit = np.linspace(lnT_arr.min(), lnT_arr.max(), 200)
    y_fit = sigmoid(x_fit, *popt)
    ax1.plot(x_fit, y_fit, '--', label=f'ratio={r} fit (inflection={x0:.2f})')
ax1.set_xlabel('ln(T)')
ax1.set_ylabel('Variance of normalized spacings')
ax1.set_title('Robustness check: varying segment ratio')
ax1.legend()
ax1.grid(True)

# Right: entropy of spacings
if 'popt_ent' in locals():
    ax2.scatter(lnT_ent, ent_vals, color='green', label='data')
    x_fit = np.linspace(lnT_ent.min(), lnT_ent.max(), 200)
    y_fit = sigmoid(x_fit, *popt_ent)
    ax2.plot(x_fit, y_fit, 'r-', label=f'Sigmoid fit (inflection={x0_e:.2f})')
    ax2.set_xlabel('ln(T)')
    ax2.set_ylabel('Shannon entropy of normalized spacings')
    ax2.set_title('Sigmoidal transition of spacing entropy')
    ax2.legend()
    ax2.grid(True)

plt.tight_layout()
plt.savefig('supplementary_fits_english.png', dpi=150)
plt.show()

print("\nSupplementary experiments completed. Image saved as supplementary_fits_english.png")