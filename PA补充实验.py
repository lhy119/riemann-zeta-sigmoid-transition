import numpy as np
from scipy.linalg import eigvalsh
from scipy.optimize import curve_fit
from scipy.stats import entropy
import matplotlib.pyplot as plt

# ============================================================
# 1. 定义计算CUE特征值统计量的函数
# ============================================================
def cue_spectral_stats(N, num_samples=2000):
    """
    计算CUE特征值的平均最近邻间距方差和熵
    N: 矩阵维度
    num_samples: 随机矩阵个数
    返回: (方差均值, 方差标准差, 熵均值, 熵标准差)
    """
    variances = []
    entropies = []
    for _ in range(num_samples):
        # 生成随机厄米矩阵（GUE）再映射到CUE
        Z = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        H = (Z + Z.conj().T) / np.sqrt(2)
        ev = eigvalsh(H)          # 特征值
        ev_sorted = np.sort(ev)
        spacings = np.diff(ev_sorted)
        mean_sp = np.mean(spacings)
        norm_sp = spacings / mean_sp   # 归一化间距
        variances.append(np.var(norm_sp))
        # 熵
        hist, _ = np.histogram(norm_sp, bins=50, density=True)
        prob = hist / (hist.sum() + 1e-12)
        prob = prob[prob > 0]
        entropies.append(entropy(prob))
    return np.mean(variances), np.std(variances), np.mean(entropies), np.std(entropies)

# ============================================================
# 2. 定义Sigmoid函数
# ============================================================
def sigmoid(x, v0, v1, k, x0):
    return v0 + (v1 - v0) / (1 + np.exp(-k * (x - x0)))

# ============================================================
# 3. 对不同矩阵尺寸N进行计算
# ============================================================
N_list = [10, 20, 30, 50, 80, 100, 150, 200, 300, 500]
lnN = np.log(N_list)
mean_var = []
std_var = []
mean_ent = []
std_ent = []

print("正在计算CUE特征值统计量，请稍候...")
for N in N_list:
    print(f"  N = {N}")
    v, sv, e, se = cue_spectral_stats(N, num_samples=1000)   # 每个N用1000个随机矩阵
    mean_var.append(v)
    std_var.append(sv)
    mean_ent.append(e)
    std_ent.append(se)

mean_var = np.array(mean_var)
std_var = np.array(std_var)
mean_ent = np.array(mean_ent)
std_ent = np.array(std_ent)

# ============================================================
# 4. 拟合方差与lnN的关系
# ============================================================
p0_var = [mean_var[0], mean_var[-1], 1.0, np.median(lnN)]
popt_var, _ = curve_fit(sigmoid, lnN, mean_var, p0=p0_var, maxfev=5000)
x0_var, k_var = popt_var[3], popt_var[2]
print(f"\nCUE方差拟合: 拐点 lnN = {x0_var:.3f}, k = {k_var:.3f}")

# 计算R²
res_var = mean_var - sigmoid(lnN, *popt_var)
ss_res = np.sum(res_var**2)
ss_tot = np.sum((mean_var - np.mean(mean_var))**2)
r2_var = 1 - ss_res/ss_tot
print(f"  R² = {r2_var:.4f}")

# ============================================================
# 5. 拟合熵与lnN的关系
# ============================================================
p0_ent = [mean_ent[0], mean_ent[-1], 1.0, np.median(lnN)]
popt_ent, _ = curve_fit(sigmoid, lnN, mean_ent, p0=p0_ent, maxfev=5000)
x0_ent, k_ent = popt_ent[3], popt_ent[2]
print(f"CUE熵拟合: 拐点 lnN = {x0_ent:.3f}, k = {k_ent:.3f}")
res_ent = mean_ent - sigmoid(lnN, *popt_ent)
ss_res = np.sum(res_ent**2)
ss_tot = np.sum((mean_ent - np.mean(mean_ent))**2)
r2_ent = 1 - ss_res/ss_tot
print(f"  R² = {r2_ent:.4f}")

# ============================================================
# 6. 绘图：方差 vs lnN
# ============================================================
plt.figure(figsize=(8,6))
plt.errorbar(lnN, mean_var, yerr=std_var, fmt='o', capsize=3, label='CUE data')
x_fit = np.linspace(lnN[0], lnN[-1], 200)
y_fit = sigmoid(x_fit, *popt_var)
plt.plot(x_fit, y_fit, 'r-', label=f'Sigmoid fit (inflection={x0_var:.2f})')
plt.xlabel(r'$\ln N$')
plt.ylabel('Variance of normalized spacings')
plt.title('CUE eigenvalue spacing variance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('cue_variance.pdf', dpi=300)
plt.show()

# 同样画熵的图
plt.figure(figsize=(8,6))
plt.errorbar(lnN, mean_ent, yerr=std_ent, fmt='s', capsize=3, label='CUE data')
x_fit = np.linspace(lnN[0], lnN[-1], 200)
y_fit = sigmoid(x_fit, *popt_ent)
plt.plot(x_fit, y_fit, 'r-', label=f'Sigmoid fit (inflection={x0_ent:.2f})')
plt.xlabel(r'$\ln N$')
plt.ylabel('Shannon entropy of normalized spacings')
plt.title('CUE eigenvalue spacing entropy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('cue_entropy.pdf', dpi=300)
plt.show()

print("\n所有计算完成，图片已保存: cue_variance.pdf, cue_entropy.pdf")