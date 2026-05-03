import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
np.random.seed(42)

# ============================================================
# 1. 生成 CUE 矩阵的本征值，并提取完整的归一化间距
# ============================================================
def generate_cue_eigenvalues(N, num_matrices=1000):
    all_spacings = []
    for _ in range(num_matrices):
        A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        Q, R = np.linalg.qr(A)
        D = np.diag(np.diag(R) / np.abs(np.diag(R)))
        U = Q @ D
        phases = np.angle(np.linalg.eigvals(U))
        phases.sort()
        diffs = np.diff(phases)
        circular_gap = 2 * np.pi - (phases[-1] - phases[0])
        all_diffs = np.concatenate([diffs, [circular_gap]])
        spacings = all_diffs * N / (2 * np.pi)
        all_spacings.extend(spacings)
    return np.array(all_spacings)

# ============================================================
# 2. 计算方差与香农熵
# ============================================================
def compute_variance_and_entropy(spacings, nbins=50):
    var = np.var(spacings)
    counts, _ = np.histogram(spacings, bins=nbins)
    prob = counts / counts.sum()
    prob = prob[prob > 0]
    ent = -np.sum(prob * np.log(prob))
    return var, ent

# ============================================================
# 3. 模型函数与拟合比较（使用AICc）
# ============================================================
def sigmoid(x, v0, v1, k, x0):
    return v0 + (v1 - v0) / (1 + np.exp(-k * (x - x0)))

def power_law(x, a, b, c):
    return a + b * np.power(x, c)

def log_decay(x, a, b):
    return a + b / x

def fit_and_compare(x, y, model_func, p0, model_name):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(model_func, x, y, p0=p0, maxfev=10000)
        y_pred = model_func(x, *popt)
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot
        n = len(y)
        k = len(popt)
        aic = n * np.log(ss_res / n) + 2 * k
        # 转换为AICc
        aic += 2 * k * (k + 1) / (n - k - 1)
        return popt, r2, aic
    except Exception as e:
        print(f"   {model_name} 拟合失败: {e}")
        return None, -np.inf, np.inf

# ============================================================
# 4. 主计算
# ============================================================
N_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
          25, 30, 35, 40, 50, 60, 80, 100, 150, 200, 300, 500]
lnN = np.log(N_list)
var_list = []
ent_list = []

print("计算真实 CUE 统计量 (每个 N 生成 1000 个矩阵)...")
for N in N_list:
    spacings = generate_cue_eigenvalues(N, num_matrices=1000)
    var, ent = compute_variance_and_entropy(spacings)
    var_list.append(var)
    ent_list.append(ent)
    print(f"  N = {N:3d}, 间距数 = {len(spacings):6d}, Variance = {var:.4f}, Entropy = {ent:.4f}")

var_list = np.array(var_list)
ent_list = np.array(ent_list)

# ============================================================
# 5. 模型比较（方差）
# ============================================================
print("\n===== 方差模型比较 =====")
p0_sig = [var_list[0], var_list[-1], 1.0, np.median(lnN)]
popt_sig, r2_sig, aic_sig = fit_and_compare(lnN, var_list, sigmoid, p0_sig, "Sigmoid")
if popt_sig is not None:
    print(f"Sigmoid:  R²={r2_sig:.4f}, AICc={aic_sig:.2f}")

popt_pow, r2_pow, aic_pow = fit_and_compare(lnN, var_list, power_law, [0.2, 0.5, -1.0], "Power-law")
if popt_pow is not None:
    print(f"Power-law: R²={r2_pow:.4f}, AICc={aic_pow:.2f}")

popt_log, r2_log, aic_log = fit_and_compare(lnN, var_list, log_decay, [0.2, 0.5], "Logarithmic")
if popt_log is not None:
    print(f"Logarithmic: R²={r2_log:.4f}, AICc={aic_log:.2f}")

# 打印ΔAICc表格
aicc_min = min(aic_sig, aic_pow, aic_log)
print(f"\n{'Model':<20} {'R²':<8} {'AICc':<10} {'ΔAICc':<8}")
print("-" * 50)
print(f"{'Sigmoid':<20} {r2_sig:<8.4f} {aic_sig:<10.2f} {aic_sig - aicc_min:<8.2f}")
print(f"{'Power-law':<20} {r2_pow:<8.4f} {aic_pow:<10.2f} {aic_pow - aicc_min:<8.2f}")
print(f"{'Logarithmic':<20} {r2_log:<8.4f} {aic_log:<10.2f} {aic_log - aicc_min:<8.2f}")

# ============================================================
# 6. 绘图
# ============================================================
x_fit = np.linspace(lnN[0], lnN[-1], 200)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(lnN, var_list, c='black', s=20)
if popt_sig is not None:
    plt.plot(x_fit, sigmoid(x_fit, *popt_sig), 'r-', label=f'Sigmoid (AICc={aic_sig:.1f})')
if popt_pow is not None:
    plt.plot(x_fit, power_law(x_fit, *popt_pow), 'b--', label=f'Power-law (AICc={aic_pow:.1f})')
if popt_log is not None:
    plt.plot(x_fit, log_decay(x_fit, *popt_log), 'g-.', label=f'Logarithmic (AICc={aic_log:.1f})')
plt.xlabel(r'$\ln N$')
plt.ylabel('Variance')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(lnN, ent_list, c='black', s=20)
plt.xlabel(r'$\ln N$')
plt.ylabel('Entropy')

plt.tight_layout()
plt.savefig('cue_corrected_results.pdf')
plt.show()

print("\n图片已保存为 cue_corrected_results.pdf")