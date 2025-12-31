import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from numpy.linalg import norm, cholesky, solve
import warnings
from tqdm import tqdm
import pandas as pd
from time import time
import os

# 忽略警告
warnings.filterwarnings('ignore')

# 创建结果保存目录
os.makedirs('experiment_results', exist_ok=True)

# ====================================================================
# 1. 参数设置
# ====================================================================
# 数据维度组合
n_p_pairs = [
    (100, 20),  # 小样本+低维
    (200, 50),  # 中样本+中维
    (500, 100),  # 大样本+中维
    (200, 200)  # 中样本+高维（p=n）
]

# 实验参数
max_iter = 100
n_trials = 30
lambda_ratio = 0.1

# 算法配置（按类别分组）
algo_configs = {
    # --- [1] 坐标下降类 ---
    'BCD (Adaptive)': {'color': 'firebrick', 'style': '-', 'width': 2.5, 'group': 'Coordinate Descent'},

    # --- [2] Huber平滑类 ---
    'Huber Gradient (Adaptive)': {'color': 'limegreen', 'style': '-', 'width': 2.5, 'group': 'Huber'},
    'Huber Gradient (Accel+Adaptive)': {'color': 'forestgreen', 'style': '-', 'width': 2.5, 'group': 'Huber'},
    'Huber (Accel+Restart+Adaptive)': {'color': 'darkgreen', 'style': '--', 'width': 2.5, 'group': 'Huber'},

    # --- [3] 近端梯度类 ---
    'FISTA (Adaptive)': {'color': 'darkblue', 'style': '-', 'width': 2.5, 'group': 'Proximal Gradient'},
    'FISTA (Restart+Adaptive)': {'color': 'blue', 'style': '--', 'width': 2.5, 'group': 'Proximal Gradient'},

    # --- [4] 分裂乘子类 ---
    'ADMM (rho=0.5)': {'color': 'orange', 'style': '-', 'width': 2, 'group': 'ADMM'},
    'ADMM (rho=1.0)': {'color': 'red', 'style': '--', 'width': 2, 'group': 'ADMM'},  # 改为红色虚线
    'ADMM (rho=2.0)': {'color': 'purple', 'style': ':', 'width': 2, 'group': 'ADMM'},  # 改为紫色点线

    # --- [5] 随机优化类 ---
    'SGD (Adaptive)': {'color': 'brown', 'style': '-', 'width': 2, 'group': 'Stochastic'},

    # --- [6] 次梯度类 ---
    'Subgradient (Adaptive)': {'color': 'gray', 'style': '-', 'width': 2, 'group': 'Subgradient'}
}

# 按算法分组
algorithm_groups = {
    'Coordinate Descent': ['BCD (Adaptive)'],
    'Proximal Gradient': ['FISTA (Adaptive)', 'FISTA (Restart+Adaptive)'],
    'ADMM': ['ADMM (rho=0.5)', 'ADMM (rho=1)', 'ADMM (rho=2)'],
    'Stochastic': ['SGD (Adaptive)'],
    'Subgradient': ['Subgradient (Adaptive)']
}

# 初始化结果存储
all_results = {
    (n, p): {name: [] for name in algo_configs.keys()}
    for n, p in n_p_pairs
}
all_trial_times = {
    (n, p): {name: [] for name in algo_configs.keys()}
    for n, p in n_p_pairs
}

# ====================================================================
# 2. 辅助函数
# ====================================================================
def soft_threshold(x, tau):
    """软阈值函数"""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)

def lasso_objective(beta, X, y, n, lam):
    """计算LASSO目标函数值"""
    residual = X @ beta - y
    l2_loss = (0.5 / n) * (residual @ residual)
    l1_norm = lam * norm(beta, 1)
    return l2_loss + l1_norm

def get_algo_params(algo_name, n, p):
    """根据算法名称和数据维度获取自适应参数"""
    params = {}
    # 维度特征判断
    is_high_dim = p >= 100
    is_large_sample = n >= 500
    is_square = n == p

    # BCD参数
    if 'BCD' in algo_name:
        params.update({
            'block_size': min(20, p // 5) if is_high_dim else 1,
            'max_iter_adjust': int(max_iter * 0.8 if is_high_dim else max_iter)
        })
    # FISTA参数
    elif 'FISTA' in algo_name:
        params.update({
            'L_scale': 1.2 if is_high_dim else 1.0,
            'alpha_scale': 0.8 if is_large_sample else 1.0,
            'restart_threshold': 0.1 if is_high_dim else 0.0
        })
    # ADMM参数
    elif 'ADMM' in algo_name:
        params.update({
            'reuse_cholesky': True if is_large_sample else False
        })
    # SGD参数
    elif 'SGD' in algo_name:
        params.update({
            'batch_size': min(64, n // 10) if is_large_sample else 32,
            'lr': 0.005 if is_high_dim else 0.01,
            'lr_decay': 0.99 if is_large_sample else 1.0
        })
    # Subgradient参数
    elif 'Subgradient' in algo_name:
        params.update({
            'lr': 0.003 if is_high_dim else 0.01,
            'lr_decay': 0.98 if is_large_sample else 1.0
        })
    return params

# ====================================================================
# 3. 算法求解器（删除Huber相关函数）
# ====================================================================
# ==========================================
# 3.1 BCD（分块坐标下降）
# ==========================================
def bcd_adaptive(X, y, n, p, lam, max_iter, f_star, params):
    """自适应块坐标下降"""
    beta = np.zeros(p)
    history = []
    block_size = params['block_size']
    max_iter_adj = params['max_iter_adjust']

    # 分块索引
    blocks = [np.arange(i, min(i + block_size, p)) for i in range(0, p, block_size)]

    # 预计算
    A_j = np.zeros(p)
    for j in range(p):
        A_j[j] = (X[:, j] @ X[:, j]) / n
        if A_j[j] == 0:
            A_j[j] = 1e-8

    for k in range(max_iter_adj):
        # 随机化块顺序以提升收敛
        np.random.shuffle(blocks)
        for block in blocks:
            for j in block:
                old_beta_j = beta[j]
                residual_no_j = y - (X @ beta - X[:, j] * old_beta_j)
                c_j = (X[:, j] @ residual_no_j) / n
                beta[j] = soft_threshold(c_j / A_j[j], lam / A_j[j])

        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))

    # 补全长度
    if len(history) < max_iter:
        history += [history[-1]] * (max_iter - len(history))
    return history

# ==========================================
# 3.2 FISTA类
# ==========================================
def fista_adaptive(X, y, n, p, lam, max_iter, f_star, params):
    """自适应FISTA"""
    beta = np.zeros(p)
    z = np.zeros(p)
    t = 1.0
    history = []

    L = norm(X.T @ X / n, ord=2) * params['L_scale']
    alpha = (1.0 / L) * params['alpha_scale'] if L > 0 else 0.01

    for k in range(max_iter):
        beta_old = beta.copy()
        # 近端梯度步骤
        grad_z = (X.T @ (X @ z - y)) / n
        beta = soft_threshold(z - alpha * grad_z, alpha * lam)
        # Nesterov加速
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        z = beta + ((t - 1) / t_new) * (beta - beta_old)
        t = t_new

        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history

def fista_restart_adaptive(X, y, n, p, lam, max_iter, f_star, params):
    """自适应重启FISTA"""
    beta = np.zeros(p)
    z = np.zeros(p)
    t = 1.0
    history = []

    L = norm(X.T @ X / n, ord=2) * params['L_scale']
    alpha = (1.0 / L) * params['alpha_scale'] if L > 0 else 0.01
    restart_threshold = params['restart_threshold']

    for k in range(max_iter):
        beta_old = beta.copy()
        # 近端梯度步骤
        grad_z = (X.T @ (X @ z - y)) / n
        beta_new = soft_threshold(z - alpha * grad_z, alpha * lam)
        # 动态重启逻辑
        if np.dot(z - beta_new, beta_new - beta_old) > restart_threshold:
            t_new = 1.0
            z = beta_new
        else:
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            z = beta_new + ((t - 1) / t_new) * (beta_new - beta_old)

        beta = beta_new
        t = t_new
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history

# ==========================================
# 3.3 ADMM
# ==========================================
def admm_adaptive(X, y, n, p, lam, rho, max_iter, f_star, params):
    """自适应ADMM"""
    beta = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    history = []
    I = np.identity(p)

    # 大样本复用Cholesky分解
    if params['reuse_cholesky']:
        L_cho = cholesky(X.T @ X / n + rho * I)

    for k in range(max_iter):
        # x-子问题
        rhs = (X.T @ y / n) + rho * (z - u)
        if params['reuse_cholesky']:
            beta = solve(L_cho.T, solve(L_cho, rhs))
        else:
            beta = np.linalg.solve(X.T @ X / n + rho * I, rhs)
        # z-子问题（软阈值）
        z = soft_threshold(beta + u, lam / rho)
        # 对偶变量更新
        u = u + beta - z

        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history

# ==========================================
# 3.4 随机梯度下降
# ==========================================
def sgd_adaptive(X, y, n, p, lam, max_iter, f_star, params):
    """自适应随机梯度下降"""
    beta = np.zeros(p)
    history = []
    batch_size = params['batch_size']
    lr = params['lr']
    lr_decay = params['lr_decay']

    for k in range(max_iter):
        # 动态批次采样
        idx = np.random.choice(n, size=batch_size, replace=False)
        X_batch = X[idx]
        y_batch = y[idx]
        # 梯度计算
        grad = (X_batch.T @ (X_batch @ beta - y_batch)) / batch_size
        # 梯度更新
        beta = beta - lr * grad
        beta = soft_threshold(beta, lr * lam)
        # 学习率衰减
        lr *= lr_decay

        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history

# ==========================================
# 3.5 次梯度下降
# ==========================================
def subgradient_adaptive(X, y, n, p, lam, max_iter, f_star, params):
    """自适应次梯度下降"""
    beta = np.zeros(p)
    history = []
    lr = params['lr']
    lr_decay = params['lr_decay']

    for k in range(max_iter):
        # 次梯度计算
        grad = (X.T @ (X @ beta - y)) / n
        beta = beta - lr * (grad + lam * np.sign(beta))
        # 学习率衰减
        lr *= lr_decay

        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history

# ====================================================================
# 4. 实验主循环
# ====================================================================
print(f"Starting {n_trials} trials for {len(n_p_pairs)} (n,p) combinations...")
print("=" * 60)

for idx, (n, p) in enumerate(n_p_pairs):
    print(f"\n=== Processing (n={n}, p={p}) [{idx + 1}/{len(n_p_pairs)}] ===")
    # 创建当前维度的进度条
    for i in tqdm(range(n_trials), desc=f"Trials for (n={n},p={p})"):
        # 数据生成
        X = np.random.randn(n, p)
        true_beta = np.zeros(p)
        n_informative = min(10, p // 2)
        true_beta[:n_informative] = np.random.uniform(-5, 5, n_informative)
        y = X @ true_beta + np.random.randn(n) * 0.5

        # 正则化参数计算
        lam_max = norm(X.T @ y, ord=np.inf) / n
        lam = lam_max * lambda_ratio

        # 计算最优解（使用scikit-learn）
        lasso_sklearn = Lasso(alpha=lam, fit_intercept=False, tol=1e-14, max_iter=20000)
        lasso_sklearn.fit(X, y)
        f_star = lasso_objective(lasso_sklearn.coef_, X, y, n, lam)

        # =================== 运行所有算法 ===================
        # 1. BCD算法
        if 'BCD (Adaptive)' in algo_configs:
            params = get_algo_params('BCD (Adaptive)', n, p)
            start = time()
            all_results[(n, p)]['BCD (Adaptive)'].append(
                bcd_adaptive(X, y, n, p, lam, max_iter, f_star, params)
            )
            all_trial_times[(n, p)]['BCD (Adaptive)'].append(time() - start)

        # 2. FISTA算法类
        if 'FISTA (Adaptive)' in algo_configs:
            params = get_algo_params('FISTA (Adaptive)', n, p)
            start = time()
            all_results[(n, p)]['FISTA (Adaptive)'].append(
                fista_adaptive(X, y, n, p, lam, max_iter, f_star, params)
            )
            all_trial_times[(n, p)]['FISTA (Adaptive)'].append(time() - start)
        if 'FISTA (Restart+Adaptive)' in algo_configs:
            params = get_algo_params('FISTA (Restart+Adaptive)', n, p)
            start = time()
            all_results[(n, p)]['FISTA (Restart+Adaptive)'].append(
                fista_restart_adaptive(X, y, n, p, lam, max_iter, f_star, params)
            )
            all_trial_times[(n, p)]['FISTA (Restart+Adaptive)'].append(time() - start)

        # 3. ADMM算法类
        for rho in [0.5, 1.0, 2.0]:
            algo_name = f'ADMM (rho={rho})'
            if algo_name in algo_configs:
                params = get_algo_params(algo_name, n, p)
                start = time()
                all_results[(n, p)][algo_name].append(
                    admm_adaptive(X, y, n, p, lam, rho, max_iter, f_star, params)
                )
                all_trial_times[(n, p)][algo_name].append(time() - start)

        # 4. SGD算法
        if 'SGD (Adaptive)' in algo_configs:
            params = get_algo_params('SGD (Adaptive)', n, p)
            start = time()
            all_results[(n, p)]['SGD (Adaptive)'].append(
                sgd_adaptive(X, y, n, p, lam, max_iter, f_star, params)
            )
            all_trial_times[(n, p)]['SGD (Adaptive)'].append(time() - start)

        # 5. Subgradient算法
        if 'Subgradient (Adaptive)' in algo_configs:
            params = get_algo_params('Subgradient (Adaptive)', n, p)
            start = time()
            all_results[(n, p)]['Subgradient (Adaptive)'].append(
                subgradient_adaptive(X, y, n, p, lam, max_iter, f_star, params)
            )
            all_trial_times[(n, p)]['Subgradient (Adaptive)'].append(time() - start)

    # 显示当前维度完成进度
    print(f"✓ Completed (n={n}, p={p})")

# ====================================================================
# 5. 可视化分析
# ====================================================================
print("\n" + "=" * 60)
print("All trials complete. Generating visualizations...")
print("=" * 60)

# 5.1 收敛曲线对比（子图布局）
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()
k_axis = np.arange(1, max_iter + 1)

for idx, (n, p) in enumerate(n_p_pairs):
    ax = axes[idx]
    results = all_results[(n, p)]

    # 为每个算法绘制曲线
    for algo_name in algo_configs.keys():
        if algo_name not in results or len(results[algo_name]) == 0:
            continue

        histories = results[algo_name]
        data_matrix = np.array(histories)
        min_len = min([len(h) for h in histories])
        data_matrix = data_matrix[:, :min_len]
        current_k_axis = k_axis[:min_len]
        mean_curve = np.mean(data_matrix, axis=0)

        cfg = algo_configs[algo_name]
        color = cfg['color']
        style = cfg['style']
        width = cfg.get('width', 2)

        # 绘制均值曲线
        ax.plot(current_k_axis, mean_curve, color=color, linestyle=style,
                linewidth=width, label=algo_name)

    # 子图设置
    ax.set_yscale('log')
    ax.set_xlabel('Iteration k', fontsize=12)
    ax.set_ylabel('Suboptimality $f(x_k) - f^*$', fontsize=12)
    ax.set_title(f'Convergence: (n={n}, p={p})', fontsize=14, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.set_ylim(bottom=1e-12)
    ax.set_xlim(0, max_iter)

# 创建统一的图例（放在图表外面）
# 收集所有算法的句柄和标签
handles, labels = [], []
for algo_name in algo_configs.keys():
    # 为每个算法创建一个代理线条用于图例
    cfg = algo_configs[algo_name]
    color = cfg['color']
    style = cfg['style']
    width = cfg.get('width', 2)

    # 创建代理线条
    proxy_line = plt.Line2D([0], [0], color=color, linestyle=style,
                            linewidth=width, label=algo_name)
    handles.append(proxy_line)
    labels.append(algo_name)

# 将图例放在图表下方
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9,
           framealpha=0.95, title='Algorithms (Adaptive)', title_fontsize=10)

# 调整布局，为图例留出空间
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

plt.suptitle('LASSO Optimization: Adaptive Algorithms Comparison', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('experiment_results/convergence_adaptive.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.2 运行时间对比（分组柱状图）
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
# 准备数据
algorithms = list(algo_configs.keys())
x = np.arange(len(algorithms))
width = 0.15  # 每个柱子的宽度
colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 不同(n,p)组合的颜色

# 绘制柱状图
for idx, (n, p) in enumerate(n_p_pairs):
    avg_times = []
    for algo in algorithms:
        if algo in all_trial_times[(n, p)] and len(all_trial_times[(n, p)][algo]) > 0:
            avg_times.append(np.mean(all_trial_times[(n, p)][algo]))
        else:
            avg_times.append(0)
    # 计算位置偏移
    offset = (idx - len(n_p_pairs) / 2 + 0.5) * width
    ax.bar(x + offset, avg_times, width, label=f'(n={n}, p={p})', color=colors_bar[idx], edgecolor='black')

# 图表设置
ax.set_xlabel('Adaptive Algorithms', fontsize=12)
ax.set_ylabel('Average Runtime (s)', fontsize=12)
ax.set_title('Runtime Comparison: Adaptive Algorithms by (n,p) Combinations', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=10)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, axis='y', ls='--', alpha=0.4)

# 添加数值标签
for idx, (n, p) in enumerate(n_p_pairs):
    avg_times = []
    for algo in algorithms:
        if algo in all_trial_times[(n, p)] and len(all_trial_times[(n, p)][algo]) > 0:
            avg_times.append(np.mean(all_trial_times[(n, p)][algo]))
        else:
            avg_times.append(0)
    offset = (idx - len(n_p_pairs) / 2 + 0.5) * width
    for j, time_val in enumerate(avg_times):
        if time_val > 0:  # 只显示非零值
            ax.text(j + offset, time_val + 0.001, f'{time_val:.3f}',
                    ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
plt.savefig('experiment_results/runtime_adaptive.png', dpi=300, bbox_inches='tight')
plt.show()

# ====================================================================
# 6. 数据分析与报告
# ====================================================================
print("\n" + "=" * 60)
print("Generating comprehensive analysis report...")
print("=" * 60)

def generate_comprehensive_report():
    """生成详细的分析报告"""
    report_data = []
    for (n, p) in n_p_pairs:
        trial_times = all_trial_times[(n, p)]
        results = all_results[(n, p)]
        for algo_name in algo_configs.keys():
            if algo_name not in trial_times or len(trial_times[algo_name]) == 0:
                continue
            # 基础指标
            avg_time = np.mean(trial_times[algo_name])
            std_time = np.std(trial_times[algo_name])
            # 收敛性能指标
            if algo_name in results and len(results[algo_name]) > 0:
                histories = results[algo_name]
                final_subopts = [h[-1] for h in histories if len(h) > 0]
                avg_final_subopt = np.mean(final_subopts) if final_subopts else np.inf
                std_final_subopt = np.std(final_subopts) if len(final_subopts) > 1 else 0
                # 收敛迭代数（达到1e-3）
                conv_iters = []
                for h in histories:
                    for iter_idx, val in enumerate(h):
                        if val < 1e-3:
                            conv_iters.append(iter_idx + 1)
                            break
                    else:
                        conv_iters.append(np.inf)
                avg_conv_iter = np.mean(conv_iters) if conv_iters else np.inf
                success_rate = np.sum(np.array(conv_iters) < np.inf) / len(conv_iters) * 100
            else:
                avg_final_subopt = np.inf
                std_final_subopt = np.inf
                avg_conv_iter = np.inf
                success_rate = 0
            # 每迭代耗时
            iter_time = avg_time / max_iter if max_iter > 0 else np.inf
            # 算法组信息
            algo_group = algo_configs[algo_name]['group']
            report_data.append({
                'n': n,
                'p': p,
                'Algorithm': algo_name,
                'Algorithm Group': algo_group,
                'Avg Runtime (s)': round(avg_time, 4),
                'Std Runtime (s)': round(std_time, 4),
                'Avg Final Suboptimality': f"{avg_final_subopt:.2e}",
                'Std Final Suboptimality': f"{std_final_subopt:.2e}" if std_final_subopt < np.inf else 'Inf',
                'Avg Iter to 1e-3': round(avg_conv_iter, 1) if not np.isinf(avg_conv_iter) else 'Inf',
                'Success Rate (%)': round(success_rate, 1),
                'Time per Iter (ms)': round(iter_time * 1000, 2),
                'Raw Final Suboptimality': avg_final_subopt
            })
    # 保存详细报告
    df = pd.DataFrame(report_data)
    df.to_csv('experiment_results/comprehensive_analysis_adaptive.csv', index=False, encoding='utf-8-sig')
    # 打印关键汇总
    print("\n=== 自适应算法实验关键结果汇总 ===")
    print("=" * 80)
    # 按算法组汇总
    print("\n1. 按算法组平均性能:")
    print("-" * 80)
    group_summary = df.groupby('Algorithm Group').agg({
        'Avg Runtime (s)': 'mean',
        'Raw Final Suboptimality': 'mean',
        'Success Rate (%)': 'mean'
    }).round(4)
    group_summary['Avg Final Suboptimality'] = group_summary['Raw Final Suboptimality'].apply(
        lambda x: f"{x:.2e}"
    )
    group_summary = group_summary.drop('Raw Final Suboptimality', axis=1)
    group_summary = group_summary.sort_values('Avg Runtime (s)')
    print(group_summary.to_string())
    # 按维度汇总
    print("\n2. 按数据维度最优算法:")
    print("-" * 80)
    for (n, p) in n_p_pairs:
        dim_data = df[(df['n'] == n) & (df['p'] == p)]
        if len(dim_data) > 0:
            # 最快算法
            fastest = dim_data.loc[dim_data['Avg Runtime (s)'].idxmin()]
            # 最精确算法（排除失败情况）
            valid_data = dim_data[dim_data['Success Rate (%)'] > 0]
            if len(valid_data) > 0:
                most_accurate = valid_data.loc[valid_data['Raw Final Suboptimality'].idxmin()]
            else:
                most_accurate = fastest
            print(f"(n={n}, p={p}):")
            print(f"  最快算法: {fastest['Algorithm']} ({fastest['Avg Runtime (s)']:.3f}s)")
            print(f"  最精确算法: {most_accurate['Algorithm']} (次优性: {most_accurate['Avg Final Suboptimality']})")
            print(
                f"  收敛成功率最高: {dim_data.loc[dim_data['Success Rate (%)'].idxmax()]['Algorithm']} ({dim_data['Success Rate (%)'].max():.1f}%)")
            print()
    # 算法推荐
    print("\n3. 算法选择推荐:")
    print("-" * 80)
    recommendations = {
        "(100, 20) - 小样本低维": "BCD (Adaptive) 或 FISTA (Adaptive)",
        "(200, 50) - 中样本中维": "FISTA (Restart+Adaptive) 或 ADMM (rho=1)",
        "(500, 100) - 大样本中维": "SGD (Adaptive) 或 ADMM (rho=1)",
        "(200, 200) - 中样本高维": "FISTA (Restart+Adaptive) 或 ADMM (rho=0.5)"
    }
    for scenario, recommendation in recommendations.items():
        print(f"  {scenario}: {recommendation}")
    print("\n" + "=" * 80)
    print("详细数据已保存至: experiment_results/comprehensive_analysis_adaptive.csv")
    print("=" * 80)

# 生成报告
generate_comprehensive_report()

print("\n" + "=" * 60)
print("🎉 LASSO算法实验完成！")
print("📊 结果文件:")
print("   - convergence_adaptive.png: 收敛曲线对比图")
print("   - runtime_adaptive.png: 运行时间对比图")
print("   - comprehensive_analysis_adaptive.csv: 详细分析数据")
print("=" * 60)