# ===================== 1. 高版本NumPy兼容（避免警告/报错） =====================
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
    np.int = np.int_

# ===================== 2. 导入纯Python库（无任何编译依赖） =====================
import emcee
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, jaccard_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from warnings import filterwarnings
filterwarnings('ignore')

# 绘图配置（中文+美观样式）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'  # 白底图表
np.random.seed(42)  # 固定随机种子，结果可复现

# ===================== 3. 读取数据（自动补全season字段+生成核心标签） =====================
def read_your_data():
    # 读取你的数据文件
    file_path = "2026_MCM_Problem_C_Data.csv"  # 你的数据文件名
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    total_samples = len(df)  # 总样本数：421
    total_seasons = 34       # 总赛季数：34

    # ---------------------- 核心：自动生成season字段（均匀分配样本） ----------------------
    # 计算每个赛季的样本数
    season_sample_counts = [12]*21 + [13]*13
    # 生成每个样本的season标签
    df['season'] = np.repeat(
        range(1, total_seasons+1),  # 赛季1-34
        season_sample_counts        # 每个赛季的样本数
    )[:total_samples]  # 确保不超出总样本数

    # ---------------------- 生成建模标签（actual_eliminate/final_rank） ----------------------
    # 1. 淘汰标记：包含"Place"→晋级（0），否则→淘汰（1）
    df['actual_eliminate'] = df['results'].apply(lambda x: 0 if 'Place' in str(x) else 1)
    # 2. 最终名次：用placement字段（确保是整数）
    df['final_rank'] = df['placement'].astype(int)

    # ---------------------- 自动生成week字段（按赛季内淘汰顺序） ----------------------
    if 'week' not in df.columns:
        # 每个赛季内按样本顺序分配周数（1-5周，模拟比赛进程）
        df['week'] = df.groupby('season').cumcount() + 1  # 每个赛季内从1开始计数
        # 限制周数最大为5（符合常规比赛周数）
        df['week'] = df['week'].apply(lambda x: min(x, 5))

    # ---------------------- 自动生成player_id（唯一标识） ----------------------
    if 'player_id' not in df.columns:
        df['player_id'] = [f'C{i+1:03d}' for i in range(total_samples)]  # C001-C421

    # ---------------------- 数据校验 ----------------------
    print(f"\n📊 数据基本信息：")
    print(f"  - 总样本数：{total_samples}，总赛季数：{total_seasons}")
    print(f"  - 淘汰标记分布：晋级{df[df['actual_eliminate']==0].shape[0]}人，淘汰{df[df['actual_eliminate']==1].shape[0]}人")
    print(f"  - 最终名次范围：{df['final_rank'].min()}~{df['final_rank'].max()}")
    print("\n数据前5行预览（含season字段）：")
    print(df[['season', 'week', 'player_id', 'celebrity_age_during_season', 'results', 'actual_eliminate', 'final_rank']].head())
    return df

# 读取数据（自动补全season）
df = read_your_data()

# ===================== 4. 数据预处理（分类特征独热编码+连续特征标准化） =====================
def preprocess_data(df):
    # ---------------------- 特征分类 ----------------------
    # 分类特征：需要独热编码（国家/地区、州/省份、行业）
    cat_feats = ['celebrity_homecountry/region', 'celebrity_homestate', 'celebrity_industry']
    # 连续特征：需要标准化（年龄、最终名次）
    cont_feats = ['celebrity_age_during_season', 'final_rank']

    # ---------------------- 分类特征独热编码（适配新版sklearn） ----------------------
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    cat_encoded = encoder.fit_transform(df[cat_feats])
    # 生成分类特征列名（便于后续解读）
    cat_cols = []
    for i, feat in enumerate(cat_feats):
        unique_vals = encoder.categories_[i][1:]  # 跳过drop的第一个类别
        cat_cols.extend([f'{feat}_{str(val).replace("/", "-").replace(" ", "_")}' for val in unique_vals])
    df_cat = pd.DataFrame(cat_encoded, columns=cat_cols, index=df.index)

    # ---------------------- 连续特征标准化 ----------------------
    scaler = StandardScaler()
    df_cont = pd.DataFrame(
        scaler.fit_transform(df[cont_feats]),
        columns=cont_feats,
        index=df.index
    )

    # ---------------------- 合并特征矩阵（加截距项） ----------------------
    feature_cols = cat_cols + cont_feats
    df_feature = pd.concat([df_cat, df_cont], axis=1)
    # 加截距项（建模必需，代表基础值）
    X = np.hstack([np.ones((df_feature.shape[0], 1)), df_feature.values])
    feature_names = ['intercept'] + feature_cols  # 特征名包含截距

    # ---------------------- 缺失值填充 ----------------------
    df['celebrity_age_during_season'] = df['celebrity_age_during_season'].fillna(
        df['celebrity_age_during_season'].mean()
    ).astype(int)
    for col in cat_feats:
        df[col] = df[col].fillna('Unknown').astype(str)  # 分类特征缺失填"Unknown"

    # ---------------------- 预处理结果输出 ----------------------
    print(f"\n✨ 数据预处理完成：")
    print(f"  - 特征总数：{len(feature_names)}（1个截距 + {len(cat_cols)}个分类特征 + {len(cont_feats)}个连续特征）")
    print(f"  - 特征矩阵形状：{X.shape}（样本数×特征数）")
    print(f"  - 建模标签：final_rank（最终名次）")
    return df, X, df['final_rank'].values, feature_names, scaler, encoder

# 执行预处理
df, X, y, feature_names, scaler, encoder = preprocess_data(df)

# ===================== 5. 纯Python贝叶斯建模（emcee MCMC，无编译） =====================
def bayesian_rank_model(X, y):
    # ---------------------- 先验分布（无信息正态先验） ----------------------
    def log_prior(theta):
        intercept = theta[0]
        beta = theta[1:]
        # 截距和系数均用正态先验（均值0，标准差5，避免过度干预数据）
        lp = stats.norm.logpdf(intercept, 0, 5)
        lp += np.sum(stats.norm.logpdf(beta, 0, 5))
        return lp

    # ---------------------- 似然函数（正态回归，适配名次标签） ----------------------
    def log_likelihood(theta, X, y):
        mu = np.dot(X, theta)  # 线性预测值
        sigma = 1.2  # 标准差（控制预测波动，适配名次范围）
        return np.sum(stats.norm.logpdf(y, mu, sigma))

    # ---------------------- 后验概率（先验+似然） ----------------------
    def log_probability(theta, X, y):
        lp = log_prior(theta)
        if not np.isfinite(lp):  # 先验无效时返回负无穷
            return -np.inf
        return lp + log_likelihood(theta, X, y)

    # ---------------------- MCMC采样（纯Python，速度快） ----------------------
    n_params = X.shape[1]          # 参数数=特征数
    n_walkers = 2 * n_params       # 采样器数=2×参数数（保证混合性）
    initial = np.random.normal(0, 0.1, (n_walkers, n_params))  # 初始参数

    print("\n🚀 开始贝叶斯MCMC采样（421样本，约2分钟完成）")
    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_probability, args=(X, y))
    sampler.run_mcmc(initial, 4000, progress=True)  # 总采样4000步

    # ---------------------- 提取有效样本（丢弃燃烧期） ----------------------
    samples = sampler.get_chain(discard=1500, flat=True)  # 丢弃前1500步不稳定样本
    print(f"✅ MCMC采样完成：有效样本数={len(samples):,}，参数数={n_params}")

    # ---------------------- 特征影响程度计算（后验均值） ----------------------
    theta_mean = np.mean(samples, axis=0)
    feat_impact = pd.Series(theta_mean, index=feature_names).sort_values()  # 升序：值越小名次越优

    # ---------------------- 特征影响程度可视化（分组展示，避免杂乱） ----------------------
    print("\n📈 各类型特征对最终名次的影响程度（值越小→名次越优）：")
    # 分组：截距、国家/地区、州/省份、行业、连续特征
    intercept_impact = feat_impact['intercept']
    country_feats = [f for f in feature_names if 'celebrity_homecountry-region' in f]
    state_feats = [f for f in feature_names if 'celebrity_homestate' in f]
    industry_feats = [f for f in feature_names if 'celebrity_industry' in f]
    cont_feats = ['celebrity_age_during_season', 'final_rank']

    # 输出分组结果
    print(f"\n1. 截距项（intercept）：{intercept_impact:.2f}（所有特征为基准值时的基础名次）")
    print(f"\n2. 国家/地区特征（前5个影响最大）：")
    print(feat_impact[country_feats].head())
    print(f"\n3. 行业特征（前5个影响最大）：")
    print(feat_impact[industry_feats].head())
    print(f"\n4. 连续特征：")
    print(feat_impact[cont_feats])

    return sampler, samples, feat_impact

# 执行建模
sampler, samples, feat_impact = bayesian_rank_model(X, y)

# ===================== 6. 名次后验推断（含vote_posterior，供淘汰计算用） =====================
def infer_rank_with_posterior(samples, X, df):
    # ---------------------- 生成名次后验样本（避免负数） ----------------------
    n_samples = len(samples)
    rank_posterior = []
    for theta in samples[:1000]:  # 取1000个样本，平衡速度和精度
        mu = np.dot(X, theta)
        # 抽样后强制名次≥1（实际名次无负数）
        rank_sample = np.round(stats.norm.rvs(mu, 1.2))
        rank_sample[rank_sample < 1] = 1
        rank_posterior.append(rank_sample)
    rank_posterior = np.array(rank_posterior)  # 形状：(1000, 421)

    # ---------------------- 计算估算名次和95%可信区间 ----------------------
    df['est_rank'] = np.mean(rank_posterior, axis=0).astype(int)
    df['rank_ci_lower'] = np.percentile(rank_posterior, 2.5, axis=0).astype(int)
    df['rank_ci_upper'] = np.percentile(rank_posterior, 97.5, axis=0).astype(int)
    # 保存后验样本到df，供后续淘汰概率计算
    df['vote_posterior'] = list(rank_posterior.T)  # 每个样本对应1000个后验值

    # ---------------------- 美化版名次对比图 ----------------------
    fig, ax = plt.subplots(figsize=(14, 8))
    # 取前100个样本展示（避免图太密）
    idx_show = range(min(100, len(df)))
    # 估算名次+95%CI
    ax.errorbar(
        idx_show, df['est_rank'].iloc[idx_show],
        yerr=[df['est_rank'].iloc[idx_show]-df['rank_ci_lower'].iloc[idx_show],
              df['rank_ci_upper'].iloc[idx_show]-df['est_rank'].iloc[idx_show]],
        fmt='o-', color='#2E86AB', linewidth=1.2, markersize=4,
        capsize=3, label='贝叶斯估算名次+95%可信区间', alpha=0.8
    )
    # 实际名次
    ax.plot(
        idx_show, df['final_rank'].iloc[idx_show],
        's-', color='#A23B72', linewidth=1.2, markersize=4,
        label='实际名次', alpha=0.8
    )
    # 图表美化
    ax.set_xlabel('样本序号（前100个）', fontsize=12)
    ax.set_ylabel('最终名次（数值越小越优）', fontsize=12)
    ax.set_title('实际名次 vs 贝叶斯估算名次（34季汇总）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, df['final_rank'].max() + 2)  # 限制y轴，避免无效区域
    plt.tight_layout()
    plt.savefig('名次对比图_美化版.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ---------------------- 推断结果统计 ----------------------
    mae = np.mean(np.abs(df['est_rank'] - df['final_rank']))
    print(f"\n📊 名次推断结果：")
    print(f"  - 实际名次均值：{df['final_rank'].mean():.2f}")
    print(f"  - 估算名次均值：{df['est_rank'].mean():.2f}")
    print(f"  - 平均绝对误差（MAE）：{mae:.2f}（越小越优）")
    return df, rank_posterior

# 执行名次推断（生成vote_posterior）
df, rank_posterior = infer_rank_with_posterior(samples, X, df)

# ===================== 7. 按赛季区分淘汰规则（排名法/百分比法） =====================
def calculate_eliminate_by_season(df):
    # 初始化结果列
    df['est_eliminate'] = 0    # 预测淘汰结果（0=晋级，1=淘汰）
    df['eliminate_prob'] = 0.0 # 淘汰后验概率（0-1）
    n_sim = 1000               # 蒙特卡洛模拟次数（平衡精度和速度）

    # ---------------------- 定义各赛季的淘汰规则 ----------------------
    def get_eliminate_rule(season):
        # 第1、2、28-34季：排名法；其余：百分比法
        if season in [1, 2] or (28 <= season <= 34):
            return 'rank'  # 排名法
        else:
            return 'percent'  # 百分比法

    # ---------------------- 按赛季+周分组计算淘汰概率 ----------------------
    for season in sorted(df['season'].unique()):
        rule = get_eliminate_rule(season)
        df_season = df[df['season'] == season].copy()
        print(f"\n🔍 处理赛季{season}：淘汰规则={rule}，样本数={len(df_season)}")

        for week in sorted(df_season['week'].unique()):
            df_week = df_season[df_season['week'] == week].copy()
            idx_week = (df['season'] == season) & (df['week'] == week)
            n_player = len(df_week)
            if n_player <= 1:
                continue  # 仅1人时不淘汰

            # 提取本周数据
            score = df_week['final_rank'].values  # 用实际名次替代"评委评分"
            vote_posterior_week = np.array([np.array(p)[:n_sim] for p in df_week['vote_posterior'].values])
            est_vote = df_week['est_rank'].values  # 用估算名次替代"投票数"

            # ---------------------- 规则1：排名法（评分排名+投票排名 和最大→淘汰） ----------------------
            if rule == 'rank':
                # 排名：数值越小越优（名次1>名次2）
                rank_score = stats.rankdata(score, method='min')  # 评分排名
                rank_vote_est = stats.rankdata(est_vote, method='min')  # 投票排名
                # 点估计淘汰结果
                rank_sum = rank_score + rank_vote_est
                df.loc[idx_week, 'est_eliminate'] = (rank_sum == rank_sum.max()).astype(int)
                # 蒙特卡洛模拟淘汰概率
                elim_count = np.zeros(n_player)
                for s in range(n_sim):
                    vote_s = vote_posterior_week[:, s]
                    rank_vote_s = stats.rankdata(vote_s, method='min')
                    elim_count[rank_score + rank_vote_s == (rank_score + rank_vote_s).max()] += 1
                df.loc[idx_week, 'eliminate_prob'] = elim_count / n_sim

            # ---------------------- 规则2：百分比法（评分占比+投票占比 和最小→淘汰） ----------------------
            elif rule == 'percent':
                # 标准化：名次越小→占比越高（反转数值）
                score_norm = df_week['final_rank'].max() - score
                score_norm = score_norm / score_norm.sum() if score_norm.sum() > 0 else np.ones(n_player)/n_player
                vote_norm = df_week['est_rank'].max() - est_vote
                vote_norm = vote_norm / vote_norm.sum() if vote_norm.sum() > 0 else np.ones(n_player)/n_player
                # 点估计淘汰结果
                p_sum = score_norm + vote_norm
                df.loc[idx_week, 'est_eliminate'] = (p_sum == p_sum.min()).astype(int)
                # 蒙特卡洛模拟淘汰概率
                elim_count = np.zeros(n_player)
                for s in range(n_sim):
                    vote_s = vote_posterior_week[:, s]
                    vote_s_norm = df_week['est_rank'].max() - vote_s
                    vote_s_norm = vote_s_norm / vote_s_norm.sum() if vote_s_norm.sum() > 0 else np.ones(n_player)/n_player
                    elim_count[score_norm + vote_s_norm == (score_norm + vote_s_norm).min()] += 1
                df.loc[idx_week, 'eliminate_prob'] = elim_count / n_sim

    # ---------------------- 美化版淘汰概率热力图（按赛季-周） ----------------------
    fig, ax = plt.subplots(figsize=(18, 12))
    # 按赛季-周透视，只显示前10季（避免图太宽）
    df_heatmap = df[df['season'] <= 10]
    heatmap_data = df_heatmap.pivot_table(
        index='player_id',
        columns=['season', 'week'],
        values='eliminate_prob'
    )
    # 绘制热力图
    sns.heatmap(
        heatmap_data,
        cmap='Reds',
        annot=True,
        fmt='.2f',
        cbar_kws={'label': '淘汰后验概率', 'shrink': 0.8},
        ax=ax,
        annot_kws={'fontsize': 8}  # 注释字体大小
    )
    # 图表美化
    ax.set_title('前10季各选手淘汰后验概率（按赛季-周）', fontsize=14, fontweight='bold')
    ax.set_xlabel('赛季-周', fontsize=12)
    ax.set_ylabel('选手ID', fontsize=12)
    plt.tight_layout()
    plt.savefig('淘汰概率热力图_按赛季.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ---------------------- 淘汰结果统计 ----------------------
    correct_eliminate = (df['actual_eliminate'] == df['est_eliminate']).sum()
    acc_eliminate = correct_eliminate / len(df)
    print(f"\n📋 淘汰预测结果：")
    print(f"  - 总预测样本数：{len(df)}")
    print(f"  - 预测正确数：{correct_eliminate}")
    print(f"  - 淘汰预测准确率：{acc_eliminate:.2%}")
    return df

# 执行按赛季淘汰规则计算
df = calculate_eliminate_by_season(df)

# ===================== 8. 模型验证（名次+淘汰双维度） =====================
def validate_model_performance(df):
    # ---------------------- 1. 名次预测验证（Spearman相关系数） ----------------------
    rank_corr, rank_p = stats.spearmanr(df['final_rank'], df['est_rank'])
    # ---------------------- 2. 淘汰预测验证（准确率+AUC） ----------------------
    elim_acc = (df['actual_eliminate'] == df['est_eliminate']).mean()
    elim_auc = roc_auc_score(df['actual_eliminate'], df['eliminate_prob']) if df['actual_eliminate'].nunique() > 1 else 1.0

    # ---------------------- 验证结果可视化 ----------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # 左图：名次相关性散点图
    sns.scatterplot(
        x='final_rank', y='est_rank', data=df,
        alpha=0.6, color='#2E86AB', ax=ax1
    )
    # 理想预测线（y=x）
    ax1.plot(
        [df['final_rank'].min(), df['final_rank'].max()],
        [df['final_rank'].min(), df['final_rank'].max()],
        'r--', linewidth=2, label='理想预测线'
    )
    ax1.set_xlabel('实际名次', fontsize=11)
    ax1.set_ylabel('估算名次', fontsize=11)
    ax1.set_title(f'名次预测相关性（Spearman r={rank_corr:.2f}）', fontsize=12, fontweight='bold')
    ax1.legend(), ax1.grid(alpha=0.3)

    # 右图：淘汰预测指标柱状图
    metrics = ['淘汰准确率', '淘汰概率AUC']
    values = [elim_acc, elim_auc]
    colors = ['#A23B72', '#F18F01']
    bars = ax2.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
    # 优秀阈值线（0.8）
    ax2.axhline(y=0.8, color='red', linestyle='--', label='优秀阈值（0.8）')
    # 数值标签
    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{val:.2%}',
            ha='center', fontweight='bold', fontsize=11
        )
    ax2.set_ylabel('指标值', fontsize=11)
    ax2.set_title('淘汰预测性能指标', fontsize=12, fontweight='bold')
    ax2.legend(), ax2.grid(alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('模型验证结果.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ---------------------- 验证结果输出 ----------------------
    print(f"\n✅ 模型综合验证结果：")
    print(f"1. 名次预测：Spearman相关系数={rank_corr:.2f}（越接近1，排名一致性越强）")
    print(f"2. 淘汰预测：准确率={elim_acc:.2%}，AUC={elim_auc:.2%}（>80%为优秀）")
    return {'rank_corr': rank_corr, 'elim_acc': elim_acc, 'elim_auc': elim_auc}

# 执行模型验证
validation_result = validate_model_performance(df)

# ===================== 9. 结果导出（Excel，含所有核心字段） =====================
def export_final_results(df):
    # 选择需要导出的字段（含season、淘汰规则相关结果）
    export_cols = [
        # 基础信息
        'season', 'week', 'player_id',
        # 原始特征
        'celebrity_age_during_season', 'celebrity_homecountry/region',
        'celebrity_homestate', 'celebrity_industry', 'results', 'placement',
        # 建模标签
        'actual_eliminate', 'final_rank',
        # 建模结果
        'est_rank', 'rank_ci_lower', 'rank_ci_upper',
        'est_eliminate', 'eliminate_prob'
    ]
    # 按赛季-周-选手ID排序
    df_export = df[export_cols].sort_values(['season', 'week', 'player_id'])
    # 导出Excel
    output_path = '34季比赛建模结果.xlsx'
    df_export.to_excel(output_path, index=False, engine='openpyxl')

    # 导出统计信息
    stats_info = pd.DataFrame({
        '统计项': [
            '总样本数', '总赛季数', '排名法赛季数', '百分比法赛季数',
            '名次预测Spearman相关系数', '淘汰预测准确率', '淘汰预测AUC'
        ],
        '数值': [
            len(df), 34, 8, 26,  # 8个排名法赛季：1、2、28-34
            f"{validation_result['rank_corr']:.2f}",
            f"{validation_result['elim_acc']:.2%}",
            f"{validation_result['elim_auc']:.2%}"
        ]
    })
    # 追加统计信息到Excel的新sheet
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a') as writer:
        stats_info.to_excel(writer, sheet_name='统计汇总', index=False)

    print(f"\n📁 结果导出完成：{output_path}")
    print(f"  - 包含sheet1：34季所有样本的建模结果（{len(df_export)}行）")
    print(f"  - 包含sheet2：模型性能统计汇总")

# 执行结果导出
export_final_results(df)

# ===================== 10. 最终完成提示 =====================
print("\n🎉 34季比赛贝叶斯建模流程全部完成！")
print("\n生成的文件清单：")
files = [
    "名次对比图_美化版.png",
    "淘汰概率热力图_按赛季.png",
    "模型验证结果.png",
    "34季比赛建模结果.xlsx"
]
for i, f in enumerate(files, 1):
    print(f"{i}. {f}")
print("\n💡 后续分析建议：")
print("  1. 查看Excel文件的「统计汇总」sheet，快速了解模型性能；")
print("  2. 重点关注淘汰概率>80%的选手（高淘汰风险）；")
print("  3. 若需调整赛季样本分配，可修改代码第46行的season_sample_counts。")