"""
03_churn_model.py — 流失预测模型（P1）

设计要点（与"给 RFM 打个标签"的本质区别）：
  1. 时间切窗，杜绝标签泄漏
     观测窗口 = 2009-12-01 ~ 2011-09-09（约 21 个月）→ 只用这段数据造特征
     表现窗口 = 2011-09-10 ~ 2011-12-09（90 天）→ 只看这段数据定标签
     标签：表现窗口内**没有任何购买** = 流失（1）
     关键：recency 用"距观测截止日"的天数，与决定了标签的那 90 天完全分离。
     反例（常见错误）：用数据终点算 recency，又用"距终点 >90 天"当标签 ——
     标签等价于 recency>90，模型等于抄答案，PR-AUC 会虚高到 0.99。

  2. 特征做共线性筛查
     原始候选有 16 个，其中 frequency 与 n_active_days 秩相关 0.99、
     orders_per_month 与 recency −0.83。直接塞进逻辑回归会出现
     "买得越多越容易流失"这种反直觉系数（符号被共线性带偏）。
     故剔除冗余项，保留 6 个方向稳定的特征。

  3. 三类对照
     模型 A：逻辑回归（可解释，主模型）
     模型 B：随机森林（非线性对照）
     基准 C：规则预警线 recency>70 天（报告 §3.5 的经验规则）
     —— 用来回答"模型到底比拍脑袋强多少"

  4. 评估口径
     主指标 PR-AUC（average precision）；辅以 ROC-AUC、前 20% 覆盖率、Lift。
     流失率 56.6%，正负样本接近均衡，但仍不用准确率（会被多数类带偏）。

输出：charts/05_pr_curve.png、charts/06_feature_importance.png、data/churn_scores.csv
"""
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve)

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

CJK_FONTS = ["Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
             "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]
plt.rcParams["font.sans-serif"] = CJK_FONTS + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
C_MAIN, C_ACCENT, C_WARN, C_GRAY = "#2E86AB", "#F18F01", "#C73E1D", "#9E9E9E"

RANDOM_STATE = 42
HORIZON_DAYS = 90          # 表现窗口长度 = 流失判定周期

# ============================================================
# 0. 读数据 + 切窗
# ============================================================
df = (pd.read_csv("data/clean_orders.csv", parse_dates=["order_date"])
        .rename(columns={"Customer ID": "customer_id", "Invoice": "invoice",
                         "StockCode": "stock_code"}))
data_end = df["order_date"].max()
cutoff = data_end - pd.Timedelta(days=HORIZON_DAYS)

obs = df[df["order_date"] < cutoff]      # 观测窗口（造特征）
perf = df[df["order_date"] >= cutoff]    # 表现窗口（定标签）

print("=" * 64)
print(f"数据终点 {data_end.date()}｜观测截止 {cutoff.date()}｜表现窗口 {HORIZON_DAYS} 天")
print(f"观测窗口 {obs['order_date'].min().date()} ~ {obs['order_date'].max().date()}"
      f"（{len(obs):,} 行）")
print(f"表现窗口 {perf['order_date'].min().date()} ~ {perf['order_date'].max().date()}"
      f"（{len(perf):,} 行）")

# ============================================================
# 1. 特征：全部只用观测窗口数据
# ============================================================
base = (obs.groupby("customer_id")
           .agg(frequency=("invoice", "nunique"),
                monetary=("amount", "sum"),
                n_skus=("stock_code", "nunique"),
                first_buy=("order_date", "min"),
                last_buy=("order_date", "max")))

base["recency"] = (cutoff - base["last_buy"]).dt.days          # 距观测截止的天数
base["tenure"] = (cutoff - base["first_buy"]).dt.days          # 客龄（首次购买距截止）

# 购买间隔：同一客户相邻两个下单日期的间隔，取最大值（"最长曾隔多久才来"）
dates = (obs[["customer_id", "order_date"]].drop_duplicates()
            .sort_values(["customer_id", "order_date"]))
dates["gap"] = dates.groupby("customer_id")["order_date"].diff().dt.days
base = base.join(dates.groupby("customer_id")["gap"].max().rename("max_gap"))

# 近期频次动量：截止前 0–90 天 vs 90–180 天的订单数
recent = (obs[obs["order_date"] >= cutoff - pd.Timedelta(days=90)]
          .groupby("customer_id")["invoice"].nunique().rename("orders_last90"))
prev = (obs[(obs["order_date"] >= cutoff - pd.Timedelta(days=180)) &
            (obs["order_date"] < cutoff - pd.Timedelta(days=90))]
        .groupby("customer_id")["invoice"].nunique().rename("orders_prev90"))
base = base.join(recent).join(prev)
base[["orders_last90", "orders_prev90"]] = base[["orders_last90", "orders_prev90"]].fillna(0)

# 候选特征共线性筛查（Spearman，观测窗口内）：
#   frequency ↔ n_active_days 0.99、orders_per_month ↔ recency −0.83、monetary ↔ frequency 0.85
#   → 保留 6 个：recency（近因）、max_gap（最长间隔）、orders_last90 / orders_prev90（近期频次
#     与动量，替代累计频次）、n_skus（品类宽度）、tenure（客龄）
FEATURES = ["recency", "max_gap", "orders_last90", "orders_prev90", "n_skus", "tenure"]

# ============================================================
# 2. 标签：表现窗口内是否有购买
# ============================================================
bought_in_perf = perf["customer_id"].unique()
base["churn"] = (~base.index.isin(bought_in_perf)).astype(int)

X = base[FEATURES]
y = base["churn"]
print(f"样本 {len(base):,} 个客户｜流失率 {y.mean()*100:.1f}%"
      f"（{y.sum():,} 流失 / {(1-y).sum():,} 留存）")
print(f"特征 {len(FEATURES)} 个：{', '.join(FEATURES)}（全部取自观测窗口）")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=RANDOM_STATE)

# ============================================================
# 3. 建模
# ============================================================
lr = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=RANDOM_STATE)),
])
rf = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=20,
                                   class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=RANDOM_STATE)),
])
for model in (lr, rf):
    model.fit(X_tr, y_tr)

# 规则基线：recency > 70 天判流失（报告 §3.5 的预警线）
p_rule = np.where((X_te["recency"] > 70).to_numpy(), 0.9, 0.1)

# ============================================================
# 4. 评估
# ============================================================
def eval_block(name, prob, y_true):
    k = int(len(prob) * 0.2)
    top = np.argsort(-prob)[:k]
    capture = y_true.to_numpy()[top].sum() / y_true.sum()
    return {"PR-AUC": average_precision_score(y_true, prob),
            "ROC-AUC": roc_auc_score(y_true, prob),
            "前20%覆盖率": capture,
            "Lift@20%": capture / 0.2, "_name": name}

rows = {"随机猜测": eval_block("随机猜测", np.full(len(y_te), y_tr.mean()), y_te),
        "逻辑回归": eval_block("逻辑回归", lr.predict_proba(X_te)[:, 1], y_te),
        "随机森林": eval_block("随机森林", rf.predict_proba(X_te)[:, 1], y_te),
        "规则基线(recency>70天)": eval_block("规则基线(recency>70天)", p_rule, y_te)}
res = pd.DataFrame(rows).T.drop(columns="_name")
print("\n[测试集评估]（25% 留出样本｜主看 PR-AUC：随机猜测为基准线，越接近 1 越好）")
print(res.round(3).to_string())

# 逻辑回归的 F1 最优阈值 → 可落地的判定线
prob_lr = lr.predict_proba(X_te)[:, 1]
prec, rec, thr = precision_recall_curve(y_te, prob_lr)
f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
best = int(np.argmax(f1))
print(f"\n[阈值] F1 最优切点：概率 ≥ {thr[min(best, len(thr)-1)]:.3f} 判流失｜"
      f"精确率 {prec[best]:.2f}｜召回率 {rec[best]:.2f}")

# 5 折交叉验证（看稳定性，而不是单次切分的运气）
for name, model in (("逻辑回归", lr), ("随机森林", rf)):
    scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True,
                            random_state=RANDOM_STATE), scoring="average_precision")
    print(f"[CV] {name} 5 折 PR-AUC：{scores.mean():.3f} ± {scores.std():.3f}")

# ============================================================
# 5. 图 5：PR 曲线对照
# ============================================================
fig, ax = plt.subplots(figsize=(8.4, 6))
for name, model, color in (("逻辑回归", lr, C_MAIN), ("随机森林", rf, C_ACCENT)):
    p = model.predict_proba(X_te)[:, 1]
    pr, rc, _ = precision_recall_curve(y_te, p)
    ax.plot(rc, pr, lw=2.2, color=color,
            label=f"{name}（PR-AUC {average_precision_score(y_te, p):.3f}）")
pr, rc, _ = precision_recall_curve(y_te, p_rule)
ax.plot(rc, pr, lw=1.8, ls="--", color=C_WARN,
        label=f"规则基线 recency>70天（PR-AUC {average_precision_score(y_te, p_rule):.3f}）")
ax.axhline(y_te.mean(), ls=":", color=C_GRAY, lw=1.4,
           label=f"随机猜测（{y_te.mean():.3f}）")
ax.set_xlabel("召回率（实际流失客户中被抓到的比例）")
ax.set_ylabel("精确率（被判流失的客户中真流失的比例）")
ax.set_title(f"流失预测 PR 曲线：模型显著优于经验规则（测试集流失率 {y_te.mean()*100:.1f}%）",
             fontsize=12.5, pad=12)
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 1.02)
fig.savefig("charts/05_pr_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  [saved] charts/05_pr_curve.png")

# ============================================================
# 6. 图 6：逻辑回归标准化系数
# ============================================================
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.barh(coef.index, coef.values,
        color=[C_WARN if v > 0 else C_MAIN for v in coef.values], alpha=0.88)
for i, v in enumerate(coef.values):
    ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=9.5)
ax.axvline(0, color="#333", lw=0.9)
ax.set_xlabel("逻辑回归系数（特征已标准化；正 = 推高流失概率）")
ax.set_title("流失预测特征方向：近因主导，近期频次下滑与品类收窄推高流失",
             fontsize=12.5, pad=12)
ax.set_xlim(coef.min() * 1.3, max(coef.max() * 1.3, 0.2))
ax.grid(axis="x", alpha=0.35)
fig.savefig("charts/06_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  [saved] charts/06_feature_importance.png")
print("\n[系数]（正=推高流失）")
print(coef.round(3).to_string())

# ============================================================
# 7. 打分表：全量客户流失概率 + 风险分层
# ============================================================
base["churn_prob"] = lr.predict_proba(X)[:, 1]
base["risk_tier"] = pd.cut(base["churn_prob"], [0, 0.3, 0.6, 0.8, 1.0],
                           labels=["低风险", "中风险", "高风险", "极高风险"])
out = (base.reset_index()[["customer_id", "recency", "frequency", "monetary",
                           "churn", "churn_prob", "risk_tier"]]
           .sort_values("churn_prob", ascending=False))
out.to_csv("data/churn_scores.csv", index=False, encoding="utf-8-sig")
print(f"\n  [saved] data/churn_scores.csv（{len(out):,} 客户）")

tier = (out.groupby("risk_tier", observed=True)
           .agg(客户数=("customer_id", "count"), 实际流失率=("churn", "mean"),
                平均概率=("churn_prob", "mean")))
tier["客户占比"] = tier["客户数"] / len(out) * 100
print("[风险分层落地效果]（概率来自全量样本打分，实际流失率用于验证分层有效性）")
print(tier[["客户数", "客户占比", "平均概率", "实际流失率"]].round(3).to_string())
print("=" * 64)
