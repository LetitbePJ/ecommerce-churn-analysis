"""
02_visualize.py — 可视化：月度趋势 / 客户价值帕累托 / RFM 分层 / 留存热力图
数据源：data/clean_orders.csv（与 SQL 结果同口径，可在任何机器复现）
输出：charts/ 下 4 张 PNG
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# ---------- 全局样式 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]   # 中文字体
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")
C_MAIN, C_ACCENT, C_WARN = "#2E86AB", "#F18F01", "#C73E1D"

df = pd.read_csv("data/clean_orders.csv",
                 parse_dates=["order_date"],
                 dtype={"customer_id": "int64"}).rename(
    columns={"Invoice": "invoice", "Customer ID": "customer_id"})

def save_fig(fig, name):
    fig.savefig(f"charts/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] charts/{name}")

print("=" * 50)
print("可视化开始，数据规模:", df.shape)

# ============================================================
# 图 1：月度销售趋势（销售额柱 + 订单数线，双轴）
# ============================================================
monthly = (df.groupby("order_month")
             .agg(revenue=("amount", "sum"), orders=("invoice", "nunique"))
             .reset_index())

fig, ax1 = plt.subplots(figsize=(12, 5.5))
x = range(len(monthly))
ax1.bar(x, monthly["revenue"] / 1e4, color=C_MAIN, alpha=0.85, label="销售额(万£)")
ax1.set_xticks(list(x))
ax1.set_xticklabels(monthly["order_month"], rotation=45, ha="right")
ax1.set_ylabel("销售额（万英镑）")
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.0f}"))

ax2 = ax1.twinx()
ax2.plot(x, monthly["orders"], color=C_ACCENT, marker="o", ms=4, lw=1.8, label="订单数")
ax2.set_ylabel("订单数")

# 标注旺季峰值与谷底
peak = monthly["revenue"].idxmax()
trough = monthly[monthly["order_month"].str.startswith(("2011",))]["revenue"].idxmin()
ax1.annotate(f"旺季峰值 {monthly.loc[peak,'order_month']}\n£{monthly.loc[peak,'revenue']/1e4:.0f}万",
             xy=(peak, monthly.loc[peak, "revenue"]/1e4), xytext=(peak-6, monthly["revenue"].max()/1e4 - 8),
             fontsize=9, color=C_MAIN, arrowprops=dict(arrowstyle="->", color=C_MAIN),
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
ax1.annotate(f"谷底 {monthly.loc[trough,'order_month']}",
             xy=(trough, monthly.loc[trough, "revenue"]/1e4),
             xytext=(trough-5, monthly["revenue"].min()/1e4+18),
             fontsize=9, color=C_WARN, arrowprops=dict(arrowstyle="->", color=C_WARN),
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

ax1.set_title("月度销售趋势：9-11月为旺季，11月销售额是2月的2.6倍", fontsize=13, pad=12)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
save_fig(fig, "01_monthly_trend.png")

# ============================================================
# 图 2：客户价值帕累托（Top 10% 客户贡献 64% 销售额）
# ============================================================
cust_rev = df.groupby("customer_id")["amount"].sum().sort_values(ascending=False)
cum_share = cust_rev.cumsum() / cust_rev.sum()
n = len(cust_rev)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(range(1, n + 1), cum_share * 100, color=C_MAIN, lw=2.5)
ax.fill_between(range(1, n + 1), cum_share * 100, color=C_MAIN, alpha=0.12)

# 参考线：经典二八法则
ax.axhline(80, ls="--", color="gray", alpha=0.5)
ax.axvline(n * 0.2, ls="--", color="gray", alpha=0.5)
ax.annotate("经典二八法则\n(20%客户→80%收入)", xy=(n*0.2, 80), xytext=(n*0.28, 62),
            fontsize=9, color="gray")

top10_share = cum_share.iloc[int(n * 0.1) - 1] * 100
ax.axvline(n * 0.1, ls=":", color=C_WARN, lw=1.5)
ax.annotate(f"Top 10% 客户\n贡献 {top10_share:.0f}% 销售额\n(比二八法则更陡)",
            xy=(n*0.1, top10_share), xytext=(n*0.16, 40),
            fontsize=10, color=C_WARN, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_WARN))

ax.set_xlabel("客户数（按消费额降序）")
ax.set_ylabel("累计销售额占比 %")
ax.set_title(f"客户价值帕累托曲线（n={n} 个客户）", fontsize=13, pad=12)
save_fig(fig, "02_customer_pareto.png")

# ============================================================
# 图 3：RFM 分层 —— 人数占比 vs 金额占比（复用 SQL 的分层规则）
# ============================================================
obs_end = df["order_date"].max()
rfm = (df.groupby("customer_id")
         .agg(recency=("order_date", lambda s: (obs_end - s.max()).days),
              frequency=("invoice", "nunique"),
              monetary=("amount", "sum"))
         .reset_index())
rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["m_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

def segment(row):
    if row.r_score >= 4 and row.f_score >= 4: return "重要价值客户\n(维护)"
    if row.r_score >= 4 and row.f_score < 4:  return "重要发展客户\n(提升频次)"
    if row.r_score < 4 and row.f_score >= 4:  return "重要挽留客户\n(唤醒)"
    if row.r_score <= 2 and row.f_score <= 2: return "流失客户\n(放弃/低成本触达)"
    return "一般客户"
rfm["segment"] = rfm.apply(segment, axis=1)

seg = (rfm.groupby("segment")
          .agg(人数=("customer_id", "count"), 金额=("monetary", "sum")))
seg["人数占比"] = seg["人数"] / seg["人数"].sum() * 100
seg["金额占比"] = seg["金额"] / seg["金额"].sum() * 100
seg = seg.sort_values("金额占比", ascending=False)

fig, ax = plt.subplots(figsize=(10.5, 5.5))
xp = np.arange(len(seg)); w = 0.38
b1 = ax.bar(xp - w/2, seg["人数占比"], w, color=C_MAIN, alpha=0.85, label="人数占比")
b2 = ax.bar(xp + w/2, seg["金额占比"], w, color=C_ACCENT, alpha=0.85, label="金额占比")
for bars in (b1, b2):
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=2)
ax.set_xticks(xp); ax.set_xticklabels(seg.index, fontsize=10)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title("RFM 分层：重要价值客户 27% 的人数贡献 70% 的金额", fontsize=13, pad=12)
ax.legend()
save_fig(fig, "03_rfm_segments.png")

# ============================================================
# 图 4：留存热力图（按首购月 cohort × 距首购第 N 月）
# ============================================================
first = df.groupby("customer_id")["order_month"].min().rename("cohort")
act = (df.assign(cohort=df["customer_id"].map(first))
         .groupby(["cohort", "order_month"])["customer_id"].nunique().reset_index()
         .rename(columns={"customer_id": "users"}))
cohort_start = pd.PeriodIndex(act["cohort"], freq="M").astype(int)
act_month = pd.PeriodIndex(act["order_month"], freq="M").astype(int)
act["offset"] = act_month - cohort_start
base = act[act["offset"] == 0].set_index("cohort")["users"]
act["retention"] = act["users"] / act["cohort"].map(base) * 100

heat = (act[act["offset"] <= 12]
        .pivot(index="cohort", columns="offset", values="retention"))
heat.columns = [f"第{c}月" for c in heat.columns]

fig, ax = plt.subplots(figsize=(13, 6.5))
sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=60,
            cbar_kws={"label": "留存率 %"}, linewidths=0.4, ax=ax)
ax.set_xlabel("距首次购买")
ax.set_ylabel("首购月份（cohort）")
ax.set_title("留存热力图：次月留存约 19%，留存下来的客户会持续回访", fontsize=13, pad=12)
save_fig(fig, "04_retention_heatmap.png")

# ---------- 保存 RFM 明细表供报告使用 ----------
rfm.to_csv("data/rfm_customers.csv", index=False)
print("  [saved] data/rfm_customers.csv")
print("=" * 50)
print("可视化完成：charts/ 下 4 张图")
