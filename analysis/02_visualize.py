"""
02_visualize.py — 可视化：月度趋势 / 客户价值帕累托 / RFM 分层 / 留存热力图
数据源：data/clean_orders.csv（与 SQL 结果同口径，可在任何机器复现）
输出：charts/ 下 4 张 PNG
"""
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# 压掉其他平台字体缺失的提示（Windows 下会用列表里第一个可用的微软雅黑）
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ---------- 全局样式 ----------
# 中文字体按平台依次回退：Windows → macOS → Linux，找不到任一可用字体时退回英文标签也不报错
CJK_FONTS = ["Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
             "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]
plt.rcParams["font.sans-serif"] = CJK_FONTS + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font=CJK_FONTS)
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

# 识别不完整月（最后一个月数据截止到当月某天），单独灰化，不参与趋势判断
last_month = monthly["order_month"].iloc[-1]
last_days = int(df.loc[df["order_month"] == last_month, "order_date"].dt.day.max())
is_partial = last_days < 25
bar_colors = [C_MAIN] * len(monthly)
if is_partial:
    bar_colors[-1] = "#BFBFBF"

ax1.bar(x, monthly["revenue"] / 1e4, color=bar_colors, alpha=0.85, label="销售额(万£)")
ax1.set_xticks(list(x))
ax1.set_xticklabels(monthly["order_month"], rotation=45, ha="right")
ax1.set_ylabel("销售额（万英镑）")
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.0f}"))

ax2 = ax1.twinx()
ax2.plot(x, monthly["orders"], color=C_ACCENT, marker="o", ms=4, lw=1.8, label="订单数")
ax2.set_ylabel("订单数")

# 标注旺季峰值与谷底（均在完整月内取全局极值，不做年份筛选）
peak = monthly["revenue"].idxmax()
trough = monthly["revenue"].idxmin()
peak_v = monthly.loc[peak, "revenue"] / 1e4
trough_v = monthly.loc[trough, "revenue"] / 1e4
ax1.annotate(f"旺季峰值 {monthly.loc[peak,'order_month']}\n£{peak_v:.0f}万",
             xy=(peak, peak_v), xytext=(peak-6, peak_v - 8),
             fontsize=9, color=C_MAIN, arrowprops=dict(arrowstyle="->", color=C_MAIN),
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
ax1.annotate(f"谷底 {monthly.loc[trough,'order_month']}",
             xy=(trough, trough_v), xytext=(trough - 1, 24),
             fontsize=9, color=C_WARN, arrowprops=dict(arrowstyle="->", color=C_WARN),
             bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))
if is_partial:
    fig.text(0.995, 0.97,
             f"注：{last_month} 仅 {last_days} 天（不完整月），灰柱；\n"
             f"其销售额与订单数骤降为月末截断，不作趋势判断",
             ha="right", va="top", fontsize=9, color="#7F7F7F")

ax1.set_title(f"月度销售趋势：旺季峰值 {monthly.loc[peak,'order_month']} 为谷底 "
              f"{monthly.loc[trough,'order_month']} 的 {peak_v/trough_v:.1f} 倍", fontsize=13, pad=12)
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
# 图 3：RF 分层 —— 人数占比 vs 金额占比（复用 SQL 的分层规则）
#   分层规则只消费 R / F 两个维度；M 与 F 秩相关 0.86，加入会重复计权，
#   故 M 仅作为「金额描述列」保留，不参与定层（详见 README 局限说明）。
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
# 标题数字全部由本次计算结果生成，避免改代码后标题与柱子不一致
w_row = seg[seg.index.str.startswith("重要价值客户")].iloc[0]
ax.set_title(f"RF 分层：重要价值客户 {w_row['人数占比']:.1f}% 的人数贡献 {w_row['金额占比']:.1f}% 的金额",
             fontsize=13, pad=12)
ax.legend()
save_fig(fig, "03_rfm_segments.png")

# 分层明细（控制台 & 供报告核对；文档里的数字以这里为准）
stat = (rfm.groupby("segment")
           .agg(人数=("customer_id", "count"), 金额=("monetary", "sum"),
                平均购买次数=("frequency", "mean"), 平均距上次购买=("recency", "mean")))
stat["人数占比"] = stat["人数"] / stat["人数"].sum() * 100
stat["金额占比"] = stat["金额"] / stat["金额"].sum() * 100
print("\n[分层明细] 请以此为准同步 README / 报告.md：")
print(stat[["人数占比", "金额占比", "平均购买次数", "平均距上次购买"]]
      .sort_values("金额占比", ascending=False).round(1).to_string())

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
m1 = heat["第1月"].median()          # 次月留存中位数：由数据算出，不写死
ax.set_title(f"留存热力图：次月留存中位数 {m1:.0f}%，留存下来的客户会持续回访", fontsize=13, pad=12)
save_fig(fig, "04_retention_heatmap.png")
print(f"\n[留存] 次月留存中位数 {m1:.1f}%｜各 cohort 次月留存范围 "
      f"{heat['第1月'].min():.0f}–{heat['第1月'].max():.0f}%")

# ---------- 保存 RFM 明细表供报告使用 ----------
rfm.to_csv("data/rfm_customers.csv", index=False)
print("  [saved] data/rfm_customers.csv")
print("=" * 50)
print("可视化完成：charts/ 下 4 张图")
