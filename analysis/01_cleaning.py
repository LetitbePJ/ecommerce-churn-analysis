"""
01_cleaning.py — 数据清洗
数据集：Online Retail II（英国电商 2009-12 ~ 2011-12，1,067,371 行）
目标：输出一份可做用户分析的干净数据 data/clean_orders.csv
"""

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

RAW = "data/online_retail_ii.csv"
OUT = "data/clean_orders.csv"

df = pd.read_csv(RAW)
print(f"[0] 原始数据：{df.shape[0]:,} 行 × {df.shape[1]} 列")
print("列名：", list(df.columns))

# ============ 1. 重复值 ============
before = len(df)
df = df.drop_duplicates()
print(f"[1] 重复值：删除 {before - len(df):,} 行，剩余 {len(df):,} 行")

# ============ 2. 缺失值 ============
print("[2] 缺失情况：")
print((df.isnull().mean() * 100).round(2).astype(str) + " %")

# 处理决策（非待办，已定稿）：
#   - Customer ID 缺失 22.8%（243,007 行）：无法归因到具体客户，剔除。这批多为游客单，
#     只影响"整体销售额"这一口径，不影响任何客户级指标（复购率 / 留存 / RF 分层均按客户聚合）
#   - Description 缺失 4,382 行：该字段不参与任何指标计算，仅用于人工抽查商品构成，
#     因此**不做** StockCode 反查补全，保留缺失（占比 <0.6%，不影响结论）
df = df.dropna(subset=["Customer ID"])

# ============ 3. 异常值 ============
# 阈值依据（已定稿；如需调整须同步 README 与 报告.md）：
#   - Quantity <= 0 = 退货 / 取消单 / 坏账调整 → 本版只分析正向购买行为，整段剔除
#     代价：销售额口径偏乐观（未扣退货），且丢掉"退货"这个潜在流失信号，已在报告 §5 局限声明
#   - Price <= 0（赠品、坏账调整）或 > 10,000（人工录入错误）→ 剔除
df["is_cancel"] = df["Invoice"].astype(str).str.startswith("C")
n_cancel = int((df["is_cancel"] | (df["Quantity"] <= 0)).sum())
rows_before = len(df)
df = df[(df["Quantity"] > 0) & (df["Price"] > 0) & (df["Price"] <= 10000)]
print(f"[3] 异常值：剔除 {rows_before - len(df):,} 行"
      f"（其中退货/取消 {n_cancel:,} 行），剩余 {len(df):,} 行")

# ============ 4. 类型统一 ============
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Customer ID"] = df["Customer ID"].astype(int)
df["Country"] = df["Country"].astype("category")
# 注：is_cancel 在剔除负数量后恒为 False（退货行已整段剔除），保留该列只为与 sql/00 表结构一致；
# 退货规模记录在清洗报告的「退货取消行数」里，供后续做"退货 → 流失"关联分析

# 新增派生字段（后面分析要用）
df["amount"] = df["Quantity"] * df["Price"]        # 该行金额
df["order_date"] = df["InvoiceDate"].dt.date       # 只留日期
df["order_month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

print("[4] 类型：", df.dtypes.to_dict())
print("    时间范围：", df["InvoiceDate"].min(), "~", df["InvoiceDate"].max())
print(f"    客户数：{df['Customer ID'].nunique():,}｜订单数：{df['Invoice'].nunique():,}")

# ============ 5. 导出 + 清洗报告 ============
df.to_csv(OUT, index=False, encoding="utf-8-sig")

report = {
    "原始行数": 1_067_371,
    "清洗后行数": len(df),
    "删除比例": f"{(1 - len(df) / 1_067_371) * 100:.1f}%",
    "客户数": df["Customer ID"].nunique(),
    "订单数": df["Invoice"].nunique(),
    "总销售额(英镑)": f"{df['amount'].sum():,.0f}",
    "退货取消行数(已剔除)": n_cancel,
    "观测终点": str(df["InvoiceDate"].max().date()),
}
print("\n[5] 清洗报告")
for k, v in report.items():
    print(f"    {k}: {v}")
print(f"\n已输出：{OUT}")
