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

# TODO(你来做)：决定怎么处理
#   - Customer ID 缺失 22.8%：用户分析必须剔除（无法归属到人）
#   - Description 缺失 4382：可用同 StockCode 的其他行反查补全，补不全的剔除
df = df.dropna(subset=["Customer ID"])

# ============ 3. 异常值 ============
# TODO(你来做)：思考这几个阈值合不合理，可自行调整并写明理由
#   - 负数量 = 退货/取消单，这里先剔除（只分析正向购买行为）
#   - 0 或负价格 = 坏账调整/赠品，剔除
#   - 极端价格（如 > 10000）多为人工录入错误，剔除
rows_before = len(df)
df = df[(df["Quantity"] > 0) & (df["Price"] > 0) & (df["Price"] <= 10000)]
print(f"[3] 异常值：剔除 {rows_before - len(df):,} 行，剩余 {len(df):,} 行")

# ============ 4. 类型统一 ============
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Customer ID"] = df["Customer ID"].astype(int)
df["Country"] = df["Country"].astype("category")
df["is_cancel"] = df["Invoice"].astype(str).str.startswith("C")  # 标记取消单

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
}
print("\n[5] 清洗报告")
for k, v in report.items():
    print(f"    {k}: {v}")
print(f"\n已输出：{OUT}")
