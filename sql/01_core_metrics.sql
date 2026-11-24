-- ============================================================
-- 01_core_metrics.sql
-- 作用：4 个基础核心指标——面试手写题的原型
-- 前提：orders 表已导入（见 00_schema_and_import.sql）
-- 学习目标：聚合、GROUP BY、DISTINCT、窗口函数入门
-- ============================================================

USE ecommerce;

-- ─────────────────────────────────────────────
-- 指标 1：总览大盘（总销售额 / 订单数 / 客户数 / AOV）
-- ─────────────────────────────────────────────
SELECT
  ROUND(SUM(amount), 0)                 AS total_revenue,      -- 总销售额
  COUNT(DISTINCT invoice)               AS total_orders,       -- 订单数
  COUNT(DISTINCT customer_id)           AS total_customers,    -- 客户数
  ROUND(SUM(amount) / COUNT(DISTINCT invoice), 2) AS avg_order_value -- 客单价 AOV
FROM orders;

-- ─────────────────────────────────────────────
-- 指标 2：月度销售趋势（看季节性：圣诞前是不是旺季？）
-- ─────────────────────────────────────────────
SELECT
  order_month                            AS `月份`,
  COUNT(DISTINCT invoice)                AS `订单数`,
  ROUND(SUM(amount), 0)                  AS `销售额`,
  ROUND(SUM(amount) / COUNT(DISTINCT invoice), 0) AS `客单价`
FROM orders
GROUP BY order_month
ORDER BY order_month;

-- ─────────────────────────────────────────────
-- 指标 3：复购率（本项目的主角指标）
-- 定义：下单 ≥2 个不同日期的客户占比
-- 陷阱提醒：用 invoice 数判断复购会高估——同一订单拆多行
-- ─────────────────────────────────────────────
WITH cust_days AS (            -- 每个客户去重后的下单日期
  SELECT customer_id, COUNT(DISTINCT order_date) AS active_days
  FROM orders
  GROUP BY customer_id
)
SELECT
  COUNT(*)                                                        AS 全部客户,
  SUM(active_days >= 2)                                           AS 复购客户,
  ROUND(SUM(active_days >= 2) / COUNT(*) * 100, 1)                AS `复购率%`
FROM cust_days;

-- ─────────────────────────────────────────────
-- 指标 4：客户价值二八法则（Top 10% 客户贡献多少销售额？）
-- 窗口函数：先按客户汇总，再算累计占比
-- ─────────────────────────────────────────────
WITH cust_rev AS (
  SELECT customer_id, SUM(amount) AS revenue
  FROM orders
  GROUP BY customer_id
),
ranked AS (
  SELECT *,
    NTILE(10) OVER (ORDER BY revenue DESC) AS decile   -- 按消费额分成10档
  FROM cust_rev
)
SELECT
  CONCAT('Top ', decile * 10, '%') AS 客户分层,
  COUNT(*)                                AS 客户数,
  ROUND(SUM(revenue), 0)                  AS 销售额,
  ROUND(SUM(revenue) / (SELECT SUM(amount) FROM orders) * 100, 1) AS `销售额占比%`
FROM ranked
GROUP BY decile
ORDER BY decile;
