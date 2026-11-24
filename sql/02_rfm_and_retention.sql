-- ============================================================
-- 02_rfm_and_retention.sql
-- 作用：RFM 客户分层 + 留存（Cohort）分析——本项目两大分析主角
-- 前提：orders 表已导入
-- 学习目标：DATEDIFF、CTE、窗口函数、多级聚合
-- ============================================================

USE ecommerce;

-- ─────────────────────────────────────────────
-- 第一部分：RFM 分层
--   R (Recency)   最近一次购买距观测终点多少天 —— 越小越活跃
--   F (Frequency) 有效购买次数（按发票去重）   —— 越大越忠诚
--   M (Monetary)  累计消费金额                 —— 越大越值钱
-- 观测终点：数据里最后一天（2011-12-09），不是今天！
-- ─────────────────────────────────────────────

-- 步骤 1：算出每个客户的原始 R / F / M
DROP TABLE IF EXISTS rfm_raw;
CREATE TABLE rfm_raw AS
SELECT
  customer_id,
  DATEDIFF((SELECT MAX(order_date) FROM orders), MAX(order_date)) AS recency_days,
  COUNT(DISTINCT invoice)                                          AS frequency,
  ROUND(SUM(amount), 2)                                            AS monetary
FROM orders
GROUP BY customer_id;

SELECT * FROM rfm_raw LIMIT 10;

-- 步骤 2：打分（NTILE(5) 均匀分成 1–5 分）+ 分层
-- R 得分越大 = 越近期买过；F/M 得分越大 = 消费越多
DROP TABLE IF EXISTS rfm_scored;
CREATE TABLE rfm_scored AS
SELECT
  customer_id, recency_days, frequency, monetary,
  NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,  -- recency 越大越久远→分越低
  NTILE(5) OVER (ORDER BY frequency)         AS f_score,
  NTILE(5) OVER (ORDER BY monetary)          AS m_score
FROM rfm_raw;

-- 步骤 3：给每个客户贴业务标签（简化版分层，面试讲得清）
DROP TABLE IF EXISTS rfm_tagged;
CREATE TABLE rfm_tagged AS
SELECT
  *,
  CONCAT(r_score, f_score, m_score) AS rfm_code,
  CASE
    WHEN r_score >= 4 AND f_score >= 4 THEN '重要价值客户（维护）'
    WHEN r_score >= 4 AND f_score < 4  THEN '重要发展客户（提升频次）'
    WHEN r_score < 4 AND f_score >= 4  THEN '重要挽留客户（唤醒）'
    WHEN r_score <= 2 AND f_score <= 2 THEN '流失客户（放弃或低成本触达）'
    ELSE '一般客户'
  END AS segment
FROM rfm_scored;

-- 步骤 4：分层汇总——这就是分析报告里的核心表之一
SELECT
  segment                    AS `客户分层`,
  COUNT(*)                   AS `客户数`,
  ROUND(COUNT(*) / (SELECT COUNT(*) FROM rfm_tagged) * 100, 1) AS `人数占比%`,
  ROUND(SUM(monetary), 0)    AS `累计消费`,
  ROUND(SUM(monetary) / (SELECT SUM(monetary) FROM rfm_tagged) * 100, 1) AS `金额占比%`
FROM rfm_tagged
GROUP BY segment
ORDER BY 金额占比 DESC;

-- ─────────────────────────────────────────────
-- 第二部分：留存分析（Cohort）
--   问题：某个月首次下单的客户，之后每个月还有多少比例回来买？
--   这是"流失"最直接的度量——首购月为锚点往后看
-- ─────────────────────────────────────────────
WITH first_purchase AS (       -- 每个客户的首次购买月
  SELECT customer_id, MIN(order_month) AS cohort_month
  FROM orders
  GROUP BY customer_id
),
activity AS (                  -- 客户×月份 的所有活跃记录，挂上首购月
  SELECT
    f.cohort_month,
    TIMESTAMPDIFF(MONTH, STR_TO_DATE(CONCAT(f.cohort_month,'-01'), '%Y-%m-%d'),
                         STR_TO_DATE(CONCAT(o.order_month,'-01'), '%Y-%m-%d')) AS month_offset,
    o.customer_id
  FROM orders o
  JOIN first_purchase f USING (customer_id)
  GROUP BY f.cohort_month, month_offset, o.customer_id
)
SELECT
  cohort_month                                        AS `首购月`,
  month_offset                                        AS `第N月`,
  COUNT(DISTINCT customer_id)                         AS `活跃客户数`
FROM activity
GROUP BY cohort_month, month_offset
ORDER BY cohort_month, month_offset;
-- ↑ 这张"长表"就是留存表的原型。
--   想要"留存率%"：把每个 cohort 的 month_offset=N 除以 month_offset=0 的人数，
--   在 Python 里转成透视表画热力图更直观（analysis/03_retention.py 会做）。
