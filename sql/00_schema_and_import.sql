-- ============================================================
-- 00_schema_and_import.sql
-- 作用：建库建表 + 把清洗后的 79 万行数据导入 MySQL
-- 运行环境：MySQL Workbench（打开本文件，逐段执行）
-- ============================================================

-- ① 建库（如已存在则跳过）
CREATE DATABASE IF NOT EXISTS ecommerce
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE ecommerce;

-- ② 建表
-- 注意：一行 = 一张发票上的一行商品（不是一张订单！）
DROP TABLE IF EXISTS orders;
CREATE TABLE orders (
  auto_id       INT UNSIGNED  NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  invoice       VARCHAR(10)   NOT NULL COMMENT '发票号=订单号，C开头为取消单',
  stock_code    VARCHAR(20)   NOT NULL COMMENT '商品编码',
  description   VARCHAR(255)           COMMENT '商品名',
  quantity      INT           NOT NULL COMMENT '数量（清洗后均为正）',
  invoice_date  DATETIME      NOT NULL COMMENT '下单时间',
  price         DECIMAL(10,2) NOT NULL COMMENT '单价 GBP',
  customer_id   INT           NOT NULL COMMENT '客户ID（清洗后无缺失）',
  country       VARCHAR(60)            COMMENT '国家',
  period        VARCHAR(10)            COMMENT '来源年度表',
  is_cancel     TINYINT       DEFAULT 0 COMMENT '是否取消单 0/1',
  amount        DECIMAL(12,2) NOT NULL COMMENT '行金额=数量*单价',
  order_date    DATE          NOT NULL COMMENT '下单日期',
  order_month   VARCHAR(7)    NOT NULL COMMENT '下单月份 YYYY-MM',
  PRIMARY KEY (auto_id),
  KEY idx_customer (customer_id),
  KEY idx_month (order_month),
  KEY idx_invoice (invoice)
) COMMENT='清洗后的订单明细表（793,607 行）';

-- ③ 导入数据（两种方式任选一种）
--
-- 方式 A（推荐新手）：Workbench 图形界面导入
--   左侧 SCHEMAS 面板 → ecommerce → 表 orders 右键
--   → Table Data Import Wizard → 选 data/clean_orders.csv → 一路 Next
--   ⚠️ 到"列映射"那一步，确认 CSV 列和表列按名字对上（auto_id 跳过）
--
-- 方式 B：LOAD DATA（快，10 秒导完）
--   先看当前设置是否允许（运行下面这句，看值是否为 1）：
SHOW VARIABLES LIKE 'local_infile';
--   如果值是 OFF，先用 root 执行：
--     SET GLOBAL local_infile = 1;
--   然后重连会话，再执行下面这句（路径改成你电脑上的实际路径）：
-- LOAD DATA LOCAL INFILE
--   'D:/path/to/ecommerce-churn-analysis/data/clean_orders.csv'   -- ← 改成你本机的绝对路径
-- INTO TABLE orders
-- FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
-- LINES TERMINATED BY '\n'
-- IGNORE 1 LINES
-- (invoice, stock_code, description, quantity, invoice_date,
--  price, customer_id, country, period, @skip_cancel,
--  amount, order_date, order_month)
-- SET is_cancel = (@skip_cancel = 'True');

-- ④ 验收：三个数字对得上就算导入成功
SELECT COUNT(*)            AS 总行数      -- 应 = 793607
  , COUNT(DISTINCT customer_id) AS 客户数  -- 应 = 5877
  , COUNT(DISTINCT invoice)     AS 订单数  -- 应 = 36967
  , ROUND(SUM(amount), 0)       AS 总销售额 -- 应 ≈ 17664038
FROM orders;
