DECLARE @start VARCHAR(8) = '20150101';
DECLARE @end   VARCHAR(8) = '20231231';

WITH cal AS (
    SELECT TRADE_DAYS AS TradeDate
    FROM dbo.ASHARECALENDAR
    WHERE S_INFO_EXCHMARKET = 'SSE'
      AND TRADE_DAYS BETWEEN @start AND @end
),
effective_weight_date AS (
    SELECT
        c.TradeDate,
        wd.WeightDate
    FROM cal c
    OUTER APPLY (
        SELECT TOP 1 w.TRADE_DT AS WeightDate
        FROM dbo.AINDEXHS300FREEWEIGHT w
        WHERE w.S_INFO_WINDCODE = '000300.SH'
          AND w.TRADE_DT <= c.TradeDate
        ORDER BY w.TRADE_DT DESC
    ) wd
),
daily_members AS (
    SELECT
        e.TradeDate,
        w.S_CON_WINDCODE AS StockCode,
        w.I_WEIGHT AS IndexWeight
    FROM effective_weight_date e
    JOIN dbo.AINDEXHS300FREEWEIGHT w
      ON w.S_INFO_WINDCODE = '000300.SH'
     AND w.TRADE_DT = e.WeightDate
)
SELECT
    m.StockCode,
    m.TradeDate,
    p.S_DQ_ADJCLOSE AS ClosePrice,
    p.S_DQ_VOLUME AS Volume,
    p.S_DQ_AMOUNT AS Amount,
    m.IndexWeight
FROM daily_members m
LEFT JOIN dbo.ASHAREEODPRICES p
  ON p.S_INFO_WINDCODE = m.StockCode
 AND p.TRADE_DT = m.TradeDate
ORDER BY m.TradeDate, m.StockCode;
