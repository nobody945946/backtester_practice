# 趨勢驗證系統 (Trend Validator Protocol) v2.2

生產級台股回測引擎，實現真實的訂單執行流程與台灣市場成本模型。

## 核心功能

- **交易對象**：台灣股票（4碼代碼，TWSE + TPEx）
- **回測期間**：無缺口的每日淨值記錄（即時清算價值）
- **執行模型**：信號在收盤→隔日開盤成交（Option C）
- **訂單有效期**：買單5天自動過期
- **成本模型**：
  - 手續費：0.0399%（含28%折扣）
  - 賣稅：0.3%
  - 滑點：0.1%
- **部位管理**：最多10檔股票，追蹤型停損，多重退場條件

## 四階段篩選機制

1. **市場機制** - 收盤 > 200日均線（二元門檻）
2. **趨勢效率** - 買賣訊號比 (ER) + 均線斜率（25%權重）
3. **動量持續性** - T統計量 + 尾部風險（45%權重）
4. **確認信號** - 突破 + 量能尖峰（30%權重）

**版本差異**：
- v1.1：固定ER=0.40，簡化動量檢驗
- v2：動態ER（跨截面70分位），嚴格CVaR尾部風險

## 快速開始

```python
from backtester import TrendValidatorBacktest

# 初始化回測器
bt = TrendValidatorBacktest(
    version='v2',
    initial_capital=1000000,
    max_positions=10,
    slippage_pct=0.001
)

# 執行回測（2024全年）
stats = bt.run_backtest(
    tickers=['2330', '2454', '3034'],  # 4碼代碼
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# 繪製淨值曲線
bt.plot_equity_curve()
```

## 主要參數

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `atr_period` | 20 | 波動率計算周期 |
| `atr_multiplier` | 2.5 | 停損距離（ATR倍數） |
| `max_positions` | 10 | 最大持股數 |
| `er_percentile` | 0.70 | v2動態ER閾值（跨截面） |
| `order_expiry_days` | 5 | 買單有效天數 |

## 關鍵指標

- **威爾德ATR**：指數加權移動平均（非簡單平均）
- **Kaufman ER**：方向/波動率比（0～1）
- **VWAP**：成交量加權平均價
- **Donchian通道**：使用前20根K棒（避免前瞻性偏誤）

## 回測輸出

```python
{
    'total_return': 0.245,          # 總報酬
    'annual_return': 0.182,         # 年化報酬
    'sharpe_ratio': 1.23,           # 夏普比
    'max_drawdown': -0.185,         # 最大回檔
    'win_rate': 0.58,               # 勝率
    'trades': [...],                # 完整交易記錄
    'equity_curve': [...]           # 每日淨值
}
```

## 依賴套件

```bash
pip install yfinance pandas numpy matplotlib scipy
```

## 設計特色

✅ 單一`TrendValidatorBacktest`類（無副類）  
✅ 訂單統一由`execute_pending_orders()`執行（確保成本一致性）  
✅ 預留現金機制（防止透支）  
✅ 停損自動追蹤（不會下移）  
✅ 版本分支邏輯（v1.1 vs v2）清晰內聚  

## 常見問題

**現金為負？**  
檢查預留現金是否正確釋放（訂單成交/過期/取消時）

**淨值曲線有缺口？**  
確認`record_equity()`每日都被調用（即使無交易）

**部位未退場？**  
驗證SELL訂單是否被創建，且在下個交易日成交

---

**最後更新**：2026年1月  
**版本**：v2.2 (Production Grade)
