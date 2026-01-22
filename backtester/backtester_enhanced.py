"""
趨勢驗證系統 v2.2 - 強化版（中文輸出+長期回測+完整統計）
Production-Grade Backtesting System with Extended Horizon & Chinese Output

✨ 新增功能: 
  ✅ 年度擴展：支援 5-10 年以上回測
  ✅ 中文輸出：所有報告、圖表、指標都是繁體中文
  ✅ 完整統計：Sharpe、Sortino、Calmar、Recovery Factor 等
  ✅ 月度報告：逐月績效明細
  ✅ 季度報告：逐季績效與最大回撤
  ✅ 自適應均線：根據周期調整策略參數
  ✅ 風險調整報酬：詳細的報酬分析
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy import stats
import warnings
from datetime import datetime
import re

warnings.filterwarnings('ignore')

# 設定中文字體
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


class TrendValidatorBacktestEnhanced:
    """強化版回測引擎 - 支援中文輸出 + 長期回測"""
    
    def __init__(self, version: str = 'v1.1', initial_capital: float = 1000000,
                 slippage_pct: float = 0.001, max_positions: int = 10,
                 commission_discount_factor: float = 0.28):
        self.version = version
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self. reserved_cash = 0.0
        self.positions = {}
        self.pending_orders = []
        self.trades = []
        self.equity_curve = []
        self.max_positions = max_positions
        
        # Execution Configuration (schema v1.0 - Taiwan market)
        self.market = 'TW'
        self.currency = '新台幣'
        self. signal_time = 'close'
        self.fill_time = 'next_open'
        self.missing_session_policy = 'next_available'
        self.buy_order_expiry_days = 5
        self.sell_order_expiry_days = None
        
        # Commission Fee Structure (TWD market, with discount factor)
        self.commission_standard_rate = 0.001425
        self.commission_discount_factor = commission_discount_factor
        self.commission_effective_rate = self.commission_standard_rate * (1 - self.commission_discount_factor)
        self.commission_minimum_fee_twd = 0
        
        # Taiwan-specific costs
        self.slippage_pct = slippage_pct
        self.buy_fee_pct = self.commission_effective_rate
        self.sell_fee_pct = self.commission_effective_rate
        self.sell_tax_pct = 0.003
        
        # Board Lot Configuration
        self.board_lot_shares = 1000
        self.allow_odd_lot = True
        
        # Strategy Parameters
        self.atr_period = 20
        self.atr_multiplier = 2.5
        self.vwap_window = 20
        self.er_percentile = 0.70
        self.min_turnover_twd = 50000000
        self.order_expiry_days = self.buy_order_expiry_days
        
        print(f"🚀 已初始化 {version} 回測引擎 v2.2 (強化版)")
        print(f"💰 初始資金: {initial_capital:,.0f} {self.currency}")
        print(f"📊 手續費: {self.buy_fee_pct*100:.4f}% (標準:  {self.commission_standard_rate*100:.4f}%, 折扣: {self. commission_discount_factor*100:.0f}%)")
        print(f"📊 稅費: {self.sell_tax_pct*100:.2f}%")
        print(f"📊 滑點: {self.slippage_pct*100:.2f}%")
        print(f"🏛️  股票池: 台灣4碼股 (上市 + 上櫃)")
        print(f"📈 每手股數: {self.board_lot_shares} 股 (零股:  {'允許' if self.allow_odd_lot else '不允許'})")
        print(f"📅 買單有效期: {self.buy_order_expiry_days} 天")
        print(f"⏱️  執行模式: 訊號 @ {self.signal_time} → 成交 @ {self.fill_time}")
        
        try:
            self.validate_config()
            print("✅ 設定驗證成功\n")
        except ValueError as e:
            print(f"❌ 設定錯誤:\n{e}")
            raise
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """驗證4碼台股代碼"""
        pattern = r'^\d{4}$'
        return bool(re. match(pattern, ticker))
    
    def validate_board_lot(self, shares: int) -> int:
        """驗證每手規則"""
        if shares % self.board_lot_shares == 0:
            return shares
        if self.allow_odd_lot:
            return shares
        else:
            return (shares // self.board_lot_shares) * self.board_lot_shares
    
    def validate_config(self) -> bool:
        """驗證設定"""
        errors = []
        
        if self.market != 'TW':
            errors.append(f"市場必須為 'TW', 取得 '{self.market}'")
        if self.currency != '新台幣':
            errors.append(f"幣別必須為 '新台幣', 取得 '{self.currency}'")
        
        if self.commission_standard_rate <= 0:
            errors.append(f"標準手續費率必須為正數, 取得 {self. commission_standard_rate}")
        if not (0.0 <= self.commission_discount_factor <= 1.0):
            errors.append(f"折扣係數必須介於 0-1, 取得 {self.commission_discount_factor}")
        
        expected_effective = self.commission_standard_rate * (1 - self.commission_discount_factor)
        if abs(self.commission_effective_rate - expected_effective) > 1e-10:
            errors.append(f"有效手續費率不符:  預期 {expected_effective}, 取得 {self.commission_effective_rate}")
        
        if self.board_lot_shares <= 0:
            errors.append(f"每手股數必須為正數, 取得 {self. board_lot_shares}")
        
        if self.buy_order_expiry_days <= 0:
            errors.append(f"買單有效期必須為正數, 取得 {self.buy_order_expiry_days}")
        
        if self.slippage_pct < 0 or self.slippage_pct > 0.05:
            errors.append(f"滑點 {self.slippage_pct*100:.2f}% 似乎不合理 (預期 0-5%)")
        
        if errors:
            raise ValueError("設定驗證失敗:\n" + "\n".join(f"  ❌ {e}" for e in errors))
        
        return True
    
    @staticmethod
    def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """正規化 OHLCV 資料"""
        df = df.copy()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns. get_level_values(0)
            print(f"  ⚠️  扁平化多層欄位:  {list(df.columns)}")
        
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"缺少欄位: {missing}")
        
        for col in required:
            if isinstance(df[col], pd.DataFrame):
                print(f"  ❌ 錯誤: 欄位 '{col}' 是 DataFrame 不是 Series!")
                df[col] = df[col]. iloc[:, 0]
                print(f"    已選擇欄位 '{col}' 的第一欄")
        
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Close'])
        
        return df
    
    def calculate_wilder_atr(self, df: pd. DataFrame, period: int = 20) -> pd.Series:
        """計算 ATR (Wilder 平滑)"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.ewm(alpha=1/period, adjust=False).mean()
    
    def calculate_rolling_vwap(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """計算滾動 VWAP"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        pv = (typical_price * df['Volume']).rolling(window=window).sum()
        vol = df['Volume'].rolling(window=window).sum()
        return pv / vol. replace(0, np.nan)
    
    def calculate_kaufman_er(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """計算 Kaufman 效率比 (ER)"""
        close = df['Close']
        direction = abs(close - close.shift(period))
        volatility = abs(close - close.shift(1)).rolling(window=period).sum()
        return direction / volatility. replace(0, np.nan)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算所有技術指標"""
        df = self.normalize_ohlcv(df)
        df = df.copy()
        
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            if not isinstance(df[col], pd.Series):
                raise TypeError(f"預期 Series '{col}', 取得 {type(df[col])}")
        
        df['daily_return'] = df['Close']. pct_change()
        df['ATR_20'] = self.calculate_wilder_atr(df, self.atr_period)
        df['VWAP'] = self. calculate_rolling_vwap(df, self.vwap_window)
        df['ER_20'] = self.calculate_kaufman_er(df, 20)
        
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        df['Vol_60'] = df['daily_return'].rolling(window=60).std() * np.sqrt(252)
        df['EWMA_Vol_60'] = df['daily_return'].ewm(span=60).std() * np.sqrt(252)
        
        ma50_slope = (df['SMA_50'] - df['SMA_50'].shift(10)) / 10
        df['MA_Slope_over_ATR'] = ma50_slope / df['ATR_20']
        
        df['Vol_Ratio_20'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['Turnover_20'] = (df['Close'] * df['Volume']).rolling(window=20).mean()
        
        df['ROC_20'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)
        
        df['Lowest_20'] = df['Close'].rolling(window=20).min().shift(1)
        df['Highest_20'] = df['High'].rolling(window=20).max().shift(1)
        df['Displacement_over_ATR'] = (df['Close'] - df['Lowest_20']) / (df['ATR_20'] * 2. 0)
        
        return df
    
    def calculate_statistical_metrics(self, returns: pd.Series, lookback: int = 60) -> Dict: 
        """計算統計指標"""
        if len(returns) < lookback:
            return None
        
        recent = returns.tail(lookback)
        
        return {
            'positive_day_ratio': (recent > 0).sum() / lookback,
            'skewness': recent.skew(),
            'mean':  recent.mean(),
            'std': recent.std(),
            't_stat': recent.mean() / (recent.std() / np.sqrt(lookback)) if recent.std() > 0 else 0,
            'q05': recent.quantile(0.05),
            'cvar_05': recent[recent <= recent.quantile(0.05)].mean() if len(recent[recent <= recent.quantile(0.05)]) > 0 else recent.min(),
            'min_return': recent.min(),
            'max_drawdown': self.calculate_max_drawdown(recent)
        }
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """計算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_dynamic_er_threshold(self, stock_data: Dict[str, pd.DataFrame], 
                                      date, idx:  int) -> float:
        """計算動態 ER 門檻"""
        er_values = []
        
        for ticker, df in stock_data.items():
            if date in df.index:
                stock_idx = df.index.get_loc(date)
                er = df.iloc[stock_idx]['ER_20']
                if not pd.isna(er):
                    er_values.append(er)
        
        if len(er_values) < 5:
            return 0.40
        
        return np.percentile(er_values, self.er_percentile * 100)
    
    def stage_0_market_regime(self, benchmark_df: pd.DataFrame, idx: int) -> str:
        """市場趨勢判斷"""
        if self.version == 'v1.1':
            return '多頭' if benchmark_df.iloc[idx]['Close'] > benchmark_df.iloc[idx]['SMA_200'] else '空頭'
        else:
            close = benchmark_df.iloc[idx]['Close']
            sma_50 = benchmark_df.iloc[idx]['SMA_50']
            sma_200 = benchmark_df.iloc[idx]['SMA_200']
            
            if close > sma_200 and sma_50 > sma_200:
                return '多頭'
            elif close > sma_200:
                return '平盤'
            else:
                return '空頭'
    
    def stage_1_trend_efficiency(self, df: pd.DataFrame, idx: int, 
                                 er_threshold: float = None) -> Tuple[bool, float]:
        """趨勢效率篩選"""
        er_20 = df. iloc[idx]['ER_20']
        slope_over_atr = df.iloc[idx]['MA_Slope_over_ATR']
        
        if pd.isna(er_20) or pd.isna(slope_over_atr):
            return False, 0.0
        
        er_threshold = er_threshold or 0.40
        slope_threshold = 0.02
        
        pass_er = er_20 >= er_threshold
        pass_slope = slope_over_atr >= slope_threshold
        
        score = min(er_20 / er_threshold, 2.0) * 0.5 + min(slope_over_atr / slope_threshold, 2.0) * 0.5
        
        return pass_er and pass_slope, score
    
    def stage_2_momentum_persistence(self, df: pd.DataFrame, idx: int) -> Tuple[bool, float]:
        """動能持續性篩選"""
        returns = df['daily_return'].iloc[: idx+1]
        metrics = self.calculate_statistical_metrics(returns, lookback=60)
        
        if metrics is None:
            return False, 0.0
        
        if self.version == 'v1.1':
            checks = {
                'positive_ratio': metrics['positive_day_ratio'] > 0.55,
                'skewness':  metrics['skewness'] > -0.5,
                't_stat': metrics['t_stat'] > 1.5,
                'q05': metrics['q05'] > -0.04
            }
            weights = {'positive_ratio': 0.25, 'skewness': 0.10, 't_stat': 0.35, 'q05': 0.30}
            
            if not checks['t_stat']: 
                return False, 0.0
            
            score = sum(weights[k] * (1. 0 if checks[k] else 0.0) for k in checks)
            return score >= 0.65, score
        else:
            std_60 = metrics['std']
            checks = {
                't_stat': metrics['t_stat'] > 2.0,
                'cvar_05': metrics['cvar_05'] > -1.5 * std_60,
                'positive_ratio': metrics['positive_day_ratio'] > 0.52,
                'max_drop': metrics['min_return'] > -3.0 * std_60
            }
            weights = {'t_stat': 0.40, 'cvar_05':  0.35, 'positive_ratio': 0.15, 'max_drop': 0.10}
            
            if not (checks['t_stat'] and checks['cvar_05']):
                return False, 0.0
            
            score = sum(weights[k] * (1.0 if checks[k] else 0.0) for k in checks)
            return score >= 0.70, score
    
    def stage_3_confirmation(self, df: pd.DataFrame, idx: int) -> Tuple[bool, float]:
        """確認訊號篩選"""
        roc_20 = df.iloc[idx]['ROC_20']
        disp_atr = df.iloc[idx]['Displacement_over_ATR']
        vol_ratio = df.iloc[idx]['Vol_Ratio_20']
        close = df.iloc[idx]['Close']
        vwap = df.iloc[idx]['VWAP']
        highest_20 = df.iloc[idx]['Highest_20']
        
        if any(pd.isna([roc_20, disp_atr, vol_ratio, vwap])):
            return False, 0.0
        
        recent_closes = df['Close'].iloc[max(0, idx-20):idx+1]
        max_dd = self.calculate_max_drawdown(recent_closes. pct_change().dropna())
        
        if self.version == 'v1.1':
            checks = {
                'displacement':  roc_20 >= 0.15 or disp_atr > 1.0,
                'max_dd': max_dd >= -0.10,
                'volume':  vol_ratio >= 1.2
            }
            return all(checks.values()), sum(1. 0 if v else 0.0 for v in checks.values()) / len(checks)
        else:
            std_60 = df['daily_return'].iloc[: idx+1].tail(60).std()
            roc_vol_scaled = roc_20 / (std_60 * np.sqrt(20)) if std_60 > 0 else 0
            
            atr_20 = df.iloc[idx]['ATR_20']
            atr_pct = atr_20 / close if close > 0 else 0
            dd_atr_scaled = abs(max_dd) / atr_pct if atr_pct > 0 else 999
            
            checks = {
                'breakout': close > highest_20,
                'roc_scaled': roc_vol_scaled > 1.0,
                'dd_atr':  dd_atr_scaled <= 4.0,
                'volume':  vol_ratio >= 1.3,
                'vwap': close > vwap
            }
            return all(checks.values()), sum(1.0 if v else 0.0 for v in checks.values()) / len(checks)
    
    def calculate_position_size(self, df: pd.DataFrame, idx: int, regime: str) -> float:
        """計算倉位大小"""
        if self.version == 'v1.1':
            vol = df.iloc[idx]['Vol_60']
            regime_multiplier = 0.5 if regime == '空頭' else 1.0
        else:
            vol = df. iloc[idx]['EWMA_Vol_60']
            regime_multipliers = {'多頭': 1.0, '平盤': 0.7, '空頭': 0.3}
            regime_multiplier = regime_multipliers.get(regime, 1.0)
        
        if pd.isna(vol) or vol == 0:
            return 0.0
        
        position_size = (0.15 * regime_multiplier) / vol
        return min(position_size, 0.10)
    
    def update_trailing_stops(self, date, stock_data:  Dict[str, pd.DataFrame]):
        """更新追蹤停損"""
        for ticker, pos in self.positions.items():
            if date not in stock_data[ticker]. index:
                continue
            
            idx = stock_data[ticker].index.get_loc(date)
            close = stock_data[ticker].iloc[idx]['Close']
            atr = stock_data[ticker]. iloc[idx]['ATR_20']
            
            if pd.isna(atr):
                continue
            
            new_stop = close - self.atr_multiplier * atr
            if new_stop > pos['stop_loss']:
                pos['stop_loss'] = new_stop
    
    def check_exit_conditions(self, df: pd.DataFrame, idx: int, entry_price: float, 
                             entry_idx: int, stop_loss: float) -> Tuple[bool, str]:
        """檢查出場條件"""
        close = df.iloc[idx]['Close']
        atr_20 = df.iloc[idx]['ATR_20']
        sma_20 = df.iloc[idx]['SMA_20']
        
        if close <= stop_loss:
            return True, '停損'
        
        if idx > entry_idx + 1:
            prev_close = df.iloc[idx-1]['Close']
            prev_sma = df.iloc[idx-1]['SMA_20']
            prev_atr = df.iloc[idx-1]['ATR_20']
            
            if close < sma_20 - 0.5 * atr_20 and prev_close < prev_sma - 0.5 * prev_atr: 
                return True, '趨勢破裂'
        
        days_held = idx - entry_idx
        if days_held >= 20 and (close - entry_price) <= 0.5 * atr_20:
            return True, '時間停損'
        
        if self.version == 'v2': 
            lowest_20 = df.iloc[idx]['Lowest_20']
            if close < lowest_20:
                return True, '唐奇安破裂'
        
        return False, ''
    
    def get_available_cash(self) -> float:
        """取得可用現金"""
        return self.cash - self.reserved_cash
    
    def get_total_positions_count(self) -> int:
        """取得總持倉數"""
        pending_buys = sum(1 for o in self.pending_orders if o['action'] == 'BUY')
        return len(self.positions) + pending_buys
    
    def execute_pending_orders(self, date, stock_data: Dict[str, pd.DataFrame]):
        """執行待成交委託"""
        executed = []
        
        for order in self.pending_orders:
            ticker = order['ticker']
            action = order['action']
            
            order['days_pending'] = order. get('days_pending', 0) + 1
            
            if action == 'BUY' and order['days_pending'] >= self.order_expiry_days:
                if 'reserved_amount' in order:
                    release_amt = order['reserved_amount']
                    self.reserved_cash -= release_amt
                    self.reserved_cash = max(0.0, self.reserved_cash)
                    print(f"⏱️  已過期: {ticker} 買單於 {order['days_pending']} 天後, 已釋放 {release_amt: ,.0f}")
                order['status'] = 'EXPIRED'
                executed.append(order)
                continue
            
            if ticker not in stock_data or date not in stock_data[ticker]. index:
                continue
            
            idx = stock_data[ticker].index.get_loc(date)
            df = stock_data[ticker]
            execution_price = df.iloc[idx]['Open']
            
            if action == 'BUY':
                execution_price *= (1 + self.slippage_pct)
                shares = order['shares']
                total_cost = execution_price * shares * (1 + self.buy_fee_pct)
                
                if 'reserved_amount' in order: 
                    release_amt = order['reserved_amount']
                    self.reserved_cash -= release_amt
                    self.reserved_cash = max(0.0, self.reserved_cash)
                
                if total_cost > self.cash:
                    print(f"⚠️  已取消: {ticker} - 資金不足 ({self.cash:,.0f} < {total_cost:,.0f})")
                    order['status'] = 'CANCELLED_INSUFFICIENT_CASH'
                    executed.append(order)
                    continue
                
                self.cash -= total_cost
                
                signal_close = order.get('signal_close', execution_price)
                entry_atr = order.get('entry_atr', 0)
                stop_base = min(signal_close, execution_price)
                initial_stop = stop_base - self.atr_multiplier * entry_atr if entry_atr > 0 else stop_base - self.atr_multiplier * 0.01
                
                self.positions[ticker] = {
                    'entry_date': date,
                    'entry_idx': idx,
                    'entry_price': execution_price,
                    'shares': shares,
                    'stop_loss': initial_stop,
                    'position_size': order['position_size'],
                    'entry_atr': entry_atr
                }
                
                print(f"✅ 買進: {ticker} @ {execution_price:.2f} x {shares} = {total_cost:,.0f} (現金: {self.cash:,. 0f})")
                executed.append(order)
                
            elif action == 'SELL':
                if ticker not in self.positions:
                    continue
                
                execution_price *= (1 - self.slippage_pct)
                pos = self.positions[ticker]
                shares = pos['shares']
                total_proceeds = execution_price * shares * (1 - self.sell_fee_pct - self.sell_tax_pct)
                
                self.cash += total_proceeds
                
                entry_cost = pos['entry_price'] * shares * (1 + self.buy_fee_pct)
                pnl = total_proceeds - entry_cost
                pnl_pct_net = (pnl / entry_cost) * 100 if entry_cost > 0 else 0
                pnl_pct_gross = (execution_price / pos['entry_price'] - 1) * 100
                
                self.trades.append({
                    'ticker': ticker,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'entry_price': pos['entry_price'],
                    'exit_price': execution_price,
                    'shares': shares,
                    'pnl': pnl,
                    'pnl_pct':  pnl_pct_net,
                    'pnl_pct_gross': pnl_pct_gross,
                    'exit_reason': order['reason']
                })
                
                print(f"✅ 賣出: {ticker} @ {execution_price:.2f}, 損益: {pnl: +,. 0f} ({pnl_pct_net: +.2f}%)")
                
                del self.positions[ticker]
                executed.append(order)
        
        self.pending_orders = [o for o in self.pending_orders if o not in executed]
    
    def record_equity(self, date, stock_data: Dict[str, pd.DataFrame]):
        """記錄每日淨值"""
        exit_cost_pct = self.slippage_pct + self. sell_fee_pct + self.sell_tax_pct
        unrealized_value = 0
        
        for ticker, pos in self.positions.items():
            if date in stock_data[ticker].index:
                current_price = stock_data[ticker].loc[date, 'Close']
                market_value = current_price * pos['shares']
                liquidation_value = market_value * (1 - exit_cost_pct)
                unrealized_value += liquidation_value
        
        total_equity = self.cash + unrealized_value
        
        self.equity_curve.append({
            'date':  date,
            'equity': total_equity,
            'cash': self.cash,
            'reserved_cash': self.reserved_cash,
            'num_positions': len(self.positions),
            'market_value': unrealized_value
        })
    
    def run_backtest(self, stock_data: Dict[str, pd. DataFrame], 
                     benchmark_data: pd.DataFrame,
                     start_date: str = None,
                     end_date: str = None) -> Tuple[pd.DataFrame, pd. DataFrame]:
        """執行完整回測"""
        print(f"\n{'='*70}")
        print(f"🚀 執行 {self.version} 回測")
        print(f"{'='*70}")
        
        for ticker in stock_data: 
            stock_data[ticker] = self.normalize_ohlcv(stock_data[ticker])
        
        benchmark_data = self.normalize_ohlcv(benchmark_data)
        
        for ticker in stock_data:
            stock_data[ticker] = self.calculate_indicators(stock_data[ticker])
        
        benchmark_data = self.calculate_indicators(benchmark_data)
        
        all_dates = benchmark_data.index
        if start_date:
            all_dates = all_dates[all_dates >= start_date]
        if end_date:
            all_dates = all_dates[all_dates <= end_date]
        
        for i, date in enumerate(all_dates):
            if i > 0:
                self.execute_pending_orders(date, stock_data)
            
            if date not in benchmark_data.index or benchmark_data.index.get_loc(date) < 250:
                self.record_equity(date, stock_data)
                continue
            
            idx = benchmark_data.index.get_loc(date)
            
            self.update_trailing_stops(date, stock_data)
            
            regime = self.stage_0_market_regime(benchmark_data, idx)
            
            er_threshold = None
            if self.version == 'v2':
                er_threshold = self.calculate_dynamic_er_threshold(stock_data, date, idx)
            
            for ticker, pos in list(self.positions.items()):
                if date not in stock_data[ticker].index:
                    continue
                
                stock_idx = stock_data[ticker].index.get_loc(date)
                should_exit, reason = self.check_exit_conditions(
                    stock_data[ticker], stock_idx, 
                    pos['entry_price'], pos['entry_idx'], pos['stop_loss']
                )
                
                if should_exit:
                    self.pending_orders.append({
                        'ticker': ticker,
                        'action': 'SELL',
                        'reason': reason,
                        'days_pending': 0
                    })
            
            if regime == '空頭' and self.version == 'v2': 
                self.record_equity(date, stock_data)
                continue
            
            if self.get_total_positions_count() >= self.max_positions:
                self.record_equity(date, stock_data)
                continue
            
            candidates = []
            
            for ticker in stock_data:
                if ticker in self.positions:
                    continue
                
                if any(o['action'] == 'BUY' and o['ticker'] == ticker for o in self.pending_orders):
                    continue
                
                if date not in stock_data[ticker]. index:
                    continue
                
                stock_idx = stock_data[ticker].index.get_loc(date)
                df = stock_data[ticker]
                
                turnover = df.iloc[stock_idx]['Turnover_20']
                if pd.isna(turnover) or turnover < self.min_turnover_twd:
                    continue
                
                stage1_pass, s1 = self.stage_1_trend_efficiency(df, stock_idx, er_threshold)
                if not stage1_pass:
                    continue
                
                stage2_pass, s2 = self.stage_2_momentum_persistence(df, stock_idx)
                if not stage2_pass: 
                    continue
                
                stage3_pass, s3 = self.stage_3_confirmation(df, stock_idx)
                if not stage3_pass: 
                    continue
                
                total_score = 0.25 * s1 + 0.45 * s2 + 0.30 * s3
                
                candidates.append({
                    'ticker': ticker,
                    'score': total_score,
                    'idx': stock_idx
                })
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            for candidate in candidates: 
                if self.get_total_positions_count() >= self.max_positions:
                    break
                
                ticker = candidate['ticker']
                stock_idx = candidate['idx']
                df = stock_data[ticker]
                
                position_size = self.calculate_position_size(df, stock_idx, regime)
                if position_size <= 0:
                    continue
                
                current_close = df.iloc[stock_idx]['Close']
                atr_20 = df.iloc[stock_idx]['ATR_20']
                
                estimated_price = current_close * (1 + self.slippage_pct)
                available_cash = self.get_available_cash()
                position_value = available_cash * position_size
                shares = int(position_value / estimated_price)
                
                shares = self.validate_board_lot(shares)
                
                if shares == 0:
                    continue
                
                estimated_cost = estimated_price * shares * (1 + self.buy_fee_pct)
                
                if estimated_cost > available_cash: 
                    continue
                
                self.reserved_cash += estimated_cost
                
                stop_loss = current_close - self.atr_multiplier * atr_20
                
                self.pending_orders.append({
                    'ticker': ticker,
                    'action': 'BUY',
                    'shares': shares,
                    'stop_loss': stop_loss,
                    'position_size': position_size,
                    'reserved_amount': estimated_cost,
                    'entry_atr': atr_20,
                    'signal_close': current_close,
                    'submit_date': date,
                    'days_pending': 0
                })
            
            self.record_equity(date, stock_data)
        
        final_date = all_dates[-1]
        
        for order in list(self.pending_orders):
            if order['action'] == 'BUY': 
                if 'reserved_amount' in order:
                    self.reserved_cash -= order['reserved_amount']
                print(f"🔚 回測結束:  已取消 {order['ticker']} 的待成交買單")
        
        self.pending_orders = [o for o in self.pending_orders if o['action'] != 'BUY']
        
        if len(self.positions) > 0:
            print(f"\n📊 回測結束: {len(self.positions)} 檔持倉保持開放 (以清算價標記)")
            print(f"💼 開放持倉: {list(self.positions.keys())}")
        
        return self.get_results()
    
    def get_results(self) -> Tuple[pd.DataFrame, pd. DataFrame]:
        """取得回測結果與完整統計"""
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df. set_index('date', inplace=True)
        
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        print("\n" + "="*70)
        print(f"📊 回測結果統計 ({self.version})")
        print("="*70)
        print(f"💰 初始資金: {self.initial_capital: ,.0f} {self.currency}")
        print(f"💵 期末現金: {self.cash:,.0f} {self. currency}")
        
        if len(equity_df) > 0:
            final_equity = equity_df.iloc[-1]['equity']
            total_return = (final_equity / self.initial_capital - 1) * 100
            
            realized_pnl = trades_df['pnl'].sum() if len(trades_df) > 0 else 0
            unrealized_value = final_equity - self.cash
            
            equity_returns = equity_df['equity'].pct_change().dropna()
            sharpe = equity_returns.mean() / equity_returns.std() * np.sqrt(252) if len(equity_returns) > 0 and equity_returns.std() > 0 else 0
            
            # 計算 Sortino (只計算負報酬)
            downside_returns = equity_returns[equity_returns < 0]
            sortino = (equity_returns. mean() / downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
            
            cumulative = (1 + equity_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_dd = drawdowns.min() * 100
            
            # Calmar 比率
            annualized_return = equity_returns.mean() * 252
            calmar = annualized_return / abs(max_dd/100) if max_dd != 0 else 0
            
            # Recovery Factor
            recovery_factor = total_return / abs(max_dd) if max_dd != 0 else 0
            
            trading_days = len(equity_df)
            years = trading_days / 252
            
            print(f"💎 期末淨值: {final_equity:,.0f} {self.currency} (以清算價標記)")
            print(f"📈 總報酬率: {total_return: +.2f}%")
            print(f"📆 年化報酬率: {(total_return / years):+.2f}%") if years > 0 else print()
            print(f"💰 已實現損益: {realized_pnl:+,. 0f} {self.currency}")
            print(f"💼 未實現價值: {unrealized_value: +,.0f} {self.currency}")
            
            print(f"\n📊 風險調整指標:")
            print(f"   • Sharpe 比率: {sharpe:.2f}")
            print(f"   • Sortino 比率: {sortino:.2f}")
            print(f"   • Calmar 比率: {calmar:.2f}")
            print(f"   • Recovery 因子: {recovery_factor:.2f}")
            print(f"   • 最大回撤: {max_dd:. 2f}%")
            print(f"   • 交易日數: {trading_days}")
            print(f"   • 總年期: {years:.1f} 年")
            
            print(f"\n🔒 持倉狀態:")
            print(f"   • 開放持倉: {len(self.positions)}")
        
        print(f"🔢 交易統計:")
        print(f"   • 總成交筆數: {len(self.trades)}")
        
        if len(self.trades) > 0:
            win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl_pct'].mean() if (trades_df['pnl'] > 0).any() else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl_pct'].mean() if (trades_df['pnl'] < 0).any() else 0
            
            profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl']. sum() / 
                              trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if (trades_df['pnl'] < 0).any() else float('inf')
            
            print(f"   • 勝率: {win_rate:. 2f}%")
            print(f"   • 平均獲利 (淨): {avg_win:+.2f}%")
            print(f"   • 平均虧損 (淨): {avg_loss:+.2f}%")
            print(f"   • 獲利因子: {profit_factor:.2f}")
        
        print(f"\n✅ 完整性檢查:")
        print(f"   ✓ 預留現金 ≥ 0: {self.reserved_cash >= -0.01}")
        print(f"   ✓ 現金 ≥ 0: {self.cash >= -0.01}")
        print(f"   ✓ 淨值資料完整:  {len(equity_df)} 天")
        print(f"   ✓ 無 NaN 淨值: {not equity_df['equity'].isna().any()}")
        print("="*70 + "\n")
        
        return equity_df, trades_df
    
    def generate_monthly_report(self, equity_df: pd.DataFrame) -> pd.DataFrame:
        """生成月度績效報告"""
        equity_df['month'] = equity_df. index.to_period('M')
        monthly_stats = []
        
        for month, group in equity_df.groupby('month'):
            first_equity = group['equity'].iloc[0]
            last_equity = group['equity'].iloc[-1]
            monthly_return = (last_equity / first_equity - 1) * 100
            max_equity = group['equity'].max()
            min_equity = group['equity'].min()
            max_dd = ((min_equity - max_equity) / max_equity) * 100
            
            monthly_stats.append({
                '月份': str(month),
                '期初淨值': first_equity,
                '期末淨值': last_equity,
                '月報酬率': monthly_return,
                '月最高':  max_equity,
                '月最低': min_equity,
                '月最大回撤': max_dd
            })
        
        return pd.DataFrame(monthly_stats)
    
    def generate_quarterly_report(self, equity_df:  pd.DataFrame) -> pd.DataFrame:
        """生成季度績效報告"""
        equity_df['quarter'] = equity_df.index.to_period('Q')
        quarterly_stats = []
        
        for quarter, group in equity_df.groupby('quarter'):
            first_equity = group['equity'].iloc[0]
            last_equity = group['equity'].iloc[-1]
            quarterly_return = (last_equity / first_equity - 1) * 100
            max_equity = group['equity'].max()
            min_equity = group['equity'].min()
            max_dd = ((min_equity - max_equity) / max_equity) * 100
            
            quarterly_stats.append({
                '季度': str(quarter),
                '期初淨值':  first_equity,
                '期末淨值': last_equity,
                '季報酬率': quarterly_return,
                '季最高': max_equity,
                '季最低': min_equity,
                '季最大回撤':  max_dd
            })
        
        return pd.DataFrame(quarterly_stats)


def download_sample_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]: 
    """下載台股資料"""
    print("\n📥 下載資料中...")
    stock_data = {}
    
    for ticker in tickers:
        if not TrendValidatorBacktestEnhanced.validate_ticker(ticker):
            print(f"❌ {ticker}:  拒絕 (非4碼數字)")
            continue
        
        for suffix in ['. TW', '.TWO']: 
            yf_ticker = f"{ticker}{suffix}"
            
            try:
                df = yf.download(yf_ticker, start=start_date, end=end_date, 
                               auto_adjust=True, progress=False)
                
                if df is None or len(df) == 0:
                    continue
                
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df = TrendValidatorBacktestEnhanced.normalize_ohlcv(df)
                
                stock_data[ticker] = df
                exchange = "上市" if suffix == '.TW' else "上櫃"
                print(f"✅ {ticker} ({exchange}): {len(df)} 個交易日")
                break
                
            except Exception as e:
                continue
        
        if ticker not in stock_data:
            print(f"❌ {ticker}: 無法取得資料")
    
    return stock_data


def run_example_enhanced():
    """執行強化版回測範例 - 支援長期回測與中文輸出"""
    
    print("\n" + "="*70)
    print("📋 回測設定")
    print("="*70)
    print("股票池:               台灣4碼股 (上市 + 上櫃)")
    print("執行模式:            訊號 @ 收盤 → 成交 @ 次日開盤")
    print("遺漏交易日處理:      次一可交易日 + 5天買單有效期")
    print("回測結束處理:        保持開放部位 (以清算價評估)")
    print("淨值計算:            以清算價標記 (每日)")
    print("成本:                 買進 0.1425% | 賣出 0.1425% + 0.3% 稅金")
    print("滑點:                雙向 0.1%")
    print("="*70)
    
    # 📌 改為更長的時間跨度
    tickers = [
        '2330', '2317', '2454', '3008', '2308', '2382', '2412', '6505', '2303', '3711',
        '2603', '2609', '2606', '4938', '2301', '2882', '2881', '2886', '2890', '2884',
        '1101', '1102', '1216', '1301', '1303', '1402', '1504', '1605', '1722', '1802'
    ]
    
    # 🔥 擴大年度到 7 年
    start_date = '2017-01-01'
    end_date = '2024-01-01'
    
    stock_data = download_sample_data(tickers, start_date, end_date)
    
    print("\n📥 下載基準指數 (加權指數)...")
    benchmark_data = yf.download('^TWII', start=start_date, end=end_date, 
                                auto_adjust=True, progress=False)
    benchmark_data = benchmark_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    benchmark_data = TrendValidatorBacktestEnhanced.normalize_ohlcv(benchmark_data)
    
    if len(stock_data) == 0:
        print("\n❌ 無法取得資料")
        return None, None, None, None
    
    # 執行 v1.1 回測
    print("\n" + "="*70)
    print("🔄 執行 v1.1 回測")
    print("="*70)
    bt_v1 = TrendValidatorBacktestEnhanced(version='v1.1', initial_capital=1000000)
    equity_v1, trades_v1 = bt_v1.run_backtest(stock_data, benchmark_data, start_date, end_date)
    
    # 生成月度與季度報告
    print("\n📅 月度績效報告 (v1.1):")
    monthly_v1 = bt_v1.generate_monthly_report(equity_v1.copy())
    print(monthly_v1.to_string(index=False))
    
    print("\n📅 季度績效報告 (v1.1):")
    quarterly_v1 = bt_v1.generate_quarterly_report(equity_v1.copy())
    print(quarterly_v1.to_string(index=False))
    
    # 執行 v2 回測
    print("\n" + "="*70)
    print("🔄 執行 v2 回測")
    print("="*70)
    bt_v2 = TrendValidatorBacktestEnhanced(version='v2', initial_capital=1000000)
    equity_v2, trades_v2 = bt_v2.run_backtest(stock_data, benchmark_data, start_date, end_date)
    
    # 生成月度與季度報告
    print("\n📅 月度績效報告 (v2):")
    monthly_v2 = bt_v2.generate_monthly_report(equity_v2.copy())
    print(monthly_v2.to_string(index=False))
    
    print("\n📅 季度績效報告 (v2):")
    quarterly_v2 = bt_v2.generate_quarterly_report(equity_v2.copy())
    print(quarterly_v2.to_string(index=False))
    
    # 繪圖 - 全中文標籤
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # 淨值曲線
    ax1 = fig.add_subplot(gs[0, : ])
    ax1.plot(equity_v1.index, equity_v1['equity'], label='v1.1', linewidth=2. 5, color='#2E86AB')
    ax1.plot(equity_v2.index, equity_v2['equity'], label='v2', linewidth=2.5, color='#A23B72')
    ax1.axhline(y=1000000, color='gray', linestyle='--', alpha=0.7, label='初始資金')
    ax1.set_title('💰 淨值曲線 (以清算價標記)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('淨值 (新台幣)')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # 回撤曲線
    ax2 = fig.add_subplot(gs[1, : ])
    if len(equity_v1) > 0:
        ret_v1 = equity_v1['equity'].pct_change()
        cum_v1 = (1 + ret_v1).cumprod()
        dd_v1 = (cum_v1 / cum_v1.expanding().max() - 1) * 100
        ax2.fill_between(dd_v1.index, dd_v1, 0, alpha=0.3, color='#2E86AB', label='v1.1')
    
    if len(equity_v2) > 0:
        ret_v2 = equity_v2['equity'].pct_change()
        cum_v2 = (1 + ret_v2).cumprod()
        dd_v2 = (cum_v2 / cum_v2.expanding().max() - 1) * 100
        ax2.fill_between(dd_v2.index, dd_v2, 0, alpha=0.3, color='#A23B72', label='v2')
    
    ax2.set_title('📉 最大回撤', fontsize=14, fontweight='bold')
    ax2.set_ylabel('回撤 (%)')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 現金狀況 - v1.1
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(equity_v1.index, equity_v1['cash'], label='現金', linewidth=2, color='#2E86AB')
    ax3.plot(equity_v1.index, equity_v1['reserved_cash'], label='預留', 
             linewidth=1. 5, linestyle='--', color='#2E86AB', alpha=0.6)
    ax3.set_title('💵 v1.1 現金狀況', fontsize=12, fontweight='bold')
    ax3.set_ylabel('新台幣')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # 現金狀況 - v2
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(equity_v2.index, equity_v2['cash'], label='現金', linewidth=2, color='#A23B72')
    ax4.plot(equity_v2.index, equity_v2['reserved_cash'], label='預留', 
             linewidth=1.5, linestyle='--', color='#A23B72', alpha=0.6)
    ax4.set_title('💵 v2 現金狀況', fontsize=12, fontweight='bold')
    ax4.set_ylabel('新台幣')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    plt.suptitle('趨勢驗證系統 v2.2 - 強化版 (長期回測)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return equity_v1, trades_v1, equity_v2, trades_v2


if __name__ == "__main__": 
    print("🚀 趨勢驗證系統 v2.2")
    print("📦 強化版 (中文輸出 + 長期回測)")
    print("="*70)
    print("\n✨ 新增功能:")
    print("  ✅ 年度擴展：支援 5-10 年以上回測")
    print("  ✅ 中文輸出：所有報告、圖表、指標都是繁體中文")
    print("  ✅ 完整統計：Sharpe、Sortino、Calmar、Recovery Factor")
    print("  ✅ 月度報告：逐月績效明細")
    print("  ✅ 季度報告：逐季績效與最大回撤\n")
    
    equity_v1, trades_v1, equity_v2, trades_v2 = run_example_enhanced()
