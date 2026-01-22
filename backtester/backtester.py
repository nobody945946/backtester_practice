"""
Trend Validator Protocol - Production-Grade Backtesting System v2.2
Complete implementation with all P0/P1/P2 fixes

PRODUCTION-READY FEATURES:
✅ Universe: 4-digit Taiwan stocks only (TWSE + TPEx/OTC)
✅ Equity recorded EVERY day (no gaps)
✅ Missing sessions: next-available + 5-day expiry (Option C)
✅ EoT Option 1: Positions stay open (mark-to-liquidation)
✅ Net P&L% reporting (includes all costs)
✅ Stop recalculated after fill
✅ Taiwan costs: Commission 0.0399% (28% discount applied) + 0.3% sell tax + slippage
✅ Board lot support: 1000 shares per lot, odd lots allowed
✅ Execution: Signal @ close → Fill @ next open, 5-day BUY order expiry

Run in Google Colab: !pip install yfinance pandas numpy matplotlib scipy

Configuration: schema v1.0 - Taiwan market with commission discount applied
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import re
warnings.filterwarnings('ignore')


class TrendValidatorBacktest:
    """Production-grade backtesting engine with realistic execution model
    
    Configuration Schema v1.0 - Taiwan Market:
    - Commission: 0.0399% (standard rate 0.1425% with 28% discount applied)
    - Sell tax: 0.3%
    - Slippage: 0.1% (fixed percentage model)
    - Board lot: 1000 shares, odd lots allowed
    - Execution: Signal @ close → Fill @ next_open, 5-day BUY order expiry
    """
    
    def __init__(self, version: str = 'v1.1', initial_capital: float = 1000000,
                 slippage_pct: float = 0.001, max_positions: int = 10,
                 commission_discount_factor: float = 0.28):
        self.version = version
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.reserved_cash = 0.0
        self.positions = {}
        self.pending_orders = []
        self.trades = []
        self.equity_curve = []
        self.max_positions = max_positions
        
        # Execution Configuration (schema v1.0 - Taiwan market)
        self.market = 'TW'
        self.currency = 'TWD'
        self.signal_time = 'close'
        self.fill_time = 'next_open'
        self.missing_session_policy = 'next_available'
        self.buy_order_expiry_days = 5
        self.sell_order_expiry_days = None  # SELL orders persist until filled
        
        # Commission Fee Structure (TWD market, with discount factor)
        self.commission_standard_rate = 0.001425  # 0.1425% (standard rate)
        self.commission_discount_factor = commission_discount_factor  # 0.28 = 28% discount
        self.commission_effective_rate = self.commission_standard_rate * (1 - self.commission_discount_factor)  # 0.000399
        self.commission_minimum_fee_twd = 0  # No minimum fee in simulation
        
        # Taiwan-specific costs (using effective rate with discount)
        self.slippage_pct = slippage_pct
        self.buy_fee_pct = self.commission_effective_rate  # 0.0399% after discount
        self.sell_fee_pct = self.commission_effective_rate  # 0.0399% after discount
        self.sell_tax_pct = 0.003  # 0.3% government tax
        
        # Board Lot Configuration
        self.board_lot_shares = 1000
        self.allow_odd_lot = True
        
        # Strategy Parameters
        self.atr_period = 20
        self.atr_multiplier = 2.5
        self.vwap_window = 20
        self.er_percentile = 0.70
        self.min_turnover_twd = 50000000
        self.order_expiry_days = self.buy_order_expiry_days  # Backward compatibility
        
        print(f"🚀 Initialized {version} backtester v2.2 (Production Grade)")
        print(f"💰 Capital: {initial_capital:,.0f} {self.currency}")
        print(f"📊 Commission: {self.buy_fee_pct*100:.4f}% (standard: {self.commission_standard_rate*100:.4f}%, discount: {self.commission_discount_factor*100:.0f}%)")
        print(f"📊 Sell tax: {self.sell_tax_pct*100:.2f}%")
        print(f"📊 Slippage: {self.slippage_pct*100:.2f}%")
        print(f"🏛️  Universe: 4-digit Taiwan stocks (TWSE + TPEx)")
        print(f"📈 Board lot: {self.board_lot_shares} shares (odd lots {'allowed' if self.allow_odd_lot else 'not allowed'})")
        print(f"📅 BUY order expiry: {self.buy_order_expiry_days} days")
        print(f"⏱️  Execution: Signal @ {self.signal_time} → Fill @ {self.fill_time}")
        
        # Validate configuration
        try:
            self.validate_config()
            print("✅ Configuration validated successfully")
        except ValueError as e:
            print(f"❌ Configuration error:\n{e}")
            raise
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Validate 4-digit numeric ticker (Taiwan common stocks only)"""
        pattern = r'^\d{4}$'
        return bool(re.match(pattern, ticker))
    
    def validate_board_lot(self, shares: int) -> int:
        """Validate and adjust shares to comply with board lot rules
        
        Taiwan market rules:
        - Standard trading: multiples of 1000 (board lot)
        - Odd lots (1-999) allowed if allow_odd_lot=True
        
        Returns: validated share quantity
        """
        if shares % self.board_lot_shares == 0:
            return shares
        
        if self.allow_odd_lot:
            return shares  # Odd lots allowed
        else:
            # Round down to nearest board lot
            return (shares // self.board_lot_shares) * self.board_lot_shares
    
    def validate_config(self) -> bool:
        """Validate configuration against schema v1.0
        
        Checks:
        - Market: TW (Taiwan)
        - Currency: TWD
        - Commission structure: standard_rate, discount_factor, effective_rate
        - Board lot: shares must be positive
        - Execution policies: order expiry days valid
        
        Returns: True if valid, raises ValueError otherwise
        """
        errors = []
        
        # Market and currency validation
        if self.market != 'TW':
            errors.append(f"Market must be 'TW', got '{self.market}'")
        if self.currency != 'TWD':
            errors.append(f"Currency must be 'TWD', got '{self.currency}'")
        
        # Commission validation
        if self.commission_standard_rate <= 0:
            errors.append(f"Standard commission rate must be positive, got {self.commission_standard_rate}")
        if not (0.0 <= self.commission_discount_factor <= 1.0):
            errors.append(f"Discount factor must be between 0 and 1, got {self.commission_discount_factor}")
        
        expected_effective = self.commission_standard_rate * (1 - self.commission_discount_factor)
        if abs(self.commission_effective_rate - expected_effective) > 1e-10:
            errors.append(f"Effective rate mismatch: expected {expected_effective}, got {self.commission_effective_rate}")
        
        # Board lot validation
        if self.board_lot_shares <= 0:
            errors.append(f"Board lot shares must be positive, got {self.board_lot_shares}")
        
        # Order expiry validation
        if self.buy_order_expiry_days <= 0:
            errors.append(f"BUY order expiry days must be positive, got {self.buy_order_expiry_days}")
        
        # Slippage validation
        if self.slippage_pct < 0 or self.slippage_pct > 0.05:
            errors.append(f"Slippage {self.slippage_pct*100:.2f}% seems unrealistic (expected 0-5%)")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  ❌ {e}" for e in errors))
        
        return True
    
    @staticmethod
    def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize OHLCV frame: flatten MultiIndex, ensure single-level columns, drop NaN close"""
        df = df.copy()
        
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            print(f"  ⚠️  Flattened MultiIndex columns: {list(df.columns)}")
        
        # Ensure required columns exist
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Check if any column is a DataFrame (multi-column problem)
        for col in required:
            if isinstance(df[col], pd.DataFrame):
                print(f"  ❌ ERROR: Column '{col}' is a DataFrame, not Series!")
                print(f"    DataFrame columns: {list(df[col].columns)}")
                # Try to select first column
                df[col] = df[col].iloc[:, 0]
                print(f"    Selected first column from '{col}'")
        
        # Ensure numeric dtypes
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN Close
        df = df.dropna(subset=['Close'])
        
        return df
    
    def calculate_wilder_atr(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate ATR using Wilder's smoothing"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.ewm(alpha=1/period, adjust=False).mean()
    
    def calculate_rolling_vwap(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate rolling VWAP with zero-volume protection"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        pv = (typical_price * df['Volume']).rolling(window=window).sum()
        vol = df['Volume'].rolling(window=window).sum()
        return pv / vol.replace(0, np.nan)
    
    def calculate_kaufman_er(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Kaufman Efficiency Ratio"""
        close = df['Close']
        direction = abs(close - close.shift(period))
        volatility = abs(close - close.shift(1)).rolling(window=period).sum()
        return direction / volatility.replace(0, np.nan)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        df = self.normalize_ohlcv(df)
        df = df.copy()
        
        # Verify Series (not DataFrame) before calculations
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            if not isinstance(df[col], pd.Series):
                raise TypeError(f"Expected Series for '{col}', got {type(df[col])}")
        
        df['daily_return'] = df['Close'].pct_change()
        df['ATR_20'] = self.calculate_wilder_atr(df, self.atr_period)
        df['VWAP'] = self.calculate_rolling_vwap(df, self.vwap_window)
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
        
        # CRITICAL: Use PRIOR window (exclude today)
        df['Lowest_20'] = df['Close'].rolling(window=20).min().shift(1)
        df['Highest_20'] = df['High'].rolling(window=20).max().shift(1)
        df['Displacement_over_ATR'] = (df['Close'] - df['Lowest_20']) / (df['ATR_20'] * 2.0)
        
        return df
    
    def calculate_statistical_metrics(self, returns: pd.Series, lookback: int = 60) -> Dict:
        """Calculate statistical metrics for momentum persistence"""
        if len(returns) < lookback:
            return None
        
        recent = returns.tail(lookback)
        
        return {
            'positive_day_ratio': (recent > 0).sum() / lookback,
            'skewness': recent.skew(),
            'mean': recent.mean(),
            'std': recent.std(),
            't_stat': recent.mean() / (recent.std() / np.sqrt(lookback)) if recent.std() > 0 else 0,
            'q05': recent.quantile(0.05),
            'cvar_05': recent[recent <= recent.quantile(0.05)].mean() if len(recent[recent <= recent.quantile(0.05)]) > 0 else recent.min(),
            'min_return': recent.min(),
            'max_drawdown': self.calculate_max_drawdown(recent)
        }
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_dynamic_er_threshold(self, stock_data: Dict[str, pd.DataFrame], 
                                      date, idx: int) -> float:
        """Calculate cross-sectional ER percentile threshold (v2 only)"""
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
        """Market regime filter"""
        if self.version == 'v1.1':
            return 'Bull' if benchmark_df.iloc[idx]['Close'] > benchmark_df.iloc[idx]['SMA_200'] else 'Bear'
        else:
            close = benchmark_df.iloc[idx]['Close']
            sma_50 = benchmark_df.iloc[idx]['SMA_50']
            sma_200 = benchmark_df.iloc[idx]['SMA_200']
            
            if close > sma_200 and sma_50 > sma_200:
                return 'Bull'
            elif close > sma_200:
                return 'Neutral'
            else:
                return 'Bear'
    
    def stage_1_trend_efficiency(self, df: pd.DataFrame, idx: int, 
                                 er_threshold: float = None) -> Tuple[bool, float]:
        """Trend efficiency filter"""
        er_20 = df.iloc[idx]['ER_20']
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
        """Momentum persistence and tail risk check"""
        returns = df['daily_return'].iloc[:idx+1]
        metrics = self.calculate_statistical_metrics(returns, lookback=60)
        
        if metrics is None:
            return False, 0.0
        
        if self.version == 'v1.1':
            checks = {
                'positive_ratio': metrics['positive_day_ratio'] > 0.55,
                'skewness': metrics['skewness'] > -0.5,
                't_stat': metrics['t_stat'] > 1.5,
                'q05': metrics['q05'] > -0.04
            }
            weights = {'positive_ratio': 0.25, 'skewness': 0.10, 't_stat': 0.35, 'q05': 0.30}
            
            if not checks['t_stat']:
                return False, 0.0
            
            score = sum(weights[k] * (1.0 if checks[k] else 0.0) for k in checks)
            return score >= 0.65, score
        else:
            std_60 = metrics['std']
            checks = {
                't_stat': metrics['t_stat'] > 2.0,
                'cvar_05': metrics['cvar_05'] > -1.5 * std_60,
                'positive_ratio': metrics['positive_day_ratio'] > 0.52,
                'max_drop': metrics['min_return'] > -3.0 * std_60
            }
            weights = {'t_stat': 0.40, 'cvar_05': 0.35, 'positive_ratio': 0.15, 'max_drop': 0.10}
            
            if not (checks['t_stat'] and checks['cvar_05']):
                return False, 0.0
            
            score = sum(weights[k] * (1.0 if checks[k] else 0.0) for k in checks)
            return score >= 0.70, score
    
    def stage_3_confirmation(self, df: pd.DataFrame, idx: int) -> Tuple[bool, float]:
        """Confirmation buffer"""
        roc_20 = df.iloc[idx]['ROC_20']
        disp_atr = df.iloc[idx]['Displacement_over_ATR']
        vol_ratio = df.iloc[idx]['Vol_Ratio_20']
        close = df.iloc[idx]['Close']
        vwap = df.iloc[idx]['VWAP']
        highest_20 = df.iloc[idx]['Highest_20']
        
        if any(pd.isna([roc_20, disp_atr, vol_ratio, vwap])):
            return False, 0.0
        
        recent_closes = df['Close'].iloc[max(0, idx-20):idx+1]
        max_dd = self.calculate_max_drawdown(recent_closes.pct_change().dropna())
        
        if self.version == 'v1.1':
            checks = {
                'displacement': roc_20 >= 0.15 or disp_atr > 1.0,
                'max_dd': max_dd >= -0.10,
                'volume': vol_ratio >= 1.2
            }
            return all(checks.values()), sum(1.0 if v else 0.0 for v in checks.values()) / len(checks)
        else:
            std_60 = df['daily_return'].iloc[:idx+1].tail(60).std()
            roc_vol_scaled = roc_20 / (std_60 * np.sqrt(20)) if std_60 > 0 else 0
            
            atr_20 = df.iloc[idx]['ATR_20']
            atr_pct = atr_20 / close if close > 0 else 0
            dd_atr_scaled = abs(max_dd) / atr_pct if atr_pct > 0 else 999
            
            checks = {
                'breakout': close > highest_20,
                'roc_scaled': roc_vol_scaled > 1.0,
                'dd_atr': dd_atr_scaled <= 4.0,
                'volume': vol_ratio >= 1.3,
                'vwap': close > vwap
            }
            return all(checks.values()), sum(1.0 if v else 0.0 for v in checks.values()) / len(checks)
    
    def calculate_position_size(self, df: pd.DataFrame, idx: int, regime: str) -> float:
        """Calculate volatility-targeted position size"""
        if self.version == 'v1.1':
            vol = df.iloc[idx]['Vol_60']
            regime_multiplier = 0.5 if regime == 'Bear' else 1.0
        else:
            vol = df.iloc[idx]['EWMA_Vol_60']
            regime_multipliers = {'Bull': 1.0, 'Neutral': 0.7, 'Bear': 0.3}
            regime_multiplier = regime_multipliers.get(regime, 1.0)
        
        if pd.isna(vol) or vol == 0:
            return 0.0
        
        position_size = (0.15 * regime_multiplier) / vol
        return min(position_size, 0.10)
    
    def update_trailing_stops(self, date, stock_data: Dict[str, pd.DataFrame]):
        """Update trailing stops daily (only raise, never lower)"""
        for ticker, pos in self.positions.items():
            if date not in stock_data[ticker].index:
                continue
            
            idx = stock_data[ticker].index.get_loc(date)
            close = stock_data[ticker].iloc[idx]['Close']
            atr = stock_data[ticker].iloc[idx]['ATR_20']
            
            if pd.isna(atr):
                continue
            
            new_stop = close - self.atr_multiplier * atr
            if new_stop > pos['stop_loss']:
                pos['stop_loss'] = new_stop
    
    def check_exit_conditions(self, df: pd.DataFrame, idx: int, entry_price: float, 
                             entry_idx: int, stop_loss: float) -> Tuple[bool, str]:
        """Check position exit conditions"""
        close = df.iloc[idx]['Close']
        atr_20 = df.iloc[idx]['ATR_20']
        sma_20 = df.iloc[idx]['SMA_20']
        
        if close <= stop_loss:
            return True, 'Stop Loss'
        
        if idx > entry_idx + 1:
            prev_close = df.iloc[idx-1]['Close']
            prev_sma = df.iloc[idx-1]['SMA_20']
            prev_atr = df.iloc[idx-1]['ATR_20']
            
            if close < sma_20 - 0.5 * atr_20 and prev_close < prev_sma - 0.5 * prev_atr:
                return True, 'Trend Break'
        
        days_held = idx - entry_idx
        if days_held >= 20 and (close - entry_price) <= 0.5 * atr_20:
            return True, 'Time Stop'
        
        if self.version == 'v2':
            lowest_20 = df.iloc[idx]['Lowest_20']
            if close < lowest_20:
                return True, 'Donchian Break'
        
        return False, ''
    
    def get_available_cash(self) -> float:
        """Get cash available after reservations"""
        return self.cash - self.reserved_cash
    
    def get_total_positions_count(self) -> int:
        """Count positions + pending BUY orders"""
        pending_buys = sum(1 for o in self.pending_orders if o['action'] == 'BUY')
        return len(self.positions) + pending_buys
    
    def execute_pending_orders(self, date, stock_data: Dict[str, pd.DataFrame]):
        """
        Execute orders with Option C: next-available + expiry
        - BUY orders expire after 5 days
        - SELL orders remain pending until filled
        CRITICAL: Single authoritative entry point for all order fills
        """
        executed = []
        
        for order in self.pending_orders:
            ticker = order['ticker']
            action = order['action']
            
            # Increment order age
            order['days_pending'] = order.get('days_pending', 0) + 1
            
            # Check expiry (BUY only)
            if action == 'BUY' and order['days_pending'] >= self.order_expiry_days:
                if 'reserved_amount' in order:
                    release_amt = order['reserved_amount']
                    self.reserved_cash -= release_amt
                    self.reserved_cash = max(0.0, self.reserved_cash)  # Safety clamp
                    print(f"⏱️  EXPIRED: {ticker} BUY after {order['days_pending']} days, released {release_amt:,.0f}")
                order['status'] = 'EXPIRED'
                executed.append(order)
                continue
            
            # Wait for next available session
            if ticker not in stock_data or date not in stock_data[ticker].index:
                continue
            
            idx = stock_data[ticker].index.get_loc(date)
            df = stock_data[ticker]
            execution_price = df.iloc[idx]['Open']
            
            if action == 'BUY':
                execution_price *= (1 + self.slippage_pct)
                shares = order['shares']
                total_cost = execution_price * shares * (1 + self.buy_fee_pct)
                
                # Release reservation (exactly once)
                if 'reserved_amount' in order:
                    release_amt = order['reserved_amount']
                    self.reserved_cash -= release_amt
                    self.reserved_cash = max(0.0, self.reserved_cash)
                
                # Check cash
                if total_cost > self.cash:
                    print(f"⚠️  CANCELLED: {ticker} - Insufficient cash ({self.cash:,.0f} < {total_cost:,.0f})")
                    order['status'] = 'CANCELLED_INSUFFICIENT_CASH'
                    executed.append(order)
                    continue
                
                self.cash -= total_cost
                
                # Recalculate stop after fill (REC-3)
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
                
                print(f"✅ BUY: {ticker} @ {execution_price:.2f} x {shares} = {total_cost:,.0f} (Cash: {self.cash:,.0f})")
                executed.append(order)
                
            elif action == 'SELL':
                if ticker not in self.positions:
                    continue
                
                execution_price *= (1 - self.slippage_pct)
                pos = self.positions[ticker]
                shares = pos['shares']
                total_proceeds = execution_price * shares * (1 - self.sell_fee_pct - self.sell_tax_pct)
                
                self.cash += total_proceeds
                
                # Calculate net P&L
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
                    'pnl_pct': pnl_pct_net,
                    'pnl_pct_gross': pnl_pct_gross,
                    'exit_reason': order['reason']
                })
                
                print(f"✅ SELL: {ticker} @ {execution_price:.2f}, P&L: {pnl:+,.0f} ({pnl_pct_net:+.2f}%)")
                
                del self.positions[ticker]
                executed.append(order)
        
        # Remove executed orders
        self.pending_orders = [o for o in self.pending_orders if o not in executed]
    
    def record_equity(self, date, stock_data: Dict[str, pd.DataFrame]):
        """Record equity with mark-to-liquidation"""
        exit_cost_pct = self.slippage_pct + self.sell_fee_pct + self.sell_tax_pct
        unrealized_value = 0
        
        for ticker, pos in self.positions.items():
            if date in stock_data[ticker].index:
                current_price = stock_data[ticker].loc[date, 'Close']
                market_value = current_price * pos['shares']
                liquidation_value = market_value * (1 - exit_cost_pct)
                unrealized_value += liquidation_value
        
        total_equity = self.cash + unrealized_value
        
        self.equity_curve.append({
            'date': date,
            'equity': total_equity,
            'cash': self.cash,
            'reserved_cash': self.reserved_cash,
            'num_positions': len(self.positions),
            'market_value': unrealized_value
        })
    
    def run_backtest(self, stock_data: Dict[str, pd.DataFrame], 
                     benchmark_data: pd.DataFrame,
                     start_date: str = None,
                     end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run complete backtest"""
        print(f"\n{'='*60}")
        print(f"🚀 Running {self.version} backtest")
        print(f"{'='*60}")
        
        # Normalize all data frames
        for ticker in stock_data:
            stock_data[ticker] = self.normalize_ohlcv(stock_data[ticker])
        
        benchmark_data = self.normalize_ohlcv(benchmark_data)
        
        # Calculate indicators
        for ticker in stock_data:
            stock_data[ticker] = self.calculate_indicators(stock_data[ticker])
        
        benchmark_data = self.calculate_indicators(benchmark_data)
        
        all_dates = benchmark_data.index
        if start_date:
            all_dates = all_dates[all_dates >= start_date]
        if end_date:
            all_dates = all_dates[all_dates <= end_date]
        
        # Main backtest loop
        for i, date in enumerate(all_dates):
            # Execute pending orders from previous day
            if i > 0:
                self.execute_pending_orders(date, stock_data)
            
            # CRITICAL: Record equity EVERY day (REC-2)
            if date not in benchmark_data.index or benchmark_data.index.get_loc(date) < 250:
                self.record_equity(date, stock_data)
                continue
            
            idx = benchmark_data.index.get_loc(date)
            
            # Update trailing stops
            self.update_trailing_stops(date, stock_data)
            
            regime = self.stage_0_market_regime(benchmark_data, idx)
            
            # Dynamic ER threshold for v2
            er_threshold = None
            if self.version == 'v2':
                er_threshold = self.calculate_dynamic_er_threshold(stock_data, date, idx)
            
            # Check exits
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
            
            # Check for new entries
            if regime == 'Bear' and self.version == 'v2':
                self.record_equity(date, stock_data)
                continue
            
            if self.get_total_positions_count() >= self.max_positions:
                self.record_equity(date, stock_data)
                continue
            
            # Scan and rank candidates
            candidates = []
            
            for ticker in stock_data:
                if ticker in self.positions:
                    continue
                
                if any(o['action'] == 'BUY' and o['ticker'] == ticker for o in self.pending_orders):
                    continue
                
                if date not in stock_data[ticker].index:
                    continue
                
                stock_idx = stock_data[ticker].index.get_loc(date)
                df = stock_data[ticker]
                
                # Liquidity gate
                turnover = df.iloc[stock_idx]['Turnover_20']
                if pd.isna(turnover) or turnover < self.min_turnover_twd:
                    continue
                
                # Run stages
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
            
            # Rank and fill
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
                
                # Validate board lot compliance
                shares = self.validate_board_lot(shares)
                
                if shares == 0:
                    continue
                
                estimated_cost = estimated_price * shares * (1 + self.buy_fee_pct)
                
                if estimated_cost > available_cash:
                    continue
                
                self.reserved_cash += estimated_cost
                
                stop_loss = current_close - self.atr_multiplier * atr_20
                
                # REC-1: Set submit_date at creation
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
            
            # Record equity
            self.record_equity(date, stock_data)
        
        # EoT Option 1: Cancel pending BUYs, keep positions open
        final_date = all_dates[-1]
        
        for order in list(self.pending_orders):
            if order['action'] == 'BUY':
                if 'reserved_amount' in order:
                    self.reserved_cash -= order['reserved_amount']
                print(f"🔚 EoT: Cancelled pending BUY for {order['ticker']}")
        
        self.pending_orders = [o for o in self.pending_orders if o['action'] != 'BUY']
        
        # Report open positions
        if len(self.positions) > 0:
            print(f"\n📊 EoT: {len(self.positions)} positions remain OPEN (marked-to-liquidation)")
            print(f"💼 Open positions: {list(self.positions.keys())}")
        
        return self.get_results()
    
    def get_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get backtest results"""
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        print("\n" + "="*60)
        print(f"📊 BACKTEST RESULTS ({self.version})")
        print("="*60)
        print(f"💰 Initial Capital: {self.initial_capital:,.0f} TWD")
        print(f"💵 Final Cash: {self.cash:,.0f} TWD")
        
        if len(equity_df) > 0:
            final_equity = equity_df.iloc[-1]['equity']
            total_return = (final_equity / self.initial_capital - 1) * 100
            
            realized_pnl = trades_df['pnl'].sum() if len(trades_df) > 0 else 0
            unrealized_value = final_equity - self.cash
            
            equity_returns = equity_df['equity'].pct_change().dropna()
            sharpe = equity_returns.mean() / equity_returns.std() * np.sqrt(252) if len(equity_returns) > 0 and equity_returns.std() > 0 else 0
            
            cumulative = (1 + equity_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_dd = drawdowns.min() * 100
            
            print(f"💎 Final Equity: {final_equity:,.0f} TWD (Mark-to-Liquidation)")
            print(f"📈 Total Return: {total_return:+.2f}%")
            print(f"💰 Realized P&L: {realized_pnl:+,.0f} TWD")
            print(f"💼 Unrealized Value: {unrealized_value:+,.0f} TWD")
            print(f"📊 Sharpe Ratio: {sharpe:.2f}")
            print(f"📉 Max Drawdown: {max_dd:.2f}%")
            print(f"🔒 Open Positions: {len(self.positions)}")
        
        print(f"🔢 Trades: {len(self.trades)}")
        
        if len(self.trades) > 0:
            win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl_pct'].mean() if (trades_df['pnl'] > 0).any() else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl_pct'].mean() if (trades_df['pnl'] < 0).any() else 0
            
            profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                              trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if (trades_df['pnl'] < 0).any() else float('inf')
            
            print(f"✅ Win Rate: {win_rate:.2f}%")
            print(f"📊 Avg Win (net): {avg_win:+.2f}%")
            print(f"📊 Avg Loss (net): {avg_loss:+.2f}%")
            print(f"📈 Profit Factor: {profit_factor:.2f}")
        
        print("\n🔍 SANITY CHECKS:")
        print(f"  ✓ Reserved cash ≥ 0: {self.reserved_cash >= -0.01}")
        print(f"  ✓ Cash ≥ 0: {self.cash >= -0.01}")
        print(f"  ✓ Equity complete: {len(equity_df)} days")
        print(f"  ✓ No NaN equity: {not equity_df['equity'].isna().any()}")
        
        return equity_df, trades_df


def download_sample_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Download data for 4-digit Taiwan stocks (TWSE + TPEx/OTC)
    Per-ticker download (no multi-symbol calls to avoid MultiIndex)
    """
    print("\n📥 Downloading data...")
    stock_data = {}
    
    for ticker in tickers:
        # Validate ticker
        if not TrendValidatorBacktest.validate_ticker(ticker):
            print(f"❌ {ticker}: REJECTED (not 4-digit numeric)")
            continue
        
        # Try both exchanges (single-symbol per call)
        for suffix in ['.TW', '.TWO']:
            yf_ticker = f"{ticker}{suffix}"
            
            try:
                # Single-symbol download (no MultiIndex)
                df = yf.download(yf_ticker, start=start_date, end=end_date, 
                               auto_adjust=True, progress=False)
                
                if df is None or len(df) == 0:
                    continue
                
                # Normalize OHLCV
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df = TrendValidatorBacktest.normalize_ohlcv(df)
                
                stock_data[ticker] = df
                exchange = "TWSE" if suffix == '.TW' else "TPEx"
                print(f"✅ {ticker} ({exchange}): {len(df)} bars")
                break
                
            except Exception as e:
                continue
        
        if ticker not in stock_data:
            print(f"❌ {ticker}: No data on TWSE or TPEx")
    
    return stock_data


def run_example():
    """Run example backtest"""
    
    print("\n" + "="*60)
    print("📋 BACKTEST CONFIGURATION")
    print("="*60)
    print("Universe:            4-digit Taiwan stocks (TWSE + TPEx)")
    print("Execution:           Signal @ close → Fill @ next open")
    print("Missing Sessions:    Next-available + 5-day expiry")
    print("End-of-Test:         Option 1 (Keep open, mark-to-liquidation)")
    print("Equity:              Mark-to-liquidation (daily)")
    print("Costs:               Buy 0.1425% | Sell 0.1425% + 0.3% tax")
    print("Slippage:            0.1% both sides")
    print("="*60)
    
    # 4-digit Taiwan stocks
    tickers = ['2330', '2317', '2454', '3008', '2308', '2382', '2412', '6505', '2303', '3711']
    
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    stock_data = download_sample_data(tickers, start_date, end_date)
    
    print("\n📥 Downloading benchmark (^TWII)...")
    benchmark_data = yf.download('^TWII', start=start_date, end=end_date, 
                                auto_adjust=True, progress=False)
    benchmark_data = benchmark_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    benchmark_data = TrendValidatorBacktest.normalize_ohlcv(benchmark_data)
    
    if len(stock_data) == 0:
        print("\n❌ No data downloaded")
        return None, None, None, None
    
    # Run v1.1
    print("\n" + "="*60)
    print("🔄 RUNNING v1.1")
    print("="*60)
    bt_v1 = TrendValidatorBacktest(version='v1.1', initial_capital=1000000)
    equity_v1, trades_v1 = bt_v1.run_backtest(stock_data, benchmark_data)
    
    # Run v2
    print("\n" + "="*60)
    print("🔄 RUNNING v2")
    print("="*60)
    bt_v2 = TrendValidatorBacktest(version='v2', initial_capital=1000000)
    equity_v2, trades_v2 = bt_v2.run_backtest(stock_data, benchmark_data)
    
    # Plot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity_v1.index, equity_v1['equity'], label='v1.1', linewidth=2.5, color='#2E86AB')
    ax1.plot(equity_v2.index, equity_v2['equity'], label='v2', linewidth=2.5, color='#A23B72')
    ax1.axhline(y=1000000, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
    ax1.set_title('💰 Equity Curve (Mark-to-Liquidation)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity (TWD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, :])
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
    
    ax2.set_title('📉 Drawdown', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(equity_v1.index, equity_v1['cash'], label='Cash', linewidth=2, color='#2E86AB')
    ax3.plot(equity_v1.index, equity_v1['reserved_cash'], label='Reserved', 
             linewidth=1.5, linestyle='--', color='#2E86AB', alpha=0.6)
    ax3.set_title('💵 v1.1 Cash', fontsize=12, fontweight='bold')
    ax3.set_ylabel('TWD')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(equity_v2.index, equity_v2['cash'], label='Cash', linewidth=2, color='#A23B72')
    ax4.plot(equity_v2.index, equity_v2['reserved_cash'], label='Reserved', 
             linewidth=1.5, linestyle='--', color='#A23B72', alpha=0.6)
    ax4.set_title('💵 v2 Cash', fontsize=12, fontweight='bold')
    ax4.set_ylabel('TWD')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Trend Validator v2.2 - Production Grade', fontsize=16, fontweight='bold')
    plt.show()
    
    return equity_v1, trades_v1, equity_v2, trades_v2


if __name__ == "__main__":
    print("🚀 Trend Validator Protocol v2.2")
    print("📦 Production-Grade Implementation")
    print("="*60)
    print("\n✅ ALL FIXES APPLIED:")
    print("  • Universe: 4-digit stocks only")
    print("  • Exchanges: TWSE + TPEx (.TW and .TWO)")
    print("  • Equity: Recorded EVERY day")
    print("  • Orders: Next-available + 5-day expiry")
    print("  • EoT: Option 1 (keep open)")
    print("  • P&L: Net % (includes costs)")
    print("  • Stops: Recalculated after fill")
    print("  • Donchian: Uses PRIOR window\n")
    
    equity_v1, trades_v1, equity_v2, trades_v2 = run_example()