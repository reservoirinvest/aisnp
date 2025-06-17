"""
State identification module for portfolio, orders, and symbols.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional


def identify_portfolio_states(df_pf: pd.DataFrame) -> pd.DataFrame:
    """
    Identify portfolio states based on positions in the portfolio.
    
    Args:
        df_pf: DataFrame containing portfolio positions with columns:
              - symbol: str - Symbol name
              - secType: str - 'STK' for stock, 'OPT' for option
              - right: str - 'P' for put, 'C' for call (only for options)
              - position: int - Position size (can be positive or negative)
    
    Returns:
        DataFrame with an additional 'state' column
    """
    df = df_pf.copy()
    df['state'] = 'unknown'
    
    # First, identify straddled options (matching calls and puts with same strike/expiry)
    def find_straddles(df_opt: pd.DataFrame) -> pd.Series:
        straddle_mask = pd.Series(False, index=df_opt.index)
        
        # Only consider options with the same symbol, expiry and strike
        for (symbol, expiry, strike), group in df_opt.groupby(['symbol', 'expiry', 'strike']):
            if len(group) == 2 and set(group['right']) == {'C', 'P'}:
                straddle_mask[group.index] = True
                
        return straddle_mask
    
    # Apply state identification in order of priority
    
    # 1. Straddled (matching calls and puts with no underlying stock)
    if not df[df.secType == 'OPT'].empty:
        df_opt = df[df.secType == 'OPT'].copy()
        straddle_mask = find_straddles(df_opt)
        df.loc[straddle_mask[straddle_mask].index, 'state'] = 'straddled'
    
    # 2. Covering (short calls or puts with underlying stock)
    covering_mask = (
        (df.secType == 'OPT') & 
        (((df.right == 'C') & (df.position < 0)) | 
         ((df.right == 'P') & (df.position < 0))) &
        (df.symbol.isin(df[df.secType == 'STK'].symbol))
    )
    df.loc[covering_mask, 'state'] = 'covering'
    
    # 3. Protecting (long calls or puts with underlying stock)
    protecting_mask = (
        (df.secType == 'OPT') & 
        (((df.right == 'C') & (df.position > 0)) | 
         ((df.right == 'P') & (df.position > 0))) &
        (df.symbol.isin(df[df.secType == 'STK'].symbol))
    )
    df.loc[protecting_mask, 'state'] = 'protecting'
    
    # 4. Sowed (short options without matching stock positions)
    sowed_mask = (
        (df.secType == 'OPT') & 
        (df.position < 0) &
        (~df.symbol.isin(df[df.secType == 'STK'].symbol))
    )
    df.loc[sowed_mask, 'state'] = 'sowed'
    
    # 5. Orphaned (long options without matching stock positions)
    orphaned_mask = (
        (df.secType == 'OPT') & 
        (df.position > 0) &
        (~df.symbol.isin(df[df.secType == 'STK'].symbol))
    )
    df.loc[orphaned_mask, 'state'] = 'orphaned'
    
    # 6. Zen (stock with both covering and protecting options)
    stock_symbols = df[df.secType == 'STK'].symbol.unique()
    for symbol in stock_symbols:
        symbol_mask = df.symbol == symbol
        symbol_covering = (df.symbol == symbol) & (df.state == 'covering')
        symbol_protecting = (df.symbol == symbol) & (df.state == 'protecting')
        
        if symbol_covering.any() and symbol_protecting.any():
            df.loc[symbol_mask & (df.secType == 'STK'), 'state'] = 'zen'
    
    # 7. Unprotected (stock with only covering options)
    for symbol in stock_symbols:
        symbol_mask = (df.symbol == symbol) & (df.secType == 'STK')
        if df.loc[symbol_mask, 'state'].iloc[0] != 'zen':  # Skip if already marked as zen
            has_covering = ((df.symbol == symbol) & (df.state == 'covering')).any()
            has_protecting = ((df.symbol == symbol) & (df.state == 'protecting')).any()
            
            if has_covering and not has_protecting:
                df.loc[symbol_mask, 'state'] = 'unprotected'
    
    # 8. Uncovered (stock with only protecting options)
    for symbol in stock_symbols:
        symbol_mask = (df.symbol == symbol) & (df.secType == 'STK')
        if df.loc[symbol_mask, 'state'].iloc[0] not in ['zen', 'unprotected']:
            has_covering = ((df.symbol == symbol) & (df.state == 'covering')).any()
            has_protecting = ((df.symbol == symbol) & (df.state == 'protecting')).any()
            
            if has_protecting and not has_covering:
                df.loc[symbol_mask, 'state'] = 'uncovered'
    
    # 9. Exposed (stock with no options)
    exposed_mask = (
        (df.secType == 'STK') & 
        (~df.symbol.isin(df[df.secType == 'OPT'].symbol)) &
        (df.state == 'unknown')
    )
    df.loc[exposed_mask, 'state'] = 'exposed'
    
    return df


def identify_order_states(df_orders: pd.DataFrame, df_pf: pd.DataFrame) -> pd.DataFrame:
    """
    Identify order states based on open orders and portfolio positions.
    
    Args:
        df_orders: DataFrame containing open orders with columns:
                 - symbol: str - Symbol name
                 - secType: str - 'STK' or 'OPT'
                 - right: str - 'P' or 'C' for options
                 - action: str - 'BUY' or 'SELL'
                 - position: int - Position size (can be positive or negative)
        df_pf: DataFrame with portfolio positions (already processed with identify_portfolio_states)
    
    Returns:
        DataFrame with an additional 'state' column
    """
    if df_orders.empty:
        return df_orders
        
    df = df_orders.copy()
    df['state'] = 'unknown'
    
    # Get stock positions from portfolio
    stock_positions = df_pf[df_pf.secType == 'STK']
    
    # 1. Covering orders (SELL options with underlying stock position)
    covering_mask = (
        (df.secType == 'OPT') & 
        (df.action == 'SELL') &
        (df.symbol.isin(stock_positions.symbol))
    )
    df.loc[covering_mask, 'state'] = 'covering'
    
    # 2. Protecting orders (BUY options with underlying stock position)
    protecting_mask = (
        (df.secType == 'OPT') & 
        (df.action == 'BUY') &
        (df.symbol.isin(stock_positions.symbol))
    )
    df.loc[protecting_mask, 'state'] = 'protecting'
    
    # 3. Sowing orders (SELL options without underlying stock position)
    sowing_mask = (
        (df.secType == 'OPT') & 
        (df.action == 'SELL') &
        (~df.symbol.isin(stock_positions.symbol))
    )
    df.loc[sowing_mask, 'state'] = 'sowing'
    
    # 4. Reaping orders (BUY options with matching short option position)
    # Get all short option positions from portfolio
    short_options = df_pf[
        (df_orders.secType == 'OPT') & 
        (df_orders.position < 0)
    ][['symbol', 'right', 'expiry', 'strike']].drop_duplicates()
    
    if not short_options.empty:
        # Create a unique key for matching
        df['option_key'] = df.apply(
            lambda x: f"{x['symbol']}_{x['right']}_{x['expiry']}_{x['strike']}" 
            if x['secType'] == 'OPT' else None, axis=1
        )
        
        short_options['option_key'] = short_options.apply(
            lambda x: f"{x['symbol']}_{x['right']}_{x['expiry']}_{x['strike']}", axis=1
        )
        
        reaping_mask = (
            (df.secType == 'OPT') & 
            (df.action == 'BUY') &
            (df.option_key.isin(short_options.option_key))
        )
        df.loc[reaping_mask, 'state'] = 'reaping'
        df = df.drop(columns=['option_key'])
    
    # 5. Straddling orders (BUY calls and puts for the same symbol, not in portfolio)
    if len(df[df.secType == 'OPT']) >= 2:
        # Group by symbol and expiry to find pairs of calls and puts
        for (symbol, expiry), group in df[df.secType == 'OPT'].groupby(['symbol', 'expiry']):
            if len(group) >= 2 and set(group.right) == {'C', 'P'} and all(group.action == 'BUY'):
                # Check if this straddle is not already in the portfolio
                straddle_in_portfolio = False
                if symbol in df_pf.symbol.unique():
                    symbol_pf = df_pf[df_pf.symbol == symbol]
                    if len(symbol_pf[symbol_pf.secType == 'OPT']) >= 2 and \
                       set(symbol_pf[symbol_pf.secType == 'OPT'].right) == {'C', 'P'}:
                        straddle_in_portfolio = True
                
                if not straddle_in_portfolio:
                    df.loc[group.index, 'state'] = 'straddling'
    
    # 6. De-orphaning orders (SELL options with orphaned long position)
    orphaned_options = df_pf[
        (df_pf.state == 'orphaned') & 
        (df_pf.secType == 'OPT')
    ][['symbol', 'right', 'expiry', 'strike']].drop_duplicates()
    
    if not orphaned_options.empty:
        # Create a unique key for matching
        if 'option_key' not in df.columns:
            df['option_key'] = df.apply(
                lambda x: f"{x['symbol']}_{x['right']}_{x['expiry']}_{x['strike']}" 
                if x['secType'] == 'OPT' else None, axis=1
            )
        
        orphaned_options['option_key'] = orphaned_options.apply(
            lambda x: f"{x['symbol']}_{x['right']}_{x['expiry']}_{x['strike']}", axis=1
        )
        
        deorphan_mask = (
            (df.secType == 'OPT') & 
            (df.action == 'SELL') &
            (df.option_key.isin(orphaned_options.option_key)) &
            (~df.symbol.isin(stock_positions.symbol))  # No stock position
        )
        df.loc[deorphan_mask, 'state'] = 'de-orphaning'
        
        if 'option_key' in df.columns:
            df = df.drop(columns=['option_key'])
    
    return df


def identify_symbol_states(df_unds: pd.DataFrame, df_pf: pd.DataFrame, df_orders: pd.DataFrame) -> pd.DataFrame:
    """
    Identify symbol states based on portfolio and order states.
    
    Args:
        df_unds: DataFrame containing underlying symbols with columns:
               - symbol: str - Symbol name
        df_pf: DataFrame with portfolio positions (already processed with identify_portfolio_states)
        df_orders: DataFrame with open orders (already processed with identify_order_states)
    
    Returns:
        DataFrame with an additional 'state' column
    """
    df = df_unds.copy()
    df['state'] = 'unknown'
    
    # Process each symbol
    for idx, row in df.iterrows():
        symbol = row['symbol']
        symbol_pf = df_pf[df_pf.symbol == symbol]
        symbol_orders = df_orders[df_orders.symbol == symbol]
        
        # Check for zen conditions
        zen_conditions = [
            # Has both covering and protecting portfolio positions or orders
            (symbol_pf[symbol_pf.state == 'covering'].shape[0] > 0 and 
             symbol_pf[symbol_pf.state == 'protecting'].shape[0] > 0),
            
            # Has straddled portfolio state
            (symbol_pf[symbol_pf.state == 'straddled'].shape[0] > 0),
            
            # Has short 'sowing' order
            (symbol_orders[symbol_orders.state == 'sowing'].shape[0] > 0),
            
            # Is in 'unprotected' portfolio state with a 'protecting' order
            ((symbol_pf[symbol_pf.state == 'unprotected'].shape[0] > 0) and 
             (symbol_orders[symbol_orders.state == 'protecting'].shape[0] > 0)),
            
            # Is in 'uncovered' portfolio state with a 'covering' order
            ((symbol_pf[symbol_pf.state == 'uncovered'].shape[0] > 0) and 
             (symbol_orders[symbol_orders.state == 'covering'].shape[0] > 0)),
            
            # Has long option 'orphaned' position with an open 'de-orphaning' order
            ((symbol_pf[(symbol_pf.state == 'orphaned') & (symbol_pf.secType == 'OPT')].shape[0] > 0) and 
             (symbol_orders[symbol_orders.state == 'de-orphaning'].shape[0] > 0)),
            
            # Has short option 'sowed' position with an open 'reaping' order
            ((symbol_pf[(symbol_pf.state == 'sowed') & (symbol_pf.secType == 'OPT')].shape[0] > 0) and 
             (symbol_orders[symbol_orders.state == 'reaping'].shape[0] > 0))
        ]
        
        if any(zen_conditions):
            df.at[idx, 'state'] = 'zen'
            continue
            
        # Check for unreaped (short option position with no reaping order)
        if (symbol_pf[(symbol_pf.state == 'sowed') & (symbol_pf.secType == 'OPT')].shape[0] > 0 and
            symbol_orders[symbol_orders.state == 'reaping'].shape[0] == 0):
            df.at[idx, 'state'] = 'unreaped'
            continue
            
        # Check for exposed (stock with no covering or protecting positions/orders)
        if (symbol_pf[(symbol_pf.secType == 'STK') & 
                     (~symbol_pf.symbol.isin(symbol_pf[symbol_pf.secType == 'OPT'].symbol))].shape[0] > 0):
            df.at[idx, 'state'] = 'exposed'
            continue
            
        # Check for uncovered (stock with protecting but no covering positions/orders)
        if (symbol_pf[symbol_pf.state == 'uncovered'].shape[0] > 0 or
            (symbol_pf[symbol_pf.secType == 'STK'].shape[0] > 0 and
             symbol_pf[symbol_pf.state == 'protecting'].shape[0] > 0 and
             symbol_pf[symbol_pf.state == 'covering'].shape[0] == 0)):
            df.at[idx, 'state'] = 'uncovered'
            continue
            
        # Check for unprotected (stock with covering but no protecting positions/orders)
        if (symbol_pf[symbol_pf.state == 'unprotected'].shape[0] > 0 or
            (symbol_pf[symbol_pf.secType == 'STK'].shape[0] > 0 and
             symbol_pf[symbol_pf.state == 'covering'].shape[0] > 0 and
             symbol_pf[symbol_pf.state == 'protecting'].shape[0] == 0)):
            df.at[idx, 'state'] = 'unprotected'
            continue
            
        # Check for orphaned (long options without underlying stock)
        if (symbol_pf[(symbol_pf.state == 'orphaned') & (symbol_pf.secType == 'OPT')].shape[0] > 0 and
            symbol_pf[symbol_pf.secType == 'STK'].shape[0] == 0):
            df.at[idx, 'state'] = 'orphaned'
            continue
            
        # Default to virgin if no positions or orders
        if symbol_pf.empty and symbol_orders.empty:
            df.at[idx, 'state'] = 'virgin'
    
    return df
