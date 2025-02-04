import os
import tempfile
import webbrowser
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import HTML, display

from ibfuncs import get_financials, get_ib, get_open_orders, ib_pf
from snp import make_snp_unds
from utils import (ROOT, classify_open_orders, classify_pf, do_i_refresh,
                   get_dte, get_pickle, load_config, update_unds_status)

def load_dataframes():
    paths = {
        'pf': ROOT / "data" / "df_pf.pkl",
        'cov': ROOT / "data" / "df_cov.pkl",
        'nkd': ROOT / "data" / "df_nkd.pkl",
        'unds': ROOT / "data" / "df_unds.pkl",
        'protect': ROOT / "data" / "df_protect.pkl",
        'reap': ROOT / "data" / "df_reap.pkl",
        'chains': ROOT / "data" / "chains.pkl"
    }
    return {k: get_pickle(v, print_msg=False) for k, v in paths.items()}

def refresh_data(unds_path, config):
    if do_i_refresh(unds_path, max_days=config.get("MAX_FILE_AGE")):
        unds = make_snp_unds()
        with get_ib("SNP") as ib:
            fin = ib.run(get_financials(ib))
            qpf = ib_pf(ib)
            df_pf = classify_pf(qpf)
            openords = get_open_orders(ib)
            df_openords = classify_open_orders(openords, df_pf)
            df_unds = update_unds_status(df_unds=unds, df_pf=df_pf, df_openords=df_openords)
        return fin, df_pf, df_unds
    return None, None, None

def prepare_dataframe(df_pf, df_openords, df_unds):
    df = pd.concat([
        df_pf.assign(source='pf'),
        df_openords.assign(source='oo')
    ], ignore_index=True)
    
    df['dte'] = df.expiry.apply(lambda x: get_dte(x) if pd.notna(x) and x else np.nan)
    df['sort_key'] = df.apply(lambda x: (
        x['symbol'],
        {'C': 0, '0': 1, 'P': 2}.get(x['right'], 3),
        1 if x['source'] == 'und' else 0
    ), axis=1)
    
    df = df.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)
    df = pd.concat([df, df_unds.assign(source='und')], ignore_index=True)
    df['source_order'] = df['right'].map({'C': 0, '0': 1, 'P': 2, np.nan: 3})
    df = df.sort_values(by=['symbol', 'source_order']).drop(columns=['source_order']).reset_index(drop=True)
    
    und_price_dict = df_unds.set_index('symbol')['undPrice'].to_dict()
    df['undPrice'] = df['symbol'].map(und_price_dict)
    
    df.loc[df.source == 'oo', 'mktVal'] = df.groupby('symbol')['mktVal'].transform(lambda x: x.fillna(x.mean()))
    df.loc[df.source == 'oo', 'avgCost'] = df.loc[df.source == 'oo', 'lmtPrice'] * 100
    df.loc[df.source == 'oo', 'position'] = df.loc[df.source == 'oo', 'qty']
    df.loc[(df.source == 'pf') & (df.secType == 'STK'), 'qty'] = df.loc[(df.source == 'pf') & (df.secType == 'STK'), 'position'] / 100
    df.loc[(df.source == 'pf') & (df.secType == 'OPT'), 'qty'] = df.loc[(df.source == 'pf') & (df.secType == 'OPT'), 'position']
    
    cols = ['source', 'symbol', 'conId', 'secType', 'position', 'state', 'undPrice', 'strike', 'avgCost', 'mktVal', 'right', 'expiry', 'dte', 'qty', 'lmtPrice', 'action', 'unPnL']
    return df[cols]

def calculate_risk_reward(df):
    df_risk = (
        df.query('state == "protecting"')
        .groupby('symbol')
        .agg({
            'source': 'first',
            'avgCost': lambda x: (x * df.loc[x.index, 'position']).sum(),
            'undPrice': 'first',
            'strike': 'first',
            'dte': 'first',
            'position': 'first',
            'qty': 'first',
            'mktVal': lambda x: (x * df.loc[x.index, 'qty']).sum()
        })
        .assign(
            cost=lambda x: x['avgCost'],
            unprot_val=lambda x: np.where(
                x['source'] == 'pf',
                abs(x['undPrice'] - x['strike']) * x['position'] * 100,
                abs((x['undPrice'] - x['strike']) * x['qty']) * 100
            )
        )
        .reset_index()[['symbol', 'source', 'cost', 'unprot_val', 'mktVal', 'dte']]
    )

    df_reward = (
        df.query('state == "covering"')
        .groupby('symbol')
        .agg({
            'source': 'first',
            'avgCost': lambda x: (x * df.loc[x.index, 'position']).sum(),
            'undPrice': 'first',
            'strike': 'first',
            'dte': 'first',
            'position': 'first',
            'qty': 'first',
            'mktVal': lambda x: (x * df.loc[x.index, 'qty']).sum()
        })
        .assign(
            premium=lambda x: x['avgCost'],
            max_reward=lambda x: abs((x['strike'] - x['undPrice']) * x['qty'] * 100)
        )
        .reset_index()[['symbol', 'source', 'premium', 'max_reward', 'mktVal', 'dte']]
    )

    df_sowed = df[df.state == 'sowed'].sort_values('unPnL')
    
    cover_projection = (df_risk.dte.mean() / 7 - 1) * abs(df_reward.premium.sum())
    sowed_projection = df_sowed.avgCost.sum()
    total_reward = cover_projection + abs(sowed_projection)
    
    return df_risk, df_reward, df_sowed, cover_projection, sowed_projection, total_reward

def analyze_premiums(df_cov, df_nkd):
    cov_premium = (df_cov.xPrice * df_cov.qty * 100).sum() if df_cov is not None and not df_cov.empty else 0
    nkd_premium = (df_nkd.xPrice * 100 * df_nkd.qty).sum() if df_nkd is not None and not df_nkd.empty else 0
    maxProfit = (
        np.where(
            df_cov.right == "C",
            (df_cov.strike - df_cov.undPrice) * df_cov.qty * 100,
            (df_cov.undPrice - df_cov.strike) * df_cov.qty * 100,
        ).sum() + cov_premium
    ) if df_cov is not None and not df_cov.empty else 0
    
    return cov_premium, nkd_premium, maxProfit

def style_dataframe(df, option_breach_index, message, calc):
    def style_rows(row):
        if row.name in option_breach_index:
            return ['background-color: black; color: white'] * len(row)
        elif row['source'] == 'und':
            return ['background-color: green; color: white'] * len(row) if row['unPnL'] > 0 else ['background-color: red; color: white'] * len(row)
        return ['background-color: #313131; color: #b4b1b1'] * len(row)
    
    int_columns = ['qty', 'position']
    float_columns = ['undPrice', 'strike', 'avgCost', 'unPnL', 'dte', 'mktVal']
    format_dict = {col: '{:.0f}' for col in int_columns}
    format_dict.update({col: '{:.2f}' for col in float_columns})
    
    caption = f'{message} US${float(calc.replace(",", "")):,.0f}' if message and calc else None
    
    return (df.style
            .format(format_dict)
            .apply(style_rows, axis=1)
            .hide(axis='index')
            .hide(axis='columns', subset=df.columns[df.isna().all()])
            .set_caption(caption))

def is_running_in_bare_terminal():
    if any(env in os.environ for env in ['VSCODE_PID', 'PYCHARM_HOSTED', 'JUPYTER_RUNTIME_DIR', 'SPYDER_ARGS']):
        return False
    if 'TERM_PROGRAM' in os.environ and 'vscode' in os.environ['TERM_PROGRAM'].lower():
        return False
    return True

def render_output(styled_df, msg_pad, is_vscode):
    if is_vscode:
        display(styled_df)
        for msg in msg_pad:
            print(msg)
    else:
        html = styled_df.to_html()
        css = Path(ROOT / 'src' / 'styles.css').read_text()
        msg_pad_html = '<br>'.join(msg_pad)
        full_html = f"<style>{css}</style>{html}<br>{msg_pad_html}"
        
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
            url = 'file://' + f.name
            f.write(full_html)
        webbrowser.open(url)

def main():
    dfs = load_dataframes()
    config = load_config("SNP")
    fin, df_pf, df_unds = refresh_data(dfs['unds'], config)
    
    if fin is None:
        fin = get_financials(get_ib("SNP"))
    if df_pf is None:
        df_pf = dfs['pf']
    if df_unds is None:
        df_unds = dfs['unds']
    
    df = prepare_dataframe(df_pf, dfs['cov'], df_unds)
    df_risk, df_reward, df_sowed, cover_projection, sowed_projection, total_reward = calculate_risk_reward(df)
    
    cov_premium, nkd_premium, maxProfit = analyze_premiums(dfs['cov'], dfs['nkd'])
    
    calls_lt_und = df[(df.right == 'C') & (df.strike < df.undPrice)].index.tolist()
    puts_gt_und = df[(df.right == 'P') & (df.strike > df.undPrice)].index.tolist()
    option_breach_index = list(set(calls_lt_und).union(set(puts_gt_und)))
    
    if option_breach_index:
        breach_pnl = df[(df.source == "und") & (df.symbol.isin(df.loc[option_breach_index, 'symbol'].unique())) & (df.state.isin(['solid', 'unprotected']))]["unPnL"]
        opt_breached_df = df[df.symbol.isin(df.loc[option_breach_index, 'symbol'])]
        total_breach_pnl = format(breach_pnl.sum(), ",.0f")
        styled_df = style_dataframe(opt_breached_df, option_breach_index, 'Option cover breaches are generating', total_breach_pnl)
        
        msg_pad = [
            "FINANCIALS",
            '==========',
            *[f"{k}: {format(v, ',.2f')}" for k, v in fin.items() if v],
            '\nRisk Analysis',
            '-------------',
            f'Our max risk is ${df_risk.unprot_val.sum():,.0f} from ${fin.get("stocks", 0):,.0f} stock for {df_risk.dte.mean():.1f} days at risk premium of ${df_risk.cost.sum():,.0f}',
            'All stock positions are protected!' if df[(df.source == "und") & (df.state == "unprotected")].empty else f'Unprotected stocks: {", ".join(df[(df.source == "und") & (df.state == "unprotected")].symbol.unique())}',
            '\nRewards',
            '-------',
            f'Total reward this month is expected to be ${total_reward:,.0f}',
            f'Our maximum cover reward in {df_reward.dte.mean():.1f} days is ${df_reward.max_reward.sum():,.0f}, if all covers get blown.',
            f'Our cover premiums from covering options is ${abs(df_reward.premium.sum()):,.0f} this week from our stock positions',
            f'...this can be projected to give us ${cover_projection:,.0f} for the protected period',
            f'Our sowed reward in about {df_sowed.dte.mean():.1f} dte days is ${df_sowed.avgCost.sum():,.0f}'
        ]
        
        render_output(styled_df, msg_pad, not is_running_in_bare_terminal())

if __name__ == "__main__":
    main()
