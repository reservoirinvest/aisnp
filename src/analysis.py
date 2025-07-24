# %%
# GETS OVERALL FINANCIALS

import os
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML, display

from ibfuncs import get_financials, get_ib, get_open_orders, ib_pf
from snp import make_snp_unds
from utils import (ROOT, classify_open_orders, classify_pf, do_i_refresh,
                   get_dte, get_pickle, how_many_days_old, load_config,
                   update_unds_status)

# display(HTML('<style>pre { white-space: pre-wrap; }</style>'))


# Get the dataframes
pf_path = ROOT / "data" / "df_pf.pkl"  # portfolio path
cov_path = ROOT / "data" / "df_cov.pkl"  # covered call and put path
nkd_path = ROOT / "data" / "df_nkd.pkl"
unds_path = ROOT / "data" / "df_unds.pkl"
protect_path = ROOT / "data" / "df_protect.pkl"
reap_path = ROOT / "data" / "df_reap.pkl"
chains_path = ROOT / "data" / "df_chains.pkl"

df_pf = get_pickle(pf_path, print_msg=False)
df_cov = get_pickle(cov_path, print_msg=False)
df_nkd = get_pickle(nkd_path, print_msg=False)
df_unds = get_pickle(unds_path, print_msg=False)
df_protect = get_pickle(protect_path, print_msg=False)
df_reap = get_pickle(reap_path, print_msg=False)
chains = get_pickle(chains_path, print_msg=False)

# Get unds. Make it fresh if stale.
config = load_config("SNP")
MAX_FILE_AGE = config.get("MAX_FILE_AGE")
REAPRATIO = config.get("REAPRATIO")

if do_i_refresh(unds_path, max_days=MAX_FILE_AGE):
    unds = make_snp_unds()


with get_ib("SNP") as ib:
    fin = ib.run(get_financials(ib))
    qpf = ib_pf(ib)
    df_pf = classify_pf(qpf)
    openords = get_open_orders(ib)
    df_openords = classify_open_orders(openords, df_pf)
    df_unds = update_unds_status(df_unds=df_unds, df_pf=df_pf, df_openords=df_openords)
    # df_acc = util.df(ib.accountValues())

# Update symbol in df_pf with undPrice in df_unds
df_pf = df_pf.merge(df_unds[["symbol", "undPrice"]], on="symbol", how="left")

print("\nFINANCIALS")
print('==========')
for k, v in fin.items():
    if v:
        if v > 1:
            print(f"{k}: {format(v, ',.0f')}")
        else:
            print(f"{k}: {format(v, ',.2f')}")

# %%
# COMPUTES RISK AND REWARD WITH COST OF MITIGATING RISK

df = pd.concat([
    df_pf.assign(source='pf'),
    df_openords.assign(source='oo')
    ], ignore_index=True)

df = df.assign(
    dte=df.expiry.apply(lambda x: get_dte(x) if pd.notna(x) and x else np.nan)
)

# First, create a temporary column for sorting
df['sort_key'] = df.apply(lambda x: (
    x['symbol'],
    {'C': 0, '0': 1, 'P': 2}.get(x['right'], 3),
    1 if x['source'] == 'und' else 0
), axis=1)

# Sort using the temporary column
df = df.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)
df = (pd.concat([df, df_unds.assign(source='und')], ignore_index=True)
            .assign(source_order=lambda x: x['right'].map({'C': 0, '0': 1, 'P': 2, np.nan: 3}))
            .sort_values(by=['symbol', 'source_order'])
            .drop(columns=['source_order'])
            .reset_index(drop=True))
und_price_dict = df_unds.set_index('symbol')['undPrice'].to_dict()
df['undPrice'] = df['symbol'].map(und_price_dict)

# get mktVal for oo from pf
df.loc[df.source == 'oo', 'mktVal'] = df.groupby('symbol')['mktVal'].transform(lambda x: x.fillna(x.mean()))

# replace avgCost for oo with lmtPrice*100
df.loc[df.source == 'oo', 'avgCost'] = df.loc[df.source == 'oo', 'lmtPrice']*100

# replace position for oo with qty
df.loc[df.source == 'oo', 'position'] = df.loc[df.source == 'oo', 'qty']

# replace qty for pf with position/100 for stocks, else position for options
df.loc[(df.source == 'pf') & (df.secType == 'STK'), 'qty'] = df.loc[(df.source == 'pf') & (df.secType == 'STK'), 'position']/100
df.loc[(df.source == 'pf') & (df.secType == 'OPT'), 'qty'] = df.loc[(df.source == 'pf') & (df.secType == 'OPT'), 'position']

cols = [
    'source', 'symbol', 'conId', 'secType', 'position', 'state', 'undPrice', 'strike', 
    'avgCost', 'mktVal', 'right', 'expiry', 'dte', 'qty', 'lmtPrice', 'action', 'unPnL'
]

df = df[cols]

df_risk = (
    df
    .query('state == "protecting"')
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
    .reset_index()
    [['symbol', 'source', 'cost', 'unprot_val', 'mktVal', 'dte']]
)

df_reward = (
    df
    .query('state == "covering"')
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
    .reset_index()
    [['symbol', 'source', 'premium', 'max_reward', 'mktVal', 'dte']]
)

df_sowed = df[df.state == 'sowed'].sort_values('unPnL')
cover_projection = (df_risk.dte.mean()/7-1)*abs(df_reward.premium.sum())
sowed_projection = df_sowed.avgCost.sum()*(1-REAPRATIO)
total_reward = cover_projection + abs(sowed_projection)

print('\nRisk Analysis')
print('-------------')
risk_msg = []
risk_msg.append(f'Our max risk is ${df_risk.unprot_val.sum():,.0f} from ${fin.get("stocks", 0):,.0f} stock for {df_risk.dte.mean():.1f} days at risk premium of ${df_risk.cost.sum():,.0f}')
unprotected_stocks = df[(df.source == "und") & (df.state == "unprotected")].symbol.unique()
print('\n')
if unprotected_stocks.size > 0:
    risk_msg.append(f'Unprotected stocks: {", ".join(unprotected_stocks)}')
else:
    risk_msg.append('All stock positions are protected!')

print('\n'.join(risk_msg))

print('\nRewards')
print('-------')

naked_premium = 0
if not df_openords.empty:
    naked_premium = (df_openords.lmtPrice * df_openords.qty).sum() * 100

reward_msg = (
    f'Total reward this month is expected to be ${total_reward:,.0f}.\n '
    f'Our maximum cover reward in {df_reward.dte.mean():.1f} days is '
    f'${df_reward.max_reward.sum():,.0f}, if all covers get blown.\n\n'
    f'Our cover premiums from covering options is ${abs(df_reward.premium.sum()):,.0f} this week from our stock positions\n'
    f' ...this can be projected to give us ${cover_projection:,.0f} for the protected period\n\n' 
    f'Our naked premiums from open orders is ${naked_premium:,.0f}\n'
)

print(reward_msg)

sow_msg = (
    f'Our sowed reward in about {df_sowed.dte.mean():.1f} dte days is ${sowed_projection:,.0f}'
)

print(sow_msg)

# %%
# GETS STATE DETAILS
# Analyze covered calls and puts
if df_cov is not None and not df_cov.empty:
    cov_premium = (df_cov.xPrice * df_cov.qty * 100).sum()
    maxProfit = (
        np.where(
            df_cov.right == "C",
            (df_cov.strike - df_cov.undPrice) * df_cov.qty * 100,
            (df_cov.undPrice - df_cov.strike) * df_cov.qty * 100,
        ).sum()
        + cov_premium
    )
else:
    cov_premium = 0
    maxProfit = 0

if df_nkd is not None and not df_nkd.empty:
    nkd_premium = (df_nkd.xPrice * 100 * df_nkd.qty).sum()
else:
    nkd_premium = 0

if cov_premium > 0 or nkd_premium > 0:
    print('\n')
    print('PREMIUMS AND PROFIT from df_cov and df_nkd')
    print('==========================================')
    print("Total Premium available is", format(cov_premium + nkd_premium, ',.0f'))
    print(f"...Cover Premium: {format(cov_premium, ',.0f')}")
    print(f"...Naked Premiums: {format(nkd_premium, ',.0f')}\n")
    print(f"Max possible profit from covers: {format(maxProfit, ',.0f')}")

print('\n')
print('SYMBOL COUNT BY STATE')
print('=====================')
print(', '.join(f"{state}: {len(df.symbol.unique())}" for state, df in df_unds.groupby('state')))
print('\n')
print('COUNT OF SYMBOLS IN EACH DATAFRAME')
print('==================================')
print(', '.join(f"{k}: {len(v) if v is not None else 0}" for k, v in {'df_cov': df_cov, 'df_protect': df_protect, 'df_reap': df_reap, 'df_nkd': df_nkd, }.items()))

# %%
# OUTLINES BREACHES

with open(ROOT / 'src' / 'styles.css', 'r') as f:
    css = f.read()

def style_rows(df, rows_index, message=None, calc=None):
    def _style_rows(row):
        if row.name in rows_index:
            return ['background-color: black; color: white'] * len(row)
        elif row['source'] == 'und':
            if row['unPnL'] > 0:
                return ['background-color: green; color: white'] * len(row)
            elif row['unPnL'] < 0:
                return ['background-color: red; color: white'] * len(row)
        return ['background-color: #313131; color: #b4b1b1'] * len(row)
    
    # DRY approach to formatting
    int_columns = ['qty', 'position']
    float_columns = ['undPrice', 'strike', 'avgCost', 'unPnL', 'dte', 'mktVal']
    format_dict = {col: '{:.0f}' for col in int_columns}
    format_dict.update({col: '{:.2f}' for col in float_columns})
    
    # Prepare caption
    caption = f'<font size=4>{message}</font>' if message else None
    if message and calc is not None:
        # Convert calc to a numeric value if it's a string
        try:
            calc_numeric = float(calc.replace(',', ''))
            caption += f'<font size=4> US${calc_numeric:,.0f}</font>'
        except (ValueError, AttributeError):
            caption += f'<font size=4> US${calc}</font>'
    
    # Styling of option breaches
    dfs = (
        df.style
        .format(format_dict)
        .apply(_style_rows, axis=1)
        .hide(axis='index')
        .hide(axis='columns', subset=df.columns[df.isna().all()])
    )
    
    # Add caption if provided
    if caption:
        dfs = dfs.set_caption(caption)
    
    return dfs

# Colour rows for option breaches
calls_lt_und = df[(df.right == 'C') & (df.strike < df.undPrice)].index.tolist()
puts_gt_und = df[(df.right == 'P') & (df.strike > df.undPrice)].index.tolist()
option_breach_index = list(set(calls_lt_und).union(set(puts_gt_und)))

# Initialize variables
opt_breached_df = pd.DataFrame()
total_breach_pnl = "0"  # Default value if no breaches
dfs = None  # Initialize dfs as None

# Data Manipulation
if option_breach_index:
    # Get the breach PnL for the relevant symbols
    breach_pnl = df[(df.source == "und") &
                  (df.symbol.isin(df.loc[option_breach_index, 'symbol'].unique())) &
                  (df.secType == 'STK')].unPnL

    # Calculate the total breach PnL for the caption
    total_breach_pnl = format(breach_pnl.sum(), ",.0f")

    # Filter the DataFrame for display
    opt_breached_df = df[df.symbol.isin(df.loc[option_breach_index, 'symbol'])]

# Only show the table if we have breached options
if not opt_breached_df.empty:
    dfs = style_rows(
        opt_breached_df, 
        rows_index=option_breach_index, 
        message='Option cover breaches are generating', 
        calc=total_breach_pnl
    )

def is_running_in_bare_terminal() -> bool:
    """
    Determines if the code is running in a bare terminal.
    """
    # Check for common IDE and notebook environments
    if any(env in os.environ for env in ['VSCODE_PID', 'PYCHARM_HOSTED', 'JUPYTER_RUNTIME_DIR', 'SPYDER_ARGS']):
        return False
    
    # Check for VSCode's integrated terminal
    if 'TERM_PROGRAM' in os.environ and 'vscode' in os.environ['TERM_PROGRAM'].lower():
        return False
    
    # Consider all other environments as "bare terminal" for this purpose
    return True

def webbrowser_styled_df(styled_df):
    # Capture print statements in a message pad
    msg_pad = []
    
    # Capture existing print statements
    msg_pad.append("FINANCIALS")
    msg_pad.append('==========')
    for k, v in fin.items():
        if v:
            if v > 1:
                msg_pad.append(f"{k}: {format(v, ',.0f')}")
            else:
                msg_pad.append(f"{k}: {format(v, ',.2f')}")
    
    msg_pad.append('\nRisk Analysis')
    msg_pad.append('-------------')
    risk_msg = []
    risk_msg.append(f'Our max risk is ${df_risk.unprot_val.sum():,.0f} from ${fin.get("stocks", 0):,.0f} stock for {df_risk.dte.mean():.1f} days at risk premium of ${df_risk.cost.sum():,.0f}')
    unprotected_stocks = df[(df.source == "und") & (df.state == "unprotected")].symbol.unique()
    if unprotected_stocks.size > 0:
        risk_msg.append(f'Unprotected stocks: {", ".join(unprotected_stocks)}')
    else:
        risk_msg.append('All stock positions are protected!')
    msg_pad.extend(risk_msg)
    
    msg_pad.append('\nRewards')
    msg_pad.append('-------')
    reward_msg = (
        f'Total reward this month is expected to be ${total_reward:,.0f} '
        f'Our maximum cover reward in {df_reward.dte.mean():.1f} days is '
        f'${df_reward.max_reward.sum():,.0f}, if all covers get blown.\n\n'
        f'Our cover premiums from covering options is ${abs(df_reward.premium.sum()):,.0f} this week from our stock positions\n'
        f' ...this can be projected to give us ${cover_projection:,.0f} for the protected period\n' 
    )
    msg_pad.append(reward_msg)
    
    sow_msg = (
        f'Our sowed reward in about {df_sowed.dte.mean():.1f} dte days is ${df_sowed.avgCost.sum():,.0f}'
    )
    msg_pad.append(sow_msg)
    
    # Add premiums and profit section if applicable
    if cov_premium > 0 or nkd_premium > 0:
        msg_pad.append('\nPREMIUMS AND PROFIT from df_cov and df_nkd')
        msg_pad.append('==========================================')
        msg_pad.append(f"Total Premium available is {format(cov_premium + nkd_premium, ',.0f')}")
        msg_pad.append(f"...Cover Premium: {format(cov_premium, ',.0f')}")
        msg_pad.append(f"...Naked Premiums: {format(nkd_premium, ',.0f')}")
        msg_pad.append(f"Max possible profit from covers: {format(maxProfit, ',.0f')}")
    
    msg_pad.append('\nSYMBOL COUNT BY STATE')
    msg_pad.append('=====================')
    msg_pad.append(', '.join(f"{state}: {len(df.symbol.unique())}" for state, df in df_unds.groupby('state')))
    
    msg_pad.append('\nCOUNT OF SYMBOLS IN EACH DATAFRAME')
    msg_pad.append('==================================')
    msg_pad.append(', '.join(f"{k}: {len(v) if v is not None else 0}" for k, v in {'df_cov': df_cov, 'df_protect': df_protect, 'df_reap': df_reap, 'df_nkd': df_nkd, }.items()))
    
    # Convert the styled DataFrame to HTML using to_html()
    html = styled_df.to_html()

    # Print the message pad contents
    print("\n".join(msg_pad))

    # Read the contents of the CSS file
    try:
        css_path = Path(ROOT / 'src' / 'styles.css')
        css_contents = css_path.read_text()
    except Exception as e:
        print(f"Error reading CSS file: {e}")
        css_contents = ""

    # Add custom CSS to improve the appearance
    custom_css = f"""
    <style>
        {css_contents}
        .msg-pad {{ 
            font-family: monospace; 
            white-space: pre-wrap; 
            background-color: #f4f4f4; 
            padding: 10px; 
            margin-top: 20px; 
            border: 1px solid #ddd; 
        }}
    </style>
    """

    # Create message pad HTML
    msg_pad_html = '<div class="msg-pad">' + '\n'.join(msg_pad) + '</div>'

    # Create a timestamp for the top of the HTML
    from datetime import datetime
    current_time = datetime.now().strftime('%d-%b-%Y %H:%M:%S')
    timestamp_html = f'<div class="timestamp">Report Generated: {current_time}</div>'

    # Modify the full_html to include the timestamp
    full_html = f"""
    <html>
    <head>{custom_css}
        <style>
            .timestamp {{
                font-family: Arial, sans-serif;
                text-align: right;
                color: #666;
                padding: 10px;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        {timestamp_html}
        {msg_pad_html}
        {html}
    </body>
    </html>"""

    # Ensure the report directory exists
    report_dir = ROOT / 'report'
    report_dir.mkdir(exist_ok=True)

    # Create a filename with current timestamp in yyyymmdd_hh format
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H')
    report_path = report_dir / f'report_{timestamp}.html'

    # Write the full HTML to the file
    with report_path.open('w') as f:
        f.write(full_html)

    # Set read permissions for the file
    report_path.chmod(0o644)

    # Open the HTML file in the default web browser
    try:
        webbrowser.open(f'file://{report_path}')
    except Exception as e:
        print(f"Error opening browser: {e}")
        print(f"Try opening this file manually: {report_path}")

    # Explicitly prevent the Styler object from being printed in the terminal
    return

# Main logic
if is_running_in_bare_terminal():
    # If in a bare terminal, render the styled DataFrame in the browser
    webbrowser_styled_df(dfs)
else:
    # If in an IDE or notebook, use IPython.display
    from IPython.display import display, HTML
    display(HTML('<style>pre { white-space: pre-wrap; }</style>'))
    display(dfs)

# %%
# ANALYZES THE BASE DATA QUALITY
# Count unique symbols in each DataFrame
unique_unds_symbols = len(df_unds['symbol'].unique())
unique_chains_symbols = chains['symbol'].nunique()
unique_oo_symbols = df_openords['symbol'].nunique() if 'symbol' in df_openords.columns else None
unique_pf_symbols = df_pf['symbol'].nunique() if 'symbol' in df_pf.columns else None

# Display the counts
unique_symbols_count = {
    'No of und symbols': (unique_unds_symbols, f"{how_many_days_old(unds_path):.2f} days old."),
    'No of chain symbols': (unique_chains_symbols, f"{how_many_days_old(chains_path):.2f} days old."),
    'No of portfolio symbols': unique_pf_symbols,
    'No of open order symbols': unique_oo_symbols,
}

if df_protect is not None and not df_protect.empty:
    s = set(df_pf[df_pf.symbol.isin(df_pf.symbol[df_pf.secType == 'OPT']) &
                             df_pf.symbol.isin(df_pf.symbol[df_pf.secType == 'STK'])] \
                             .reset_index(drop=True)['symbol'])

    df_pf[(df_pf.secType == 'STK') & ~(df_pf.symbol.isin(s))].reset_index(drop=True)

    print('DOWNSIDE PROTECTION')
    print('======================')
    print(f"Protects: ${df_protect.protection.sum():,.0f}")
    print(f"Cost: ${df_protect.cost.sum():,.0f} for dte: {df_protect.dte.mean():.1f} days")

display(HTML('<hr>'))

print('\n')
print('BASE INTEGRITY CHECK')
print('====================')

for k, v in unique_symbols_count.items():
    if v:
        print(f'{k}: {v}')

# Find symbols missing in unds, chains, and pf
missing_in_unds = chains[~chains['symbol'].isin(df_unds['symbol'])]
missing_in_chains = df_unds[~df_unds['symbol'].isin(chains['symbol'])]
missing_in_chains_from_pf = df_pf[~df_pf['symbol'].isin(chains['symbol'])]

# Display missing symbols as lists
print('\n')
print("Symbols missing in unds from chains:", missing_in_unds['symbol'].unique())
print("Symbols missing in chains from unds:", missing_in_chains['symbol'].unique())
print("Symbols missing in chains from pf:", missing_in_chains_from_pf['symbol'].unique())
# %%
