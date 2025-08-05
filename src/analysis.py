# %%
# GETS OVERALL FINANCIALS

import numpy as np
import pandas as pd
from IPython.display import HTML, display

from ibfuncs import get_financials, get_ib, get_open_orders, ib_pf
from snp import make_snp_unds
from utils import (ROOT, classify_open_orders, classify_pf, do_i_refresh,
                   get_dte, get_pickle, get_age_text, load_config,
                   update_unds_status)

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
    df_unds = make_snp_unds()


with get_ib("SNP") as ib:
    qpf = ib_pf(ib)
    df_pf = classify_pf(qpf)

    fin = ib.run(get_financials(ib))
    fin["unique symbols"] = len(df_pf.symbol.unique())
    for k, v in list(fin.items()):
        if k == 'stocks':
            fin[f"...{len(df_pf[df_pf.secType == 'STK'])} stocks"] = df_pf[df_pf.secType == 'STK'].mktVal.sum()
            del fin[k]
    fin[f"...{len(df_pf[df_pf.secType == 'OPT'])} options"] = df_pf[df_pf.secType == 'OPT'].mktVal.sum()
    

    openords = get_open_orders(ib)
    df_openords = classify_open_orders(openords, df_pf)
    df_unds = update_unds_status(df_unds=df_unds, df_pf=df_pf, df_openords=df_openords)

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

pf_states = {state: df_pf[df_pf.state == state].symbol.nunique() for state in set(df_pf.state)}
msg=' '.join(f"{state}: {n};" for state, n in pf_states.items())[:-1]
print('\nPortfolio has symbols that are ' + msg)


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

print('\nPosition Risk & Reward')
print('=======================')
risk_msg = []

stocks_val = df_pf[df_pf.symbol.isin(df_pf[df_pf.state == 'protecting'].symbol)].mktVal.sum()
risk_msg.append(f'Our risk from {df_pf[df_pf.state == "protecting"].symbol.nunique()} protected stocks valued at ${stocks_val:,.0f} is ${df_risk.unprot_val.sum():,.0f} for {df_risk.dte.mean():.1f} days.')
risk_msg.append(f' ...We paid a risk premium of ${df_risk.cost.sum():,.0f}')
unprotected_stocks = df[(df.source == "und") & (df.state.isin(["unprotected", "exposed"]))].symbol.unique()

podf = df[(df.source == 'oo')&(df.state == 'protecting')].reset_index(drop=True)
oo_protect = sum(abs((podf.undPrice-podf.strike)*podf.qty)*100)
podf_mkt = df_pf[df_pf.symbol.isin(podf.symbol.unique()) & (df_pf.secType == 'STK')].mktVal.sum()


if unprotected_stocks.size > 0:
    risk_msg.append(f'\n{len(unprotected_stocks)} Stocks needing protection: {", ".join(unprotected_stocks)}')
    if df_protect is not None and not df_protect.empty:
        dprot = df_protect[df_protect.symbol.isin(unprotected_stocks)]
        protection = dprot.protection.sum()
        protection_price = dprot.xPrice.sum()
        risk_msg.append(f'...We are protected below fall of ${protection:,.0f} at ${protection_price:,.0f} in unplaced orders, lasting for {df_protect.dte.mean():.1f} days on average.')
else:
    risk_msg.append(f'\nRemaining stock positions worth ${podf_mkt:,.0f} are protected!')
    risk_msg.append(f' ...protection of ${oo_protect:,.0f} from {len(podf)} open orders will be at the cost of ${sum(podf.avgCost*podf.qty):,.0f}')

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
# ANALYZES THE BASE DATA QUALITY
# Count unique symbols in each DataFrame
unique_unds_symbols = len(df_unds['symbol'].unique())
unique_chains_symbols = chains['symbol'].nunique() if chains is not None else None
unique_oo_symbols = df_openords['symbol'].nunique() if 'symbol' in df_openords.columns else None
unique_pf_symbols = df_pf['symbol'].nunique() if 'symbol' in df_pf.columns else None

# Display the counts
unique_symbols_count = {
    'No of und symbols': (unique_unds_symbols, get_age_text(unds_path)),
    'No of chain symbols': (unique_chains_symbols if unique_chains_symbols is not None else 'N/A', get_age_text(chains_path)),
    'No of portfolio symbols': unique_pf_symbols,
    'No of open order symbols': unique_oo_symbols,
}

if df_protect is not None and not df_protect.empty:
    s = set(df_pf[df_pf.symbol.isin(df_pf.symbol[df_pf.secType == 'OPT']) &
                             df_pf.symbol.isin(df_pf.symbol[df_pf.secType == 'STK'])] \
                             .reset_index(drop=True)['symbol'])

    df_pf[(df_pf.secType == 'STK') & ~(df_pf.symbol.isin(s))].reset_index(drop=True)

    print('\nDOWNSIDE PROTECTION')
    print('======================')
    print(f"Protects: ${df_protect.protection.sum():,.0f}")
    print(f"Cost: ${df_protect.cost.sum():,.0f} for dte: {df_protect.dte.mean():.1f} days")

# display(HTML('<hr>'))

print('\n')
print('BASE INTEGRITY CHECK')
print('====================')

for k, v in unique_symbols_count.items():
    if v:
        print(f'{k}: {v}')

if chains is not None and df_unds is not None:
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
