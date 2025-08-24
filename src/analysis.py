# %%
# GETS OVERALL FINANCIALS

import numpy as np
import pandas as pd

from ibfuncs import get_financials, get_ib, get_open_orders, ib_pf
from snp import make_snp_unds
from utils import (ROOT, classify_open_orders, classify_pf, do_i_refresh,
                   get_dte, get_pickle, get_age_text, load_config,
                   update_unds_status, get_assignment_risk)

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

config = load_config("SNP")
MAX_FILE_AGE = config.get("MAX_FILE_AGE")
REAPRATIO = config.get("REAPRATIO")
COVER_ME = config.get("COVER_ME")
PROTECT_ME = config.get("PROTECT_ME")
REAP_ME = config.get("REAP_ME")

# Get unds. Make it fresh if stale.
if do_i_refresh(unds_path, max_days=MAX_FILE_AGE):
    df_unds = make_snp_unds()
else:
    df_unds = get_pickle(unds_path)


with get_ib("SNP") as ib:
    qpf = ib_pf(ib)
    df_pf = classify_pf(qpf)
    df_pf = df_pf.merge(df_unds[["symbol", "undPrice"]], on="symbol", how="left")

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

df_assign = get_assignment_risk(df)

# To-ve Blown cover rows in df_pf where secType is STK or (secType == OPT and position is negative)

cols = ['symbol', 'secType', 'position', 'right', 'dte', 'strike', 'undPrice', 'avgCost', 'mktVal', 'unPnL']
cover_condition = (
    (df.source == 'pf') & 
    (df.symbol.isin(df_assign[df_assign.state == 'covering'].symbol)) &
    ((df.secType == 'STK') | ((df.secType == 'OPT') & (df.position < 0)))
)
cover_blown = df[cover_condition].sort_values(
    ['symbol', 'right'], 
    ascending=[True, False]
)[cols]

df_sowed = df[df.state == 'sowed'].sort_values('unPnL')
cover_projection = (df_risk.dte.mean()/7-1)*abs(df_reward.premium.sum())
sowed_projection = df_sowed.avgCost.sum()*(1-REAPRATIO)
total_reward = cover_projection + abs(sowed_projection)

print('\nRISKS')
print('======')
risk_msg = []

if not PROTECT_ME:
    print('\nPROTECT_ME is disabled (false) in configuration')

pf_states = {state: df_pf[df_pf.state == state].symbol.nunique() for state in set(df_pf.state)}
msg=' '.join(f"{state}: {n};" for state, n in pf_states.items())[:-1]
print('\nPortfolio symbols states are... \n' + msg + '\n')

stocks_val = df_pf[df_pf.symbol.isin(df_pf[df_pf.state == 'protecting'].symbol)].mktVal.sum()

if not df_risk.empty:
    risk_msg.append(f'Our risk from {df_pf[df_pf.state == "protecting"].symbol.nunique()} protected stocks valued at ${stocks_val:,.0f} is ${df_risk.unprot_val.sum():,.0f} for {df_risk.dte.mean():.1f} days.')
    risk_msg.append(f' ...We paid a risk premium of ${df_risk.cost.sum():,.0f} to protect downside below the risk')

unprotected_stocks = df[(df.source == "und") & (df.state.isin(["unprotected", "exposed"]))].symbol.unique()

podf = df[(df.source == 'oo')&(df.state == 'protecting')].reset_index(drop=True)
oo_protect = sum(abs((podf.undPrice-podf.strike)*podf.qty)*100)
podf_mkt = df_pf[df_pf.symbol.isin(podf.symbol.unique()) & (df_pf.secType == 'STK')].mktVal.sum()

if unprotected_stocks.size > 0:
    stocks_str = [", ".join(unprotected_stocks[i:i+5]) for i in range(0, len(unprotected_stocks), 5)]
    risk_msg.append(f'\n{len(unprotected_stocks)} stocks need protection: \n...\n\t{"\n\t".join(stocks_str)}')
    if df_protect is not None and not df_protect.empty:
        dprot = df_protect[df_protect.symbol.isin(unprotected_stocks)]
        protection = dprot.protection.sum()
        protection_price = (dprot.xPrice*dprot.qty*100).sum()
        dprot_val = sum(dprot.undPrice*dprot.qty*100)
        risk_msg.append(f'\nFor {len(dprot)} stocks worth ${dprot_val:,.0f},')
        risk_msg.append(f' we can have protection band of ${protection:,.0f} for {df_protect.dte.mean():.1f} days,')
        risk_msg.append(f' at a cost of ${protection_price:,.0f} in unplaced orders,')
        if not PROTECT_ME:
            risk_msg.append(' provided PROTECT_ME is enabled in configuration')
elif podf_mkt > 0:
    risk_msg.append(f'\nRemaining stock positions worth ${podf_mkt:,.0f} are protected!')
    risk_msg.append(f' ...protection of ${oo_protect:,.0f} from {len(podf)} open orders will be at the cost of ${sum(podf.avgCost*podf.qty):,.0f}')

print('\n'.join(risk_msg))

print('\nREWARDS')
print('======')

naked_premium = 0
if not df_openords.empty:
    naked_premium = (df_openords.lmtPrice * df_openords.qty).sum() * 100

if not COVER_ME:
    print('\nCOVER_ME in configuration is disabled (false). No cover premiums are calculable!!]n')

reward_msg = (
    f'Total reward in a week is expected to be ${total_reward:,.0f}.\n'
    f' ..sowed reward in {df_sowed.dte.mean():.1f} dte days is ${sowed_projection:,.0f}\n'
    f' ..cover premiums in {df_reward.dte.mean():.1f} days are ${abs(df_reward.premium.sum()):,.0f}\n'
    f'  ...maximum cover reward  is '
    f'${df_reward.max_reward.sum():,.0f}, if all covers get blown.\n'
    
)

if naked_premium > 0:
    reward_msg += f'\n ..naked premiums from open orders is ${naked_premium:,.0f}\n'

print(reward_msg)

if cover_blown is not None and not cover_blown.empty:
    print('\nCover blown rows are:\n')
    print(cover_blown.to_string(index=False))

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
    print('ORDER premiums and profits from df_cov and df_nkd')
    print('=================================================')
    print("Total Premium available is", format(cov_premium + nkd_premium, ',.0f'))
    print(f"...Cover Premium: {format(cov_premium, ',.0f')}")
    print(f"...Naked Premiums: {format(nkd_premium, ',.0f')}\n")
    print(f"Max possible profit if all df_cov covers get blown: {format(maxProfit, ',.0f')}")

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
