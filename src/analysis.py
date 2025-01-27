# %%
# # Get financials, with anticipated profit and premiums

import numpy as np

from ibfuncs import get_financials, get_ib, get_open_orders, ib_pf
from utils import (ROOT, classify_open_orders, classify_pf, get_pickle,
                   how_many_days_old, update_unds_status)

from IPython.display import display, HTML
display(HTML('<style>pre { white-space: pre-wrap; }</style>'))


# Get the dataframes
pf_path = ROOT / "data" / "df_pf.pkl"  # portfolio path
cov_path = ROOT / "data" / "df_cov.pkl"  # covered call and put path
nkd_path = ROOT / "data" / "df_nkd.pkl"
unds_path = ROOT / "data" / "df_unds.pkl"
protect_path = ROOT / "data" / "df_protect.pkl"
reap_path = ROOT / "data" / "df_reap.pkl"
chains_path = ROOT / "data" / "chains.pkl"

df_pf = get_pickle(pf_path, print_msg=False)
df_cov = get_pickle(cov_path, print_msg=False)
df_nkd = get_pickle(nkd_path, print_msg=False)
df_unds = get_pickle(unds_path, print_msg=False)
df_protect = get_pickle(protect_path, print_msg=False)
df_reap = get_pickle(reap_path, print_msg=False)
chains = get_pickle(chains_path, print_msg=False)

# %%
with get_ib("SNP") as ib:
    fin = ib.run(get_financials(ib))
    qpf = ib_pf(ib)
    df_pf = classify_pf(qpf)
    openords = get_open_orders(ib)
    df_openords = classify_open_orders(openords, df_pf)
    df_unds = update_unds_status(df_unds=df_unds, df_pf=df_pf, df_openords=df_openords)

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

print("\nFINANCIALS")
print('==========')
for k, v in fin.items():
    if v:
        if v > 1:
            print(f"{k}: {format(v, ',.0f')}")
        else:
            print(f"{k}: {format(v, ',.2f')}")

if cov_premium > 0 or nkd_premium > 0:
    print('\n')
    print('PREMIUMS AND PROFIT from df_cov and df_nkd')
    print('==========================================')
    print(f"Total Premium", format(cov_premium + nkd_premium, ',.0f'))
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

    print('\n')
    print('DOWNSIDE PROTECTION')
    print('======================')
    print(f"Protects: ${df_protect.protection.sum():,.0f}")
    print(f"Cost: ${df_protect.cost.sum():,.0f} for dte: {df_protect.dte.mean():.1f} days")

display(HTML('<hr>'))

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
