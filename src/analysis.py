# Get financials, with anticipated profit and premiums

import numpy as np
from ibfuncs import ib_pf, get_open_orders
from utils import classify_pf, classify_open_orders, update_unds_status

from ibfuncs import get_financials, get_ib
from utils import ROOT, get_pickle

# Get the dataframes
pf_path = ROOT / "data" / "df_pf.pkl"  # portfolio path
cov_path = ROOT / "data" / "df_cov.pkl"  # covered call and put path
nkd_path = ROOT / "data" / "df_nkd.pkl"
unds_path = ROOT / "data" / "df_unds.pkl"
protect_path = ROOT / "data" / "df_protect.pkl"

df_pf = get_pickle(pf_path)
df_cov = get_pickle(cov_path)
df_nkd = get_pickle(nkd_path)
df_unds = get_pickle(unds_path)
df_protect = get_pickle(protect_path)

with get_ib("SNP") as ib:
    fin = ib.run(get_financials(ib))
    qpf = ib_pf(ib)
    df_pf = classify_pf(qpf)
    openords = get_open_orders(ib)
    df_openords = classify_open_orders(openords, df_pf)
    df_unds = update_unds_status(df_unds=df_unds, df_pf=df_pf, df_openords=df_openords)

# Analyze covered calls and puts
if df_cov is not None and not df_cov.empty:
    cost = (df_cov.avgCost * df_cov.qty * 100).sum()
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
    cost = 0
    cov_premium = 0
    maxProfit = 0

if df_nkd is not None and not df_nkd.empty:
    nkd_premium = (df_nkd.xPrice * 100 * df_nkd.qty).sum()
else:
    nkd_premium = 0

print("FINANCIALS")
print('==========')
for k, v in fin.items():
    if v:
        if v > 1:
            print(f"{k}: {format(v, ',.0f')}")
        else:
            print(f"{k}: {format(v, ',.2f')}")

print('\n')
print('PREMIUMS AND PROFIT from df_cov and df_nkd')
print('==========================================')
print(f"Total Premium", format(cov_premium + nkd_premium, ',.0f'))
print(f"Position Cost: {format(cost, ',.0f')}")
print(f"Cover Premium: {format(cov_premium, ',.0f')}")
print(f"Naked Premiums: {format(nkd_premium, ',.0f')}")
print(f"Max Profit: {format(maxProfit, ',.0f')}")

s = set(df_pf[df_pf.symbol.isin(df_pf.symbol[df_pf.secType == 'OPT']) &
                         df_pf.symbol.isin(df_pf.symbol[df_pf.secType == 'STK'])] \
                         .reset_index(drop=True)['symbol'])

df_pf[(df_pf.secType == 'STK') & ~(df_pf.symbol.isin(s))].reset_index(drop=True)

print('\n')
print('DOWNSIDE PROTECTION')
print('======================')
print(f"Protects: ${df_protect.protection.sum():,.0f}")
print(f"Cost: ${df_protect.cost.sum():,.0f} for dte: {df_protect.dte.mean():.1f} days")