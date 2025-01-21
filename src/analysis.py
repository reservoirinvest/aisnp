# Get financials, with anticipated profit and premiums

import numpy as np

from ibfuncs import get_financials, get_ib
from utils import ROOT, get_pickle

with get_ib("SNP") as ib:
    d = ib.run(get_financials(ib))


# Get the dataframes
pf_path = ROOT / "data" / "df_pf.pkl"  # portfolio path
cov_path = ROOT / "data" / "df_cov.pkl"  # covered call and put path
nkd_path = ROOT / "data" / "df_nkd.pkl"
unds_path = ROOT / "data" / "df_unds.pkl"

df_pf = get_pickle(pf_path)
df_cov = get_pickle(cov_path)
df_nkd = get_pickle(nkd_path)
df_unds = get_pickle(unds_path)

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
for k, v in d.items():
    if v:
        if v > 1:
            print(f"{k}: {format(v, ',.0f')}")
        else:
            print(f"{k}: {format(v, ',.2f')}")

print('\n')
print('PREMIUMS AND PROFIT from df_cov and df_nkd')
print('========================================--')
print(f"Total Premium", format(cov_premium + nkd_premium, ',.0f'))
print(f"Position Cost: {format(cost, ',.0f')}")
print(f"Cover Premium: {format(cov_premium, ',.0f')}")
print(f"Naked Premiums: {format(nkd_premium, ',.0f')}")
print(f"Max Profit: {format(maxProfit, ',.0f')}")