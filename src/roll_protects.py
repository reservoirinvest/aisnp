# %% ROLLING PROTECT RECOMMENDATIONS

import time

from ib_async import Option, util
from loguru import logger

from ibfuncs import (df_prices, get_ib, get_open_orders, ib_pf, qualify_me)
from snp import make_snp_unds
from utils import (ROOT, classify_open_orders, classify_pf, clean_ib_util_df,
                   do_i_refresh, filter_closest_dates, filter_closest_strikes,
                   get_pickle, update_unds_status, load_config, get_dte)

# Start timing the script execution
start_time = time.time()

unds_path = ROOT / "data" / "df_unds.pkl"
chains_path = ROOT / "data" / "df_chains.pkl"
cov_path = ROOT / "data" / "df_cov.pkl"  # covered call and put path
nkd_path = ROOT / "data" / "df_nkd.pkl"
pf_path = ROOT / "data" / "df_pf.pkl"
reap_path = ROOT / "data" / "df_reap.pkl"

config = load_config("SNP")
PROTECT_DTE = config.get('PROTECT_DTE')

# Delete logs
log_file_path = ROOT / "log" / "states.log"
logger.add(log_file_path, rotation="2 days")
util.logToFile(log_file_path, level=40) # IB ERRORs only logged.

if do_i_refresh(unds_path, max_days=30):
    df_unds = make_snp_unds()
else:
    df_unds = get_pickle(unds_path)

with get_ib("SNP") as ib:
    qpf = ib_pf(ib)
    df_pf = classify_pf(qpf)
    openords = get_open_orders(ib)
    df_openords = classify_open_orders(openords, df_pf)
    df_unds = update_unds_status(df_unds=df_unds, df_pf=df_pf, df_openords=df_openords)

#%%
# insert undPrice in df_pf from df_unds based on df_unds.symbol
df_pf = df_pf.set_index('symbol').join(df_unds.set_index('symbol')[['undPrice']]).reset_index()

#%% ROLLS FOR PROTECTING PUTS

# Filter for protecting put options and calculate % difference between strike and underlying price
df_rol = (
    df_pf[(df_pf['state'] == 'protecting') & (df_pf.right == 'P')]
    .assign(
        pct_diff=lambda x: (abs(x['strike'] - x['undPrice']) / x['undPrice'] * 100)
    )
    .sort_values('pct_diff', ascending=False)
    .reset_index(drop=True)
)

# Load chains data
rol_chains = get_pickle(chains_path)

rol_chains = rol_chains[rol_chains.symbol.isin(set(df_rol.symbol))]

# Append undPrice
rol_chains = rol_chains.set_index('symbol').join(df_unds.set_index('symbol')[['undPrice']]).reset_index()

# Filter for closest dates and strikes
df_cd = filter_closest_dates(rol_chains, PROTECT_DTE, num_dates=1)
p = filter_closest_strikes(df_cd, n=-4)
p['right'] = 'P'

p['contract'] = p.apply(
    lambda x: Option(x.symbol, x.expiry, x.strike, x.right, 'SMART'),
    axis=1
)

with get_ib("SNP") as ib:
    purls = ib.run(qualify_me(ib, p.contract, desc="Qualifying long put rolls"))

# Get the qualified put rolls
df_purl = clean_ib_util_df([p for p in purls if p is not None])
df_purl = df_purl.groupby('symbol').first().reset_index()

# Get prices put rolls
with get_ib("SNP") as ib:
    purl_price = ib.run(df_prices(
            ib=ib,
            stocks=df_purl['contract'].tolist(),
            sleep_time=10,
            msg="put roll prices",
        ))

# Append undPrice to purl_price and compute difference
purls = purl_price.merge(df_unds[['symbol', 'undPrice']], on='symbol')
purls['diff'] = (purls['strike'] / purls['undPrice'] - 1)
purls = purls.sort_values('diff')

cols = ['symbol', 'secType', 'expiry', 'strike', 'undPrice', 'right',
       'price', 'diff']

# Check if there are any purls whose diff is less than -0.05
if (purls['diff'] < -0.05).any():
    print("WARNING: There are some put rolls whose strike-undPrice is larger than 5%. "
          "These will be taken out from auto-roll suggestion.")
    print(purls[purls['diff'] < -0.05][cols])

purls1 = purls[purls['diff'] >= -0.05]

# Append position/100 to purls1 and call it qty
purls1 = purls1.copy()
purls1['qty'] = purls1['symbol'].map(df_unds.set_index('symbol')['position'] / 100)

purls1 = purls1.merge(df_pf[(df_pf.secType == 'OPT') & (df_pf.right == 'P') & (df_pf.position > 0)][['symbol', 'mktPrice']], on='symbol', how='left')
purls1.rename(columns={'mktPrice': 'cost'}, inplace=True)

rollover_cost = (purls1.price - purls1.cost) * purls1.qty * 100
print(f"\nThe rollover cost to {purls1.expiry.apply(get_dte).max():.0f} days would be ${rollover_cost.sum():,.0f}")


# %%
