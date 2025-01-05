#%%
# This program identifies states and generates orders

import pandas as pd

from ibfuncs import df_iv, get_ib, get_open_orders, ib_pf, df_chains
from snp import snp_qualified_und_contracts
from utils import (ROOT, classify_open_orders, classify_pf, clean_ib_util_df,
                   do_i_refresh, get_pickle, pickle_me, update_unds_status)

# Get unds. Make it fresh if stale.
unds_path = ROOT/'data'/'und_contracts.pkl'

if do_i_refresh(unds_path, max_days=1):
    unds = snp_qualified_und_contracts(unds_path=unds_path, fresh=True)
else:
    unds = get_pickle(unds_path)

#%%
dfu = clean_ib_util_df(unds)

# Get und prices, volatilities
with get_ib('SNP') as ib:
    dfp = ib.run(df_iv(ib=ib, 
        stocks=dfu['contract'].tolist(), 
        sleep_time=10, msg='gets undPrices and Vy'))

# %%Get portfolio and open orders
with get_ib('SNP') as ib:
    qpf = ib_pf(ib)
    df_pf = classify_pf(qpf)

    df_openords = get_open_orders(ib, is_active=False)
    df_openords = classify_open_orders(df_openords, df_pf)

# Update df_unds undPrice
dfu['undPrice'] = dfu.merge(qpf[qpf.secType == 'STK'][['symbol', 'mktPrice']], 
                    on='symbol', how='left')['mktPrice']

dfu.loc[dfu.undPrice.isnull(), 'undPrice'] = \
                dfu.merge(dfp[['symbol', 'price']], 
                    on='symbol', how='left')['price']

# Merge volatility data
dfu = dfu.merge(
    dfp[['symbol', 'hv', 'iv']], 
    on='symbol', 
    how='left'
)

# ..create 'vy' field that shows 'iv' or 'hv' if 'iv' is NaN
dfu['vy'] = dfu['iv'].combine_first(dfu['hv'])

dfu = pd.concat([
    dfu,
    dfu.merge(
        qpf[qpf.secType == 'STK'][['symbol', 'position', 'mktPrice', 'mktVal', 'avgCost', 'unPnL', 'rePnL']],
        on='symbol', how='left'
    )[['position', 'mktPrice', 'mktVal', 'avgCost', 'unPnL', 'rePnL']]
], axis=1)

# %% Establish status for pf and df_unds

df_unds = classify_pf(dfu)

# ..update status for symbols not in qpf
df_unds.loc[~df_unds.symbol.isin(qpf.symbol), 'state'] = 'virgin'

df_unds = df_unds.drop(columns=['iv', 'hv', 'expiry', 'strike', 'right'], 
                errors='ignore')

# ..apply the status update to df_unds and pickle
df_unds = update_unds_status(df_unds, df_pf, df_openords)

pickle_me(df_unds, ROOT/'data'/'df_unds.pkl')
pickle_me(df_pf, ROOT/'data'/'df_pf.pkl')

# %% Get chains
chains_path = ROOT/'data'/'chains.pkl'

if do_i_refresh(chains_path, max_days=1):
    chain_recreate = True
else:
    chain_recreate = False

if chain_recreate:

    with get_ib('SNP') as ib:
        chains = ib.run(df_chains(ib, unds, msg='raw chains'))
        unds1 = clean_ib_util_df(unds)
        missing_unds = unds1[~unds1['symbol'].isin(chains['symbol'])]
        if not missing_unds.empty:
            additional_chains = ib.run(df_chains(ib, missing_unds.contract.to_list(), msg='missing chains'))
            chains = pd.concat([chains, additional_chains], ignore_index=True)
            pickle_me(chains, chains_path)
else:
    chains = pd.read_pickle(chains_path)

