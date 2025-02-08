# %%
# @@@ CAUTION: This script places orders @@@

from utils import get_prec, get_pickle, ROOT, load_config, pickle_me
from ibfuncs import get_ib, get_financials
import pandas as pd
from ib_async import LimitOrder
from typing import List
from tqdm import tqdm
from utils import delete_pkl_files
import numpy as np

cov_path = ROOT / "data" / "df_cov.pkl"  # covered call and put path
nkd_path = ROOT / "data" / "df_nkd.pkl"
reap_path = ROOT / "data" / "df_reap.pkl"
protect_path = ROOT / "data" / "df_protect.pkl"

config = load_config('SNP')

MINCUSHION = config.get("MINCUSHION")

def make_ib_orders(df: pd.DataFrame, action: str) -> tuple:
    """Make (contract, order) tuples"""

    contracts = df.contract.to_list()
    orders = [
        LimitOrder(action=action, totalQuantity=abs(int(q)), lmtPrice=get_prec(p, 0.01))
        for q, p in zip(df.qty, df.xPrice)
    ]

    cos = tuple((c, o) for c, o in zip(contracts, orders))

    return cos

def place_orders(cos: tuple, blk_size: int=25) -> List:
    """CAUTION: This places trades in the system !!!"""
    
    trades = []

    cobs = {cos[i : i + blk_size] for i in range(0, len(cos), blk_size)}

    with get_ib('SNP') as ib:
        for b in tqdm(cobs):
            for c, o in b:
                td = ib.placeOrder(c, o)
                trades.append(td)
            ib.sleep(0.75)

    return trades

# %%
# ORDER COVER OPTIONS
if (df_cov_path := cov_path).exists():
    df_cov = get_pickle(df_cov_path)
    cos = make_ib_orders(df_cov, action='SELL')
    cov_trades = place_orders(cos)
    pickle_me(cov_trades, ROOT / "data" / "traded_covers.pkl")
    print(f'\nPlaced {len(df_cov)} cover orders')
    delete_pkl_files(['df_cov.pkl'])
else:
    print('\nThere are no covers\n')

# %%
# ORDER REAP OPTIONS
if (df_reap_path := reap_path).exists():
    df_reap = get_pickle(df_reap_path)
    reap_cos = make_ib_orders(df_reap, action='BUY')
    reap_trades = place_orders(reap_cos)
    print(f'\nPlaced {len(df_reap)} reaped options')
    pickle_me(reap_trades, ROOT / "data" / "traded_reaps.pkl")
    delete_pkl_files(['df_reap.pkl'])
else:
    print("\nThere are no options to be reaped\n")

# %%
# ORDER NAKEDS BASED ON CUSHION
if (df_nkd_path := nkd_path).exists():
    with get_ib('SNP') as ib:
        fin = ib.run(get_financials(ib))
        cushion = fin.get('cushion', np.nan)
        ib.disconnect()
    if cushion < MINCUSHION:
        print(f"Cushion: {cushion:.2f} < MINCUSHION: {MINCUSHION:.2f}, not placing naked orders")
    else:
        df_nkd = get_pickle(df_nkd_path)
        nkd_cos = make_ib_orders(df_nkd, action='SELL')
        nkd_trades = place_orders(nkd_cos)
        print(f'\nPlaced {len(df_nkd)} naked options')
        pickle_me(nkd_trades, ROOT / "data" / "traded_nakeds.pkl")
        delete_pkl_files(['df_nkd.pkl'])
else:
    print("\nThere are no nakeds\n")

# %%
# ORDER PROTECT OPTIONS
if (df_protect_path := protect_path).exists():
    df_protect = get_pickle(df_protect_path)
    protect_cos = make_ib_orders(df_protect, action='BUY')
    protect_trades = place_orders(protect_cos)
    print(f'\nPlaced {len(df_protect)} protect options')
    pickle_me(protect_trades, ROOT / "data" / "traded_protects.pkl")
    delete_pkl_files(['df_protect.pkl'])
else:
    print("\nThere are no protect options\n")

# %%
# %%
# RUN ANALYSIS
exec(open(ROOT / "src" / "analysis.py").read())

