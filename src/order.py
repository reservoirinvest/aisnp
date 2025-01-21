from utils import get_prec, get_pickle, ROOT, load_config, pickle_me
from ibfuncs import get_ib
import pandas as pd
from ib_async import LimitOrder
from typing import List
from tqdm import tqdm

MINEXPOPTPRICE = load_config('SNP')['MINEXPOPTPRICE']

def make_ib_orders(df: pd.DataFrame) -> tuple:
    """Make (contract, order) tuples"""

    contracts = df.contract.to_list()
    orders = [
        LimitOrder(action="SELL", totalQuantity=abs(int(q)), lmtPrice=get_prec(p, 0.01))
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

if (df_cov_path := ROOT / "data" / "df_cov.pkl").exists():
    df_cov = get_pickle(df_cov_path)
    cos = make_ib_orders(df_cov)
    cov_trades = place_orders(cos)
    pickle_me(cov_trades, ROOT / "data" / "cover_trades.pkl")
    df_cov_path.unlink()
    print(f'Placed {len(df_cov)} cover orders\n')
else:
    print('There are no covers\n')

if (df_nkd_path := ROOT / "data" / "df_nkd.pkl").exists():
    df_nkd = get_pickle(df_nkd_path)
    nkd_cos = make_ib_orders(df_nkd)
    nkd_trades = place_orders(nkd_cos)
    pickle_me(nkd_trades, ROOT / "data" / "nkd_trades.pkl")
    df_nkd_path.unlink()
    print(f'Placed {len(df_nkd)} naked options\n')
else:
    print("There are no nakeds\n")

if (df_reap_path := ROOT / "data" / "df_reap.pkl").exists():
    df_reap = get_pickle(df_reap_path)
    reap_cos = make_ib_orders(df_reap)
    reap_trades = place_orders(reap_cos)
    pickle_me(reap_trades, ROOT / "data" / "reap_trades.pkl")
    df_reap_path.unlink()
    print(f'Placed {len(df_reap)} reaped options\n')
else:
    print("There are no sowed options\n")