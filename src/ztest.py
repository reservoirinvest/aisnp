from ibfuncs import get_ib, ib_pf, get_open_orders
from utils import ROOT, get_pickle, get_dte
import pandas as pd
import numpy as np

df_unds = get_pickle(ROOT / "data" / "df_unds.pkl")
with get_ib("SNP") as ib:
    df_pf = ib_pf(ib)
    df_openords = get_open_orders(ib)

df = pd.concat([df_unds.assign(source='unds'), 
        df_pf.assign(source='pf'), 
        df_openords.assign(source='oo')], ignore_index=True)

df = df.assign(status='unknown', dte=df.expiry.apply(lambda x: get_dte(x) if pd.notna(x) and x else np.nan))
cols = ['source', 'symbol', 'conId', 'secType', 'undPrice', 'strike', 'mktPrice', 'lmtPrice', 
        'right', 'action', 'expiry', 'dte', 'position', 'qty', 'vy' ]

df = df[cols]
