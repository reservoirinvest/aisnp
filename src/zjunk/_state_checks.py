# Test reconfigured classifications and updates.
# ----------------------------------------------
# %%
# Imports
import numpy as np
import pandas as pd

from utils import ROOT, get_pickle
from ibfuncs import get_ib, ib_pf

unds_path = ROOT / "data" / "df_unds.pkl"
chains_path = ROOT / "data" / "df_chains.pkl"
cov_path = ROOT / "data" / "df_cov.pkl"  # covered call and put path
nkd_path = ROOT / "data" / "df_nkd.pkl"
pf_path = ROOT / "data" / "df_pf.pkl"
reap_path = ROOT / "data" / "df_reap.pkl"

df_unds = get_pickle(unds_path)
chains = get_pickle(chains_path)

df_cov = get_pickle(cov_path)
df_nkd = get_pickle(nkd_path)
df_reap = get_pickle(reap_path)

#%%
with get_ib("SNP") as ib:
    pf = ib_pf(ib)

# %%
def find_straddles(df):
    straddle_mask = np.zeros(len(df), dtype=bool)
    
    for (symbol, expiry, strike), group in df.groupby(["symbol", "expiry", "strike"]):
        if (len(group) == 2 and 
            set(group["right"]) == {"C", "P"} and 
            np.sign(group["position"].iloc[0]) == np.sign(group["position"].iloc[1])):
            straddle_mask[group.index] = True
    
    return straddle_mask

def classify_pf(pf):
    pf = pf.copy()
    pf["state"] = "tbd"

    states_classification = [
        {
            "name": "straddled",
            "mask": find_straddles(pf)
        },
        {
            "name": "covering",
            "mask": (
                (pf.secType == "OPT") & 
                ((pf.right == "C") & (pf.position < 0) | (pf.right == "P") & (pf.position < 0))
            )
        },
        {
            "name": "protecting",
            "mask": (
                (pf.secType == "OPT") & 
                ((pf.right == "C") & (pf.position > 0) | (pf.right == "P") & (pf.position > 0))
            )
        },
        {
            "name": "orphaned",
            "mask": (
                (pf.secType == "OPT") &
                (pf.position > 0) &
                (~pf.symbol.isin(pf[pf.secType == "STK"].symbol))
            )
        },
        {
            "name": "sowed",
            "mask": (
                (pf.secType == "OPT") &
                (pf.position < 0) &
                (~pf.symbol.isin(pf[pf.secType == "STK"].symbol))
            )
        },
        {
            "name": "solid",
            "mask": (
                (pf.secType == "STK") &
                pf.symbol.isin(pf[(pf.secType == "OPT") & (pf.state == "covering")].symbol) &
                pf.symbol.isin(pf[(pf.secType == "OPT") & (pf.state == "protecting")].symbol)
            )
        },
        {
            "name": "unprotected",
            "mask": (
                (pf.secType == "STK") &
                pf.symbol.isin(pf[(pf.secType == "OPT") & (pf.state == "covering")].symbol) &
                ~pf.symbol.isin(pf[(pf.secType == "OPT") & (pf.state == "protecting")].symbol)
            )
        },
        {
            "name": "uncovered",
            "mask": (
                (pf.secType == "STK") &
                ~pf.symbol.isin(pf[(pf.secType == "OPT") & (pf.state == "covering")].symbol) &
                pf.symbol.isin(pf[(pf.secType == "OPT") & (pf.state == "protecting")].symbol)
            )
        },
        {
            "name": "exposed",
            "mask": (
                (pf.secType == "STK") &
                (pf.position > 0) &
                ~pf.symbol.isin(
                    pf[
                        (pf.secType == "OPT") & 
                        (pf.state.isin(["covering", "protecting"]))
                    ].symbol
                )
            )
        }
    ]

    # Apply states in order of priority
    for state_config in states_classification:
        pf.loc[state_config["mask"], "state"] = state_config["name"]

    # Fallback for any remaining 'tbd' states
    pf.loc[pf.state == "tbd", "state"] = "unclassified"

    return pf
# %%
df = classify_pf(pf)
df_pf = classify_pf(pf[pf.symbol == 'ADP'])
# %%
