import pandas as pd
import numpy as np
from utils import load_config, get_dte, get_pickle, ROOT

# Load configuration
config = load_config("SNP")

# Load data from pickle files
chains_path = ROOT / "data" / "df_chains.pkl"
unds_path = ROOT / "data" / "df_unds.pkl"

chains = get_pickle(chains_path)
df_unds = get_pickle(unds_path)

# Get virgin symbols (symbols not currently in portfolio)
df_v = df_unds[df_unds.state == 'virgin'].reset_index(drop=True)

# Filter chains for virgin symbols
df_virg_calls = chains.loc[
    chains.symbol.isin(df_v.symbol.to_list())
]

# Merge with underlying prices and volatilities
df_virg_calls = df_virg_calls.merge(
    df_unds[["symbol", "undPrice", "vy"]], 
    on="symbol", 
    how="left"
)

# Calculate standard deviation based on implied volatility and days to expiration
df_virg_calls["sdev"] = df_virg_calls.undPrice * df_virg_calls.vy * (df_virg_calls.dte / 365) ** 0.5

# Get VIRGIN_CALL_STD_MULT from config (default to 3 if not specified)
v_std = config.get("VIRGIN_CALL_STD_MULT", 3)
no_of_options = 4

# Sort df_virg_calls.strike, with ascending = False, grouped on symbol and expiry
df_virg_calls = df_virg_calls.sort_values(
    ["symbol", "expiry", "strike"], 
    ascending=[True, True, False]
)

# Get call shorts for virgin symbols
virg_short_calls = (
    df_virg_calls.groupby(["symbol", "expiry"])
    .apply(
        lambda x: x[x["strike"] > x["undPrice"] + v_std * x["sdev"]]
        .assign(diff=abs(x["strike"] - (x["undPrice"] + v_std * x["sdev"])))
        .nsmallest(no_of_options, "diff")
    )
    .reset_index(drop=True)
)

# Print results for inspection
print("Virgin Symbols:", df_v)
print("\nVirgin Call Options:", df_virg_calls)
print("\nShort Virgin Calls:", virg_short_calls)