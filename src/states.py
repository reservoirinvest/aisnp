# This program identifies states and generates orders

import numpy as np
import pandas as pd
from ib_async import MarketOrder, Option

from ibfuncs import df_chains, df_iv, get_ib, get_open_orders, ib_pf, margin, qualify_me
from snp import snp_qualified_und_contracts
from utils import (
    ROOT,
    classify_open_orders,
    classify_pf,
    clean_ib_util_df,
    configure_logging,
    do_i_refresh,
    get_pickle,
    load_config,
    pickle_me,
    update_unds_status,
)

configure_logging()

unds_ct_path = ROOT / "data" / "und_contracts.pkl"
unds_path = ROOT / "data" / "df_unds.pkl"
chains_path = ROOT / "data" / "chains.pkl"
pf_path = ROOT / "data" / "df_pf.pkl"
cov_path = ROOT / "data" / "df_cov.pkl"  # covered call and put path

unds = get_pickle(unds_path)
chains = get_pickle(chains_path)
df_pf = get_pickle(pf_path)
df_unds = get_pickle(unds_path)


# Get unds. Make it fresh if stale.
if do_i_refresh(unds_ct_path, max_days=1):
    unds = snp_qualified_und_contracts(unds_path=unds_ct_path, fresh=True)
else:
    unds = get_pickle(unds_ct_path)

dfu = clean_ib_util_df(unds)

# Get und prices, volatilities
with get_ib("SNP") as ib:
    dfp = ib.run(
        df_iv(
            ib=ib,
            stocks=dfu["contract"].tolist(),
            sleep_time=10,
            msg="gets undPrices and Vy",
        )
    )

# Get portfolio and open orders
with get_ib("SNP") as ib:
    qpf = ib_pf(ib)
    df_pf = classify_pf(qpf)

    df_openords = get_open_orders(ib, is_active=False)
    df_openords = classify_open_orders(df_openords, df_pf)

# Update df_unds undPrice
dfu["undPrice"] = dfu.merge(
    qpf[qpf.secType == "STK"][["symbol", "mktPrice"]], on="symbol", how="left"
)["mktPrice"]

dfu.loc[dfu.undPrice.isnull(), "undPrice"] = dfu.merge(
    dfp[["symbol", "price"]], on="symbol", how="left"
)["price"]

# Merge volatility data
dfu = dfu.merge(dfp[["symbol", "hv", "iv"]], on="symbol", how="left")

# ..create 'vy' field that shows 'iv' or 'hv' if 'iv' is NaN
dfu["vy"] = dfu["iv"].combine_first(dfu["hv"])

dfu = pd.concat(
    [
        dfu,
        dfu.merge(
            qpf[qpf.secType == "STK"][
                [
                    "symbol",
                    "position",
                    "mktPrice",
                    "mktVal",
                    "avgCost",
                    "unPnL",
                    "rePnL",
                ]
            ],
            on="symbol",
            how="left",
        )[["position", "mktPrice", "mktVal", "avgCost", "unPnL", "rePnL"]],
    ],
    axis=1,
)

#  Establish status for pf and df_unds

df_unds = classify_pf(dfu)

# ..update status for symbols not in qpf
df_unds.loc[~df_unds.symbol.isin(qpf.symbol), "state"] = "virgin"

df_unds = df_unds.drop(
    columns=["iv", "hv", "expiry", "strike", "right"], errors="ignore"
)

# ..apply the status update to df_unds and pickle
df_unds = update_unds_status(df_unds, df_pf, df_openords)

pickle_me(df_unds, unds_path)
pickle_me(df_pf, pf_path)

#  Get chains
if do_i_refresh(chains_path, max_days=1):
    chain_recreate = True
else:
    chain_recreate = False

if chain_recreate:
    with get_ib("SNP") as ib:
        chains = ib.run(df_chains(ib, unds, msg="raw chains"))
        unds1 = clean_ib_util_df(unds)
        missing_unds = unds1[~unds1["symbol"].isin(chains["symbol"])]
        if not missing_unds.empty:
            additional_chains = ib.run(
                df_chains(ib, missing_unds.contract.to_list(), msg="missing chains")
            )
            chains = pd.concat([chains, additional_chains], ignore_index=True)
            pickle_me(chains, chains_path)
else:
    chains = pd.read_pickle(chains_path)


# Make covered calls for 'exposed' and 'uncovered' positions


config = load_config("SNP")
COVER_MIN_DTE = config.get("COVER_MIN_DTE")

# Get exposed and uncovered long
uncov = df_unds.state.isin(["exposed", "uncovered"])
uncov_long = df_unds[uncov & (df_unds.position > 0)].reset_index(drop=True)

# Ready the chains for portfolio symbols
df_cc = (
    chains[chains.symbol.isin(uncov_long.symbol.unique())]
    .loc[(chains.dte.between(COVER_MIN_DTE, COVER_MIN_DTE + 7))][
        ["symbol", "expiry", "strike", "dte"]
    ]
    .sort_values(["symbol", "dte"])
    .reset_index(drop=True)
)

# Merge chains with underlying prices and volatilities
df_cc = df_cc.merge(df_unds[["symbol", "undPrice", "vy"]], on="symbol", how="left")

# Calculate standard deviation based on implied volatility and days to expiration
df_cc["sdev"] = df_cc.undPrice * df_cc.vy * (df_cc.dte / 365) ** 0.5

# For each symbol and expiry, get 3 strikes above undPrice + sdev

c_std = config.get("COVER_STD_MULT")
no_of_options = 3

cc_long = (
    df_cc.groupby(["symbol", "expiry"])
    .apply(
        lambda x: x[x["strike"] > x["undPrice"] + c_std * x["sdev"]]
        .assign(diff=abs(x["strike"] - (x["undPrice"] + c_std * x["sdev"])))
        .sort_values("diff")
        .head(no_of_options),
        include_groups=False,
    )
    .reset_index()
    .drop(columns=["level_2", "diff"])
)

# Make long covered call options
cov_calls = [
    Option(s, e, k, "C", "SMART")
    for s, e, k in zip(cc_long.symbol, cc_long.expiry, cc_long.strike)
]

with get_ib("SNP") as ib:
    ib.run(qualify_me(ib, cov_calls))

df_cc1 = clean_ib_util_df([c for c in cov_calls if c.conId > 0])


# Get the lower of the long covered call
df_ccf = df_cc1.loc[df_cc1.groupby("symbol")["strike"].idxmin()]

df_ccf = df_ccf.reset_index(drop=True)

# Append undPrice and vy from df_unds
df_ccf = df_ccf.merge(df_unds[["symbol", "undPrice", "vy"]], on="symbol", how="left")

# Integrate position and avgCost from df_pf into df_ccf
df_ccf = df_ccf.merge(df_pf[["symbol", "position", "avgCost"]], on="symbol", how="left")

# Make qty field as position/100
df_ccf["action"] = "SELL"
df_ccf["qty"] = df_ccf["position"] / 100
df_ccf = df_ccf.drop(columns=["position"])


# Get covered call prices, volatilities
with get_ib("SNP") as ib:
    dfx = ib.run(
        df_iv(
            ib=ib,
            stocks=df_ccf["contract"].tolist(),
            sleep_time=10,
            msg="gets optPrices and Vy",
        )
    )


# Integrate dfx.price to df_ccf and determine action
df_ccf = df_ccf.merge(dfx[["symbol", "price"]], on="symbol", how="left")



# Use margin() function to calculate margins for contracts
with get_ib("SNP") as ib:
    # Create list of (contract, order) tuples
    co = list(
        zip(
            df_ccf.contract,
            [
                MarketOrder(action, qty)
                for action, qty in zip(df_ccf.action, df_ccf.qty)
            ],
        )
    )

    # Calculate margins asynchronously
    dfm = ib.run(margin(ib=ib, co=co, sleep_time=5.5, msg="covered call margins"))

    dfm = dfm.assign(
        comm=dfm[["commission", "maxCommission"]].min(axis=1),
        margin=dfm.initMarginChange.astype("float"),
    )

    # Correct unrealistic margin and commission
    dfm = dfm.assign(
        margin=np.where(dfm.margin > 1e7, np.nan, dfm.margin),
        comm=np.where(dfm.comm > 1e7, np.nan, dfm.comm),
    ).drop(columns=["initMarginChange", "maxCommission", "commission"])


# Integrate dfm to df_ccf.
df_ccf = df_ccf.merge(dfm, left_on="conId", right_index=True, how="left")


# Make covered puts for 'exposed' and 'uncovered' short positions

# Get exposed and uncovered short
uncov_short = df_unds.state.isin(["exposed", "uncovered"])
uncov_short = df_unds[uncov_short & (df_unds.position < 0)].reset_index(drop=True)

# Ready the chains for portfolio symbols
df_cp = (
    chains[chains.symbol.isin(uncov_short.symbol.unique())]
    .loc[(chains.dte.between(COVER_MIN_DTE, COVER_MIN_DTE + 7))][
        ["symbol", "expiry", "strike", "dte"]
    ]
    .sort_values(["symbol", "dte"])
    .reset_index(drop=True)
)

# Merge chains with underlying prices and volatilities
df_cp = df_cp.merge(df_unds[["symbol", "undPrice", "vy"]], on="symbol", how="left")

# Calculate standard deviation based on implied volatility and days to expiration
df_cp["sdev"] = df_cp.undPrice * df_cp.vy * (df_cp.dte / 365) ** 0.5

# For each symbol and expiry, get 3 strikes below undPrice - sdev
cp_std = config.get("COVER_STD_MULT")
no_of_options = 3

cp_short = (
    df_cp.groupby(["symbol", "expiry"])
    .apply(
        lambda x: x[x["strike"] < x["undPrice"] - cp_std * x["sdev"]]
        .assign(diff=abs(x["strike"] - (x["undPrice"] - cp_std * x["sdev"])))
        .sort_values("diff")
        .head(no_of_options),
        include_groups=False,
    )
    .reset_index()
    .drop(columns=["level_2", "diff"])
)

# Make short covered put options
cov_puts = [
    Option(s, e, k, "P", "SMART")
    for s, e, k in zip(cp_short.symbol, cp_short.expiry, cp_short.strike)
]

with get_ib("SNP") as ib:
    ib.run(qualify_me(ib, cov_puts))

df_cp1 = clean_ib_util_df([p for p in cov_puts if p.conId > 0])

# Get the higher of the short covered put
df_cpf = df_cp1.loc[df_cp1.groupby("symbol")["strike"].idxmax()]

df_cpf = df_cpf.reset_index(drop=True)

# Append undPrice and vy from df_unds
df_cpf = df_cpf.merge(df_unds[["symbol", "undPrice", "vy"]], on="symbol", how="left")

# Integrate position and avgCost from df_pf into df_cpf
df_cpf = df_cpf.merge(df_pf[["symbol", "position", "avgCost"]], on="symbol", how="left")

# Make qty field as abs(position)/100
df_cpf["action"] = "SELL"
df_cpf["qty"] = abs(df_cpf["position"]) / 100
df_cpf = df_cpf.drop(columns=["position"])

# Get covered put prices, volatilities
with get_ib("SNP") as ib:
    dfx_cp = ib.run(
        df_iv(
            ib=ib,
            stocks=df_cpf["contract"].tolist(),
            sleep_time=10,
            msg="gets optPrices and Vy for puts",
        )
    )

# Integrate dfx_cp.price to df_cpf and determine action
df_cpf = df_cpf.merge(dfx_cp[["symbol", "price"]], on="symbol", how="left")

# Calculate margins for covered puts
with get_ib("SNP") as ib:
    # Create list of (contract, order) tuples
    co_cp = list(
        zip(
            df_cpf.contract,
            [
                MarketOrder(action, qty)
                for action, qty in zip(df_cpf.action, df_cpf.qty)
            ],
        )
    )

    # Calculate margins asynchronously
    dfm_cp = ib.run(margin(ib=ib, co=co_cp, sleep_time=5.5, msg="covered put margins"))

    dfm_cp = dfm_cp.assign(
        comm=dfm_cp[["commission", "maxCommission"]].min(axis=1),
        margin=dfm_cp.initMarginChange.astype("float"),
    )

    # Correct unrealistic margin and commission
    dfm_cp = dfm_cp.assign(
        margin=np.where(dfm_cp.margin > 1e7, np.nan, dfm_cp.margin),
        comm=np.where(dfm_cp.comm > 1e7, np.nan, dfm_cp.comm),
    ).drop(columns=["initMarginChange", "maxCommission", "commission"])

# Integrate dfm_cp to df_cpf
df_cpf = df_cpf.merge(dfm_cp, left_on="conId", right_index=True, how="left")


# Integrate df_ccf and df_cpf into df_cov
df_cov = pd.concat([df_ccf, df_cpf], ignore_index=True)

# Pickle df_cov
pickle_me(df_cov, cov_path)


# Analyze covered calls and puts
cost = (df_cov.avgCost * df_cov.qty * 100).sum()
premium = (df_cov.price * 100 - df_cov.comm).sum()
maxProfit = (
    np.where(
        df_cov.right == "C",
        (df_cov.strike - df_cov.undPrice) * 100,
        (df_cov.undPrice - df_cov.strike) * 100,
    ).sum()
    + premium
)
