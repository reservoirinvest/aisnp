# This program identifies states and generates orders
# %%
# IMPORTS AND VARIABLES

import numpy as np

import pandas as pd
from ib_async import Option, util

from ibfuncs import (df_chains, df_iv, get_ib, get_open_orders, ib_pf,
                     qualify_me, get_financials)
from snp import snp_qualified_und_contracts
from utils import (ROOT, atm_margin, classify_open_orders, classify_pf,
                   clean_ib_util_df, do_i_refresh, get_dte,
                   get_pickle, load_config, pickle_me, update_unds_status,
                   is_market_open)

util.logToFile(ROOT / "log" / "states.log")
# configure_logging()

unds_ct_path = ROOT / "data" / "und_contracts.pkl"
unds_path = ROOT / "data" / "df_unds.pkl"
chains_path = ROOT / "data" / "chains.pkl"
pf_path = ROOT / "data" / "df_pf.pkl"  # portfolio path
oo_path = ROOT / "data" / "df_oo.pkl"  # open orders path
cov_path = ROOT / "data" / "df_cov.pkl"  # covered call and put path
nkd_path = ROOT / "data" / "df_nkd.pkl"

unds = get_pickle(unds_path)
df_unds = get_pickle(unds_path)
chains = get_pickle(chains_path)
df_pf = get_pickle(pf_path)
df_openords = get_pickle(oo_path)

config = load_config("SNP")
COVER_MIN_DTE = config.get("COVER_MIN_DTE")
VIRGIN_DTE = config.get("VIRGIN_DTE")
MAX_FILE_AGE = config.get("MAX_FILE_AGE")
VIRGIN_QTY_MULT = config.get("VIRGIN_QTY_MULT")
MINEXPOPTPRICE = config.get("MINEXPOPTPRICE")

# %%
# BUILD UNDS
# Get portfolio and open orders
with get_ib("SNP") as ib:
    qpf = ib_pf(ib)
    df_pf = classify_pf(qpf)

    df_openords = get_open_orders(ib, is_active=False)
    df_openords = classify_open_orders(df_openords, df_pf)

if is_market_open() or get_pickle(unds_path) is None:
    # Get unds. Make it fresh if stale.
    if do_i_refresh(unds_ct_path, max_days=MAX_FILE_AGE):
        unds = snp_qualified_und_contracts(unds_path=unds_ct_path, fresh=True)
    else:
        print(
            f"Reusing und contracts they are less than MAX_FILE_AGE:{MAX_FILE_AGE} days old"
        )
        unds = get_pickle(unds_ct_path)

    dfu = clean_ib_util_df(unds)

    # Update df_unds undPrice
    dfu["undPrice"] = dfu.merge(
        qpf[qpf.secType == "STK"][["symbol", "mktPrice"]], on="symbol", how="left"
    )["mktPrice"]

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

    # Merge price data
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
    df_unds = dfu.merge(
        df_pf[["symbol", "secType", "state"]], on=["symbol", "secType"], how="left"
    )
else:
    df_unds = get_pickle(unds_path)

df_unds.loc[df_unds.state.isna(), "state"] = "tbd"

# ..update status for symbols in df_unds but not in qpf
opt_symbols = df_pf[df_pf.secType == "OPT"].symbol
opt_state_dict = dict(
    zip(
        df_pf.loc[df_pf.secType == "OPT", "symbol"],
        df_pf.loc[df_pf.secType == "OPT", "state"],
    )
)
df_unds.loc[
    (df_unds.symbol.isin(opt_symbols)) & (df_unds.state == "tbd"),
    "state",
] = df_unds.loc[
    (df_unds.symbol.isin(opt_symbols)) & (df_unds.state == "tbd"),
    "symbol",
].map(opt_state_dict)

# ..update status for symbols not in qpf
df_unds.loc[~df_unds.symbol.isin(qpf.symbol), "state"] = "virgin"

df_unds = df_unds.drop(
    columns=["iv", "hv", "expiry", "strike", "right"], errors="ignore"
)

# ..apply the status update to df_unds and pickle
df_unds = update_unds_status(df_unds, df_pf, df_openords)

pickle_me(df_unds, unds_path)
pickle_me(df_pf, pf_path)
pickle_me(df_openords, oo_path)

# %%
#  GET CHAINS

if do_i_refresh(chains_path, max_days=MAX_FILE_AGE):
    chain_recreate = True
else:
    print(f"Reusing chains. They are less than MAX_FILE_AGE:{MAX_FILE_AGE} days old")
    chain_recreate = False

if chain_recreate:
    with get_ib("SNP", LIVE=True) as ib:
        chains = ib.run(df_chains(ib, unds, sleep_time=5.5, msg="raw chains"))
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

# %%
# MAKE COVERS FOR EXPOSED AND UNCOVERED STOCK POSITIONS

# Get exposed and uncovered long
uncov = df_unds.state.isin(["exposed", "uncovered"])
uncov_long = df_unds[uncov & (df_unds.position > 0)].reset_index(drop=True)

if uncov_long.empty:
    df_ccf = pd.DataFrame()
else:
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
        .drop(columns=["level_2", "diff"], errors="ignore")
    )

    # Make long covered call options
    cov_calls = [
        Option(s, e, k, "C", "SMART")
        for s, e, k in zip(cc_long.symbol, cc_long.expiry, cc_long.strike)
    ]

    with get_ib("SNP") as ib:
        ib.run(qualify_me(ib, cov_calls, desc="Qualifying covered calls"))

    df_cc1 = clean_ib_util_df([c for c in cov_calls if c.conId > 0])

    # Get the lower of the long covered call
    df_ccf = df_cc1.loc[df_cc1.groupby("symbol")["strike"].idxmin()]

    df_ccf = df_ccf.reset_index(drop=True)

    # Append undPrice and vy from df_unds
    df_ccf = df_ccf.merge(
        df_unds[["symbol", "undPrice", "vy"]], on="symbol", how="left"
    )

    # Integrate position and avgCost from df_pf into df_ccf
    df_ccf = df_ccf.merge(
        df_pf[["symbol", "position", "avgCost"]], on="symbol", how="left"
    )

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
                msg="covered call prices and vy",
            )
        )

    # Integrate dfx.price to df_ccf and determine action
    df_ccf = df_ccf.merge(dfx[["symbol", "price"]], on="symbol", how="left")


    df_ccf["margin"] = df_ccf.apply(
        lambda x: atm_margin(x.strike, x.undPrice, get_dte(x.expiry), x.vy), axis=1
    )

# Make covered puts for 'exposed' and 'uncovered' short positions

# Get exposed and uncovered short
uncov_short = df_unds.state.isin(["exposed", "uncovered"])
uncov_short = df_unds[uncov_short & (df_unds.position < 0)].reset_index(drop=True)

if uncov_short.empty:
    df_cpf = pd.DataFrame()
else:
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
        ib.run(qualify_me(ib, cov_puts, desc='Qualifying covered puts'))

    df_cp1 = clean_ib_util_df([p for p in cov_puts if p.conId > 0])

    # Get the higher of the short covered put
    df_cpf = df_cp1.loc[df_cp1.groupby("symbol")["strike"].idxmax()]

    df_cpf = df_cpf.reset_index(drop=True)

    # Append undPrice and vy from df_unds
    df_cpf = df_cpf.merge(
        df_unds[["symbol", "undPrice", "vy"]], on="symbol", how="left"
    )

    # Integrate position and avgCost from df_pf into df_cpf
    df_cpf = df_cpf.merge(
        df_pf[["symbol", "position", "avgCost"]], on="symbol", how="left"
    )

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
                msg="covered puts prices and vy",
            )
        )

    # Integrate dfx_cp.price to df_cpf and determine action
    df_cpf = df_cpf.merge(dfx_cp[["symbol", "price"]], on="symbol", how="left")

    df_cpf["margin"] = df_cpf.apply(
        lambda x: atm_margin(x.strike, x.undPrice, get_dte(x.expiry), x.vy), axis=1
    )

# Integrate df_ccf and df_cpf into df_cov
df_cov = pd.concat([df_ccf, df_cpf], ignore_index=True)

# delete df_cov.pkl
if cov_path.exists():
    cov_path.unlink()


if not df_cov.empty:
    # add 'dte' column with get_dte(expiry) as the 5th column
    df_cov.insert(4, "dte", df_cov.expiry.apply(get_dte))

    # Pickle df_cov
    pickle_me(df_cov, cov_path)

    # Analyze covered calls and puts
    cost = (df_cov.avgCost * df_cov.qty * 100).sum()
    premium = (df_cov.price * 100).sum()
    maxProfit = (
        np.where(
            df_cov.right == "C",
            (df_cov.strike - df_cov.undPrice) * 100,
            (df_cov.undPrice - df_cov.strike) * 100,
        ).sum()
        + premium
    )

    print(f"Cost: {cost:.2f}")
    print(f"Premium: {premium:.2f}")
    print(f"Max Profit: {maxProfit:.2f}")

# %%
# MAKE SOWING CONTRACTS FOR VIRGIN SYMBOLS

df_v = df_unds[df_unds.state == "virgin"].reset_index(drop=True)

# Get chains of df_unds with dtes nearest to VIRGIN_DTE
VIRGIN_DTE = config.get("VIRGIN_DTE")
VIRGIN_STD_MULT = config.get("VIRGIN_STD_MULT")

df_virg = chains.loc[
    chains[chains.symbol.isin(df_v.symbol.to_list())]
    .groupby(["symbol", "strike"])["dte"]
    .apply(lambda x: x.sub(VIRGIN_DTE).abs().idxmin())
]

df_virg = df_virg.merge(df_unds[["symbol", "undPrice", "vy"]], 
            on="symbol", how="left")

# Calculate standard deviation based on implied volatility and days to expiration
df_virg["sdev"] = df_virg.undPrice * df_virg.vy * (df_virg.dte / 365) ** 0.5

# For each symbol and expiry, get 3 strikes above undPrice + sdev
v_std = config.get("VIRGIN_STD_MULT", 3)  # Default to 3 if not specified
no_of_options = 3

# Sort df_virg.strike, with ascending = False,  grouped on symbol and expiry
df_virg = df_virg.sort_values(["symbol", "expiry", "strike"], ascending=[True, True, False])

# Get put shorts for virgin symbols
virg_short = (
    df_virg.groupby(["symbol", "expiry"])
    .apply(
        lambda x: x[x["strike"] < x["undPrice"] - v_std * x["sdev"]]
        .assign(diff=abs(x["strike"] - (x["undPrice"] - v_std * x["sdev"])))
        .sort_values("diff")
        .head(no_of_options),
        include_groups=False,
    )
    .reset_index()
    .drop(columns=["level_2", "diff"], errors="ignore")
)

# Make short virgin put options
virg_puts = [
    Option(s, e, k, "P", "SMART")
    for s, e, k in zip(virg_short.symbol, virg_short.expiry, virg_short.strike)
]

with get_ib("SNP") as ib:
    ib.run(qualify_me(ib, virg_puts, desc="Qualifying virgin puts"))

df_virg1 = clean_ib_util_df([p for p in virg_puts if p.conId > 0])

df_virg1["dte"] = df_virg1.expiry.apply(lambda x: get_dte(x))

# Get the lower strike of the short virgin put
nakeds = df_virg1.loc[df_virg1.groupby("symbol")["strike"].idxmax()]

nakeds = nakeds.reset_index(drop=True)

# Append undPrice and vy from df_unds
nakeds = nakeds.merge(
    df_unds[["symbol", "undPrice", "vy"]], on="symbol", how="left"
)

# Get prices and volatilities of nakeds
with get_ib("SNP") as ib:
    dfx_n = ib.run(
        df_iv(
            ib=ib,
            stocks=nakeds["contract"].tolist(),
            sleep_time=10,
            msg="naked put prices and vy",
        )
    )

# Integrate dfx_n to nakeds
df_nkd = nakeds.merge(
    dfx_n[["symbol", "price"]],
    on="symbol",
    how="left",
)

# Calculate atm_margin of df_nkd
df_nkd["margin"] = df_nkd.apply(
    lambda x: atm_margin(x.strike, x.undPrice, get_dte(x.expiry), x.vy), axis=1
)

# Get financials
with get_ib('SNP') as ib:
    fin = ib.run(get_financials(ib))

# Calculate qty of nakeds
VIRGIN_QTY_MULT = config.get("VIRGIN_QTY_MULT")
max_fund_per_symbol = VIRGIN_QTY_MULT * fin.get("nlv", 0)
df_nkd["qty"] = df_nkd.margin.apply(
    lambda x: max(1, int(max_fund_per_symbol / x)) if x > 0 else 1
)

if nkd_path.exists():
    nkd_path.unlink()

if not df_nkd.empty:
    pickle_me(df_nkd, nkd_path)


# Analyze naked puts
premium = (df_nkd.price * 100 * df_nkd.qty).sum()
print(f"Naked Premiums: {premium:.2f}")
