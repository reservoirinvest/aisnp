# This program identifies states and generates orders
# %%
# IMPORTS AND VARIABLES

from contextlib import ExitStack

import numpy as np
import pandas as pd
from ib_async import Option, util
from loguru import logger

from ibfuncs import (df_chains, df_iv, get_financials, get_ib, get_open_orders,
                     ib_pf, qualify_me)
from snp import make_snp_unds
from utils import (ROOT, atm_margin, classify_open_orders, classify_pf,
                   clean_ib_util_df, delete_files, delete_pkl_files, do_i_refresh, get_dte,
                   get_pickle, get_prec, is_market_open, load_config,
                   pickle_me, tqdm, update_unds_status, how_many_days_old)

# Get parameters
config = load_config("SNP")
COVER_MIN_DTE = config.get("COVER_MIN_DTE")
VIRGIN_DTE = config.get("VIRGIN_DTE")
MAX_FILE_AGE = config.get("MAX_FILE_AGE")
VIRGIN_QTY_MULT = config.get("VIRGIN_QTY_MULT")
MINEXPOPTPRICE = config.get("MINEXPOPTPRICE")
MINNAKEDOPTPRICE = config.get("MINNAKEDOPTPRICE")
PROTECT_DTE = config.get("PROTECT_DTE")
PROTECTION_STRIP = config.get("PROTECTION_STRIP")
REAPRATIO = config.get("REAPRATIO")

# Delete pickles that are old
for f in ['df_nkd.pkl', 'df_reap.pkl', 'df_protect.pkl']:
    f_path = ROOT / "data" / f
    if f_path.exists() and how_many_days_old(f_path) > MAX_FILE_AGE:
        delete_pkl_files([f_path])

log_file_path = ROOT / "log" / "states.log"
delete_files([log_file_path])

unds_path = ROOT / "data" / "df_unds.pkl"
chains_path = ROOT / "data" / "df_chains.pkl"

cov_path = ROOT / "data" / "df_cov.pkl"  # covered call and put path
nkd_path = ROOT / "data" / "df_nkd.pkl"

pf_path = ROOT / "data" / "df_pf.pkl"

util.logToFile(log_file_path, level=40) # IB ERRORs only logged.
logger.add(log_file_path, rotation="1 week")

df_unds = get_pickle(unds_path)
chains = get_pickle(chains_path)

df_cov = get_pickle(cov_path)
df_nkd = get_pickle(nkd_path)



# %%
# BUILD UNDS
# Get portfolio, open orders and financials
with get_ib("SNP") as ib:
    qpf = ib_pf(ib)
    df_pf = classify_pf(qpf)

    openords = get_open_orders(ib)

    fin = ib.run(get_financials(ib))

df_openords = classify_open_orders(openords, df_pf)

if is_market_open() or get_pickle(unds_path) is None:

    # Get unds. Make it fresh if stale.
    if do_i_refresh(unds_path, max_days=MAX_FILE_AGE):
        df_unds = make_snp_unds()
    else:
        print(
            f"Reusing und contracts they are less than MAX_FILE_AGE:{MAX_FILE_AGE} days old"
        )
        df_unds = get_pickle(unds_path)

        pickle_me(df_unds, unds_path)

# ..apply the status update to df_unds and pickle
df_unds = update_unds_status(df_unds=df_unds, df_pf=df_pf, df_openords=df_openords).sort_values("symbol").reset_index(drop=True)

# Update symbol in df_pf with undPrice in df_unds
df_pf = df_pf.merge(df_unds[["symbol", "undPrice"]], on="symbol", how="left")

# %%
#  GET CHAINS

if do_i_refresh(chains_path, max_days=MAX_FILE_AGE):
    chain_recreate = True
else:
    print(f"Reusing chains. They are less than MAX_FILE_AGE:{MAX_FILE_AGE} days old")
    chain_recreate = False

if chain_recreate:

    unds = df_unds.contract.to_list()

    # Check if IBG PAPER is online for comprehensive chains. TWS chains usually get incomplete.
    with ExitStack() as es:
        try:
            ib = es.enter_context(get_ib("SNP", LIVE=False))
        except Exception:
            logger.info("Failed to use LIVE=False (paper). Falling back to LIVE=True.")
            ib = get_ib("SNP", LIVE=True)
        finally:
            chains = ib.run(df_chains(ib, unds, sleep_time=5.5, msg="raw chains"))
            ib.disconnect()
    
        unds1 = clean_ib_util_df(unds)
        missing_unds = unds1[~unds1["symbol"].isin(chains["symbol"])]
        
        if not missing_unds.empty:
            with get_ib("SNP") as ib:
                additional_chains = ib.run(
                    df_chains(ib, missing_unds.contract.to_list(), msg="missing chains")
                    )
                if additional_chains is not None and not additional_chains.empty:
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
        cov_calls = ib.run(qualify_me(ib, cov_calls, desc="Qualifying covered calls"))

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

    df_cov["xPrice"] = df_cov.apply(
        lambda x: get_prec(max(x.price, MINEXPOPTPRICE / x.qty), 0.01)
        if x.qty != 0 else 0,
        axis=1,
    )

    # Pickle df_cov
    pickle_me(df_cov, cov_path)

    # Analyze covered calls and puts
    cost = (df_cov.avgCost * df_cov.qty * 100).sum()
    premium = (df_cov.xPrice * df_cov.qty * 100).sum()
    maxProfit = (
        np.where(
            df_cov.right == "C",
            (df_cov.strike - df_cov.undPrice) * df_cov.qty * 100,
            (df_cov.undPrice - df_cov.strike) * df_cov.qty * 100,
        ).sum()
        + premium
    )

    print(f"Position Cost: $ {cost:,.2f}")
    print(f"Cover Premium: $ {premium:,.2f}")
    print(f"Max Profit: $ {maxProfit:,.2f}")

else:
    print("No covers available!")

# %%
# MAKE SOWING CONTRACTS FOR VIRGIN AND ORPHANED SYMBOLS

df_v = df_unds[(df_unds.state == "virgin") | (df_unds.state == "orphaned")].reset_index(drop=True)

# Get chains of df_unds with dtes nearest to VIRGIN_DTE
VIRGIN_DTE = config.get("VIRGIN_DTE")
VIRGIN_PUT_STD_MULT = config.get("VIRGIN_PUT_STD_MULT")

df_virg = chains.loc[
    chains[chains.symbol.isin(df_v.symbol.to_list())]
    .groupby(["symbol", "strike"])["dte"]
    .apply(lambda x: x.sub(VIRGIN_DTE).abs().idxmin())
]

df_virg = df_virg.merge(df_unds[["symbol", "undPrice", "vy"]], 
            on="symbol", how="left")

# Calculate standard deviation based on implied volatility and days to expiration
df_virg["sdev"] = df_virg.undPrice * df_virg.vy * (df_virg.dte / 365) ** 0.5

# For each symbol and expiry, get strikes above undPrice + sdev
v_std = config.get("VIRGIN_PUT_STD_MULT", 3)  # Default to 3 if not specified
no_of_options = 4

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

# Check if IBG PAPER is online to qualify virigin puts.
with ExitStack() as es:
    try:
        ib = es.enter_context(get_ib("SNP", LIVE=False))
    except Exception:
        ib = get_ib("SNP", LIVE=True)
    finally:
        virg_puts = ib.run(qualify_me(ib, virg_puts, desc="Qualifying virgin puts"))
        ib.disconnect()

if not virg_puts:
    make_virg_puts = False
    virg_puts_dict = virg_short.groupby('symbol').agg({
    'expiry': 'first',
    'strike': lambda x: x.tolist()
}).apply(
    lambda x: {
        'expiry': str(x['expiry']),
        'strike': x['strike']
    }, axis=1
).to_dict()

    print(
        "\n".join(f"Virgin put for {k}: {v} is not available! " for k, v in virg_puts_dict.items())
    )
    df_nkd = pd.DataFrame()
else:
    make_virg_puts = True

if make_virg_puts:
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

    # Calculate qty of nakeds
    VIRGIN_QTY_MULT = config.get("VIRGIN_QTY_MULT")
    max_fund_per_symbol = VIRGIN_QTY_MULT * fin.get("nlv", 0)
    df_nkd["qty"] = df_nkd.margin.apply(
        lambda x: max(1, int(max_fund_per_symbol / x)) if x > 0 else 1
    )
    df_nkd['xPrice'] = df_nkd.apply(
        lambda x: get_prec(max(x.price, MINNAKEDOPTPRICE / x.qty), 0.01), axis=1)


if nkd_path.exists():
    nkd_path.unlink()

if not df_nkd.empty:

    pickle_me(df_nkd, nkd_path)

    # Analyze naked puts
    premium = (df_nkd.xPrice * 100 * df_nkd.qty).sum()
    print(f"Naked Premiums: $ {premium:,.2f}")
else:
    print("No naked puts available!")

# %%
# MAKE REAPS
# Extract unreaped contracts from df_unds
df_sowed = df_unds[df_unds.state == "unreaped"].reset_index(drop=True)

# Extract unreaped option contracts from df_pf
df_reap = df_pf[df_pf.symbol.isin(df_sowed.symbol) 
            & (df_pf.secType == "OPT")].reset_index(drop=True)

# Remove in-the-money options from df_reap
df_reap = df_reap[~((df_reap.right == 'C') & (df_reap.strike < df_reap.undPrice)) &
                      ~((df_reap.right == 'P') & (df_reap.undPrice < df_reap.strike))].reset_index(drop=True)


# Integrate Vy (volatility) into df_sowed_pf from df_unds
df_reap = df_reap.merge(
    df_unds[["symbol", "vy"]], on="symbol", how="left"
)

if df_reap is not None and not df_reap.empty:
    with get_ib("SNP") as ib:
        reaped = {}
        sow_cts = ib.run(qualify_me(ib, df_reap.contract.tolist(), desc="Qualifying reap unds"))
        df_reap = df_reap.assign(contract = sow_cts)

        for c, vy, undPrice in tqdm(zip(df_reap.contract, df_reap.vy, df_reap.undPrice),
                                    desc="reap opt price", total=len(sow_cts)):
            s = ib.calculateOptionPrice(c, vy, undPrice)
            reaped[c.conId] = s
    df_reap['optPrice'] = [s.optPrice if s else np.nan for s in df_reap.conId.map(reaped)]
    df_reap["xPrice"] = [get_prec(max(0.01,s),0.01) for s in df_reap['optPrice']]
    df_reap['xPrice'] = df_reap.apply(lambda x: min(x.xPrice, get_prec(abs(x.avgCost*REAPRATIO/100), 0.01)), axis=1)

    df_reap['qty'] = df_reap.position.abs().astype(int)
    reaps = (abs(df_reap.mktPrice - df_reap.xPrice)*df_reap.qty*100).sum()

    reap_path = ROOT/'data'/'df_reap.pkl'

    pickle_me(df_reap, reap_path)
    print(f'Have {len(df_reap)} reaping options unlocking US$ {reaps:,.0f}')
else:
    
    print("There are no reaping options")

# %%
# BUILD PROTECTION RECOMMENDATIONS
df_unprot = df_unds[df_unds.state.isin(['unprotected', 'exposed'])].reset_index(drop=True)

# Protect longs
df_ulong = df_unprot[df_unprot.position > 0]

make_long_protect = not df_ulong.empty

if make_long_protect:
    # Get chains nearest to desired dte
    df_uch = chains.loc[
        chains[chains.symbol.isin(df_ulong.symbol.to_list())]
        .groupby(["symbol", "strike"])["dte"]
        .apply(lambda x: x.sub(PROTECT_DTE).abs().idxmin())
    ]

    # get PROTECTION_STRIP contracts lower than undPrice
    df_ul = df_uch[df_uch.symbol.isin(df_ulong.symbol)]
    df_ul = df_ul.sort_values(["symbol", "expiry", "strike"], ascending=[True, True, False])
    df_ul = df_ul.merge(df_unds[["symbol", "undPrice"]], on="symbol", how="left")
    df_ul = df_ul.groupby("symbol").apply(
        lambda x: x[x.strike <= x["undPrice"].iloc[0]].head(PROTECTION_STRIP),
        include_groups=False
    ).reset_index().drop(columns="level_1", errors='ignore')

    df_ul['right'] = 'P'

    df_ul['contract'] = df_ul.apply(
        lambda x: Option(x.symbol, x.expiry, x.strike, x.right, 'SMART'),
        axis=1
    )

    with get_ib("SNP") as ib:
        ul1 = ib.run(qualify_me(ib, df_ul.contract, desc="Qualifying long protects"))
        df_iv_p = ib.run(df_iv(
            ib=ib,
            stocks=ul1,
            sleep_time=10,
            msg="long protect option prices and vy",
        ))

    # Long Protection recommendation suite
    df_ivp = df_iv_p.merge(
        df_unds[["symbol", "vy", "undPrice"]], on="symbol", how="left"
    )
    df_ivp = df_ivp.assign(vy=df_ivp["iv"].combine_first(df_ivp["vy"]))

    df_ivp = df_ivp.merge(df_pf[df_pf.secType == 'STK'][['symbol', 'position']], on='symbol')

    df_ivp['qty'] = (df_ivp.position.abs()/100).astype('int')
    df_ivp['dte'] = get_dte(df_ivp.expiry)
    df_ivp["protection"] = (df_ivp["undPrice"] - df_ivp["strike"])*100*df_ivp.qty

    # Median protection
    df_lprot = df_ivp.groupby('symbol').apply(lambda x: x.iloc[len(x)//2] if len(x) > 0 else x, include_groups=False).reset_index()

else:
    df_lprot = pd.DataFrame()

# Protect shorts
df_ushort = df_unprot[df_unprot.position < 0]

make_short_protect = not df_ushort.empty

if make_short_protect:
    # Get chains nearest to desired dte for short positions
    df_sch = chains.loc[
        chains[chains.symbol.isin(df_ushort.symbol.to_list())]
        .groupby(["symbol", "strike"])["dte"]
        .apply(lambda x: x.sub(PROTECT_DTE).abs().idxmin())
    ]

    # get PROTECTION_STRIPS  contracts higher than undPrice for protection
    df_us = df_sch[df_sch.symbol.isin(df_ushort.symbol)]
    df_us = df_us.sort_values(["symbol", "expiry", "strike"], ascending=[True, True, True])
    df_us = df_us.merge(df_unds[["symbol", "undPrice"]], on="symbol", how="left")
    df_us = df_us.groupby("symbol").apply(
        lambda x: x[x.strike >= x["undPrice"].iloc[0]].head(PROTECTION_STRIP),
        include_groups=False
    ).reset_index().drop(columns="level_1")

    df_us['right'] = 'C'

    df_us['contract'] = df_us.apply(
        lambda x: Option(x.symbol, x.expiry, x.strike, x.right, 'SMART'),
        axis=1
    )

    with get_ib("SNP") as ib:
        us1 = ib.run(qualify_me(ib, df_us.contract, desc="Qualifying short protects"))
        df_iv_s = ib.run(df_iv(
            ib=ib,
            stocks=us1,
            sleep_time=10,
            msg="short protect option prices and vy",
        ))

    # Short protection recommendation suite
    df_ivs = df_iv_s.merge(
        df_unds[["symbol", "vy", "undPrice"]], on="symbol", how="left"
    )
    df_ivs = df_ivs.assign(vy=df_ivs["iv"].combine_first(df_ivs["vy"]))

    df_ivs = df_ivs.merge(
        df_pf[df_pf.secType == "STK"][["symbol", "position"]], on="symbol", how="left"
    )

    df_ivs['qty'] = (df_ivs.position.abs()/100).astype('int')
    df_ivs['dte'] = get_dte(df_ivs.expiry)
    df_ivs["protection"] = (df_ivs["strike"] - df_ivs["undPrice"])*100*df_ivs.qty

    # Median protection for shorts
    df_sprot = df_ivs.groupby('symbol').apply(lambda x: x.iloc[len(x)//2] if len(x) > 0 else x, include_groups=False).reset_index()

else:
    df_sprot = pd.DataFrame()

# Combine protect and get xPrice, based on ib's calculations
df_protect = pd.concat([df_lprot, df_sprot], ignore_index=True)

if not df_lprot.empty or not df_sprot.empty:
    df_protect.loc[df_protect.vy.isna(), "vy"] = df_protect.loc[df_protect.vy.isna(), "symbol"].map(df_unds.set_index("symbol")["vy"])

    protect = {}
    with get_ib("SNP") as ib:
        for c, vy, undPrice in tqdm(zip(df_protect.contract, df_protect.vy, df_protect.undPrice),
                                    desc="protect opt price", total=len(df_protect)):
            s = ib.calculateOptionPrice(c, vy, undPrice)
            protect[c.conId] = s

    df_protect['optPrice'] = [s.optPrice if s else np.nan for s in df_protect.conId.map(protect)]
    df_protect['xPrice'] = [get_prec(max(0.01,s.optPrice),0.01) if s else 0.01 for s in df_protect.conId.map(protect)]

    df_protect = df_protect.assign(cost = df_protect.xPrice*df_protect.qty*100)
    df_protect = df_protect.assign(puc = df_protect.protection/df_protect.cost)
    df_protect.drop(columns=['iv', 'hv', 'position'], inplace=True, errors='ignore')

    print(f"Damage after ${df_protect.protection.sum():,.0f}")
    print(f"...protected for a cost of ${df_protect.cost.sum():,.0f} for dte: {df_protect.dte.mean():.1f} days")
    pickle_me(df_protect, ROOT/'data'/'df_protect.pkl')
else:
    print("All are protected. No protection needed.")
# %%
