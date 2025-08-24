# This program identifies states and generates orders

# %%
# IMPORTS AND VARIABLES

import time
import numpy as np
import pandas as pd
from ib_async import Option, util
from loguru import logger

from ibfuncs import (make_df_iv, get_ib, qualify_me, df_prices)
from utils import (ROOT, atm_margin, clean_ib_util_df, delete_pkl_files, get_dte,
                   get_pickle, get_prec, load_config,
                   pickle_me, tqdm, how_many_days_old, filter_closest_dates,
                   filter_closest_strikes)

from build_unds_chains import build_data

# Start timing the script execution
start_time = time.time()

# Get parameters
config = load_config("SNP")
COVER_ME = config.get("COVER_ME")
REAP_ME = config.get("REAP_ME")
PROTECT_ME = config.get("PROTECT_ME")
COVER_MIN_DTE = config.get("COVER_MIN_DTE")
VIRGIN_DTE = config.get("VIRGIN_DTE")
MAX_FILE_AGE = config.get("MAX_FILE_AGE")
VIRGIN_QTY_MULT = config.get("VIRGIN_QTY_MULT")
COVXPMULT = config.get("COVXPMULT")
MINNAKEDOPTPRICE = config.get("MINNAKEDOPTPRICE")
NAKEDXPMULT = config.get("NAKEDXPMULT")
PROTECT_DTE = config.get("PROTECT_DTE")
PROTECTION_STRIP = config.get("PROTECTION_STRIP")
REAPRATIO = config.get("REAPRATIO")
MINREAPDTE = config.get("MINREAPDTE")

unds_path = ROOT / "data" / "df_unds.pkl"
chains_path = ROOT / "data" / "df_chains.pkl"
cov_path = ROOT / "data" / "df_cov.pkl"  # covered call and put path
nkd_path = ROOT / "data" / "df_nkd.pkl"
pf_path = ROOT / "data" / "df_pf.pkl"
reap_path = ROOT / "data" / "df_reap.pkl"
purls_path = ROOT / 'data' / 'protect_rolls.pkl'

# Delete logs
log_file_path = ROOT / "log" / "states.log"
logger.add(log_file_path, rotation="2 days")
util.logToFile(log_file_path, level=40) # IB ERRORs only logged.

# Delete pickles that are old
for f in ['df_nkd.pkl', 'df_reap.pkl', 'df_protect.pkl']:
    f_path = ROOT / "data" / f
    if f_path.exists() and how_many_days_old(f_path) > MAX_FILE_AGE:
        delete_pkl_files([f_path])


data = build_data()

df_pf = data['df_pf']
df_openords = data['df_openords']
df_unds = data['df_unds']
chains = data['chains']
fin = data['fin']

df_cov = get_pickle(cov_path)
df_nkd = get_pickle(nkd_path)
df_reap = get_pickle(reap_path)

#%%
# BASE INTEGRITY CHECK
print('\n')
print('BASE INTEGRITY CHECK')
print('====================')

# Count unique symbols in each DataFrame
unique_unds_symbols = len(df_unds['symbol'].unique())
unique_chains_symbols = chains['symbol'].nunique()
unique_oo_symbols = df_openords['symbol'].nunique() if 'symbol' in df_openords.columns else None
unique_pf_symbols = df_pf['symbol'].nunique() if 'symbol' in df_pf.columns else None

# Display the counts
unique_symbols_count = {
    'No of und symbols': (unique_unds_symbols, f"{how_many_days_old(unds_path):.2f} days old."),
    'No of chain symbols': (unique_chains_symbols, f"{how_many_days_old(chains_path):.2f} days old."),
    'No of portfolio symbols': unique_pf_symbols,
    'No of open order symbols': unique_oo_symbols,
}

for k, v in unique_symbols_count.items():
    if v:
        print(f'{k}: {v}')

# Find symbols missing in unds, chains, and pf
missing_in_unds = chains[~chains['symbol'].isin(df_unds['symbol'])]
missing_in_chains = df_unds[~df_unds['symbol'].isin(chains['symbol'])]
missing_in_chains_from_pf = df_pf[~df_pf['symbol'].isin(chains['symbol'])]

# Display missing symbols as lists
print('\n')
print("Symbols missing in unds from chains:", missing_in_unds['symbol'].unique())
print("Symbols missing in chains from unds:", missing_in_chains['symbol'].unique())
print("Symbols missing in chains from pf:", missing_in_chains_from_pf['symbol'].unique())
print('\n')

# %%
# BUILD PROTECTION RECOMMENDATIONS
if not PROTECT_ME:
    print("PROTECT_ME in config is False. Protecting orders will not be placed.\n")

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
    df_ul = df_ul.groupby("symbol")[['symbol','expiry','strike','undPrice']].apply(
        lambda x: x[x.strike <= x["undPrice"].iloc[0]].head(PROTECTION_STRIP)
    ).drop(columns="level_1", errors='ignore')

    df_ul['right'] = 'P'

    df_ul['contract'] = df_ul.apply(
        lambda x: Option(x.symbol, x.expiry, x.strike, x.right, 'SMART'),
        axis=1
    )

    with get_ib("SNP") as ib:
        ul1 = ib.run(qualify_me(ib, df_ul.contract, desc="Qualifying long protects"))
        ul1 = [c for c in ul1 if c is not None]

        df_iv_p = ib.run(make_df_iv(
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

    # # Median protection
    # df_lprot = df_ivp.groupby('symbol')[df_ivp.columns.to_list()].apply(lambda x: x.iloc[len(x)//2] if len(x) > 0 else x)

    # Closest put protection
    df_lprot = df_ivp.groupby('symbol').apply(lambda x: x.iloc[x['protection'].argmin()] if len(x) > 0 else x, include_groups=False).reset_index()

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
        lambda x: x[x.strike >= x["undPrice"].iloc[0]].head(PROTECTION_STRIP)
    ).drop(columns="level_1", errors='ignore')

    df_us['right'] = 'C'

    df_us['contract'] = df_us.apply(
        lambda x: Option(x.symbol, x.expiry, x.strike, x.right, 'SMART'),
        axis=1
    )

    with get_ib("SNP") as ib:
        us1 = ib.run(qualify_me(ib, df_us.contract, desc="Qualifying short protects"))
        df_iv_s = ib.run(make_df_iv(
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
    df_sprot = df_ivs.groupby('symbol').apply(lambda x: x.iloc[len(x)//2] if len(x) > 0 else x)

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

    print(f"Damage after ${df_protect.protection.sum():,.0f}\n")
    print(f"...{len(df_protect.symbol.unique())} symbols can be protected for a cost of ${df_protect.cost.sum():,.0f} for dte: {df_protect.dte.mean():.1f} days\n")
    pickle_me(df_protect, ROOT/'data'/'df_protect.pkl')
else:
    print("All are protected. No protection needed.\n")

#%% 
# ROLLS FOR PROTECTING PUTS

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
rol_chains = chains[chains.symbol.isin(set(df_rol.symbol))]

# Append undPrice
rol_chains = rol_chains.set_index('symbol').join(df_unds.set_index('symbol')[['undPrice']]).reset_index()

# Filter for closest dates and strikes
df_cd = filter_closest_dates(rol_chains, PROTECT_DTE, num_dates=1)
p = filter_closest_strikes(df_cd, n=-4)

if not p.empty:
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
        print("\nWARNING: There are some put rolls whose strike-undPrice is larger than 5%. "
              "These will be taken out from auto-roll suggestion.")
        print(purls[purls['diff'] < -0.05][cols])

    purls1 = purls[purls['diff'] >= -0.05]

    # Append position/100 to purls1 and call it qty
    purls1 = purls1.copy()
    purls1['qty'] = purls1['symbol'].map(df_unds.set_index('symbol')['position'] / 100)

    purls1 = purls1.merge(df_pf[(df_pf.secType == 'OPT') & (df_pf.right == 'P') & (df_pf.position > 0)][['symbol', 'mktPrice']], on='symbol', how='left')
    purls1.rename(columns={'mktPrice': 'cost'}, inplace=True)

    rollover_cost = (purls1.price - purls1.cost) * purls1.qty * 100
    print(f"\nThe rollover cost of {purls1.symbol.unique().shape[0]} symbols for {purls1.expiry.apply(get_dte).max():.0f} days would be ${rollover_cost.sum():,.0f}."
          f"\nTotal protect cost can be north of ${rollover_cost.sum() + df_protect.cost.sum():,.0f}.")

    pickle_me(purls1, purls_path)

# %%
# MAKE COVERS FOR EXPOSED AND UNCOVERED STOCK POSITIONS
# Get exposed and uncovered long
if not COVER_ME:
    print("\nCOVER_ME configured to be False. Cover orders will not be placed\n")

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

    # Merge chains with underlying prices, volatilities and avgCost
    df_cc = df_cc.merge(df_unds[["symbol", "undPrice", "vy", "avgCost"]], on="symbol", how="left")

    # Calculate standard deviation based on implied volatility and days to expiration
    df_cc["sdev"] = df_cc.undPrice * df_cc.vy * (df_cc.dte / 365) ** 0.5
    
    # Calculate the minimum price for cover strikes 
    vol_based_price = df_cc.undPrice + config.get("COVER_STD_MULT") * df_cc.sdev
    df_cc["covPrice"] = np.maximum(df_cc.avgCost, vol_based_price)

    # For each symbol and expiry, get 3 strikes above covPrice
    no_of_options = 3

    cc_long = (
        df_cc.groupby(["symbol", "expiry"])[["symbol", "expiry", "strike", "undPrice", "sdev", "covPrice"]]
        .apply(
            lambda x: x[x["strike"] > x["covPrice"]]
            .assign(diff=x["strike"] - x["covPrice"])
            .sort_values("diff")
            .head(no_of_options)
        )
        .drop(columns=["level_2", "diff"], errors="ignore")
    )

    # Make long covered call options
    cov_calls = [
        Option(s, e, k, "C", "SMART")
        for s, e, k in zip(cc_long.symbol, cc_long.expiry, cc_long.strike)
    ]

    with get_ib("SNP") as ib:
        cov_calls = ib.run(qualify_me(ib, cov_calls, desc="Qualifying covered calls"))
        cov_calls = [c for c in cov_calls if c is not None]
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
        df_pf[df_pf.state.isin(["uncovered", "exposed"])][["symbol", "position", "avgCost"]], on="symbol", how="left"
    )

    # Make qty field as position/100
    df_ccf["action"] = "SELL"
    df_ccf["qty"] = df_ccf["position"] / 100
    df_ccf = df_ccf.drop(columns=["position"])

    # Get covered call prices, volatilities
    with get_ib("SNP") as ib:
        dfx = ib.run(
            make_df_iv(
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

    # Merge chains with underlying prices, volatilities and avgCost
    df_cp = df_cp.merge(df_unds[["symbol", "undPrice", "vy", "avgCost"]], on="symbol", how="left")

    # Calculate standard deviation based on implied volatility and days to expiration
    df_cp["sdev"] = df_cp.undPrice * df_cp.vy * (df_cp.dte / 365) ** 0.5
    
    # Calculate the maximum price for put covers (lower of avgCost or undPrice - c_std * sdev)
    vol_based_price = df_cp.undPrice - config.get("COVER_STD_MULT") * df_cp.sdev
    df_cp["covPrice"] = np.minimum(df_cp.avgCost, vol_based_price)

    # For each symbol and expiry, get 3 strikes below covPrice
    no_of_options = 3

    cp_short = (
        df_cp.groupby(["symbol", "expiry"])[["symbol", "expiry", "strike", "undPrice", "sdev", "covPrice"]]
        .apply(
            lambda x: x[x["strike"] < x["covPrice"]]
            .assign(diff=x["covPrice"] - x["strike"])
            .sort_values("diff")
            .head(no_of_options)
        )
        .drop(columns=["level_2", "diff"], errors="ignore")
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
        df_pf[df_pf.state.isin(["uncovered", "exposed"])][["symbol", "position", "avgCost"]], on="symbol", how="left"
    )

    # Make qty field as abs(position)/100
    df_cpf["action"] = "SELL"
    df_cpf["qty"] = abs(df_cpf["position"]) / 100
    df_cpf = df_cpf.drop(columns=["position"])

    # Get covered put prices, volatilities
    
    with get_ib("SNP") as ib:
        dfx_cp = ib.run(
            make_df_iv(
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

# correct the option expiry
if not df_cov.empty:
    _ = [setattr(option, 'lastTradeDateOrContractMonth', "20" + option.localSymbol[6:12]) 
        for option in df_cov.contract.to_list() if option.conId > 0]

# delete df_cov.pkl
if cov_path.exists():
    cov_path.unlink()

if not df_cov.empty:

    # add 'dte' column with get_dte(expiry) as the 5th column
    df_cov.insert(4, "dte", df_cov.expiry.apply(get_dte))

    df_cov["xPrice"] = df_cov.apply(
        lambda x: max(get_prec(x.price*COVXPMULT, 0.01), 0.05)
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
    print(f"Max Profit: $ {maxProfit:,.2f}\n")

else:
    print("No covers available!\n")


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
    df_virg.groupby(["symbol", "expiry"])[["symbol", "expiry", "strike", "undPrice", "sdev"]]
    .apply(
        lambda x: x[x["strike"] < x["undPrice"] - v_std * x["sdev"]]
        .assign(diff=abs(x["strike"] - (x["undPrice"] - v_std * x["sdev"])))
        .sort_values("diff")
        .head(no_of_options)
    )
    .drop(columns=["level_2", "diff"], errors="ignore")
)

# Make short virgin put options
virg_puts = [
    Option(s, e, k, "P", "SMART")
    for s, e, k in zip(virg_short.symbol, virg_short.expiry, virg_short.strike)
    if not pd.isna(k)
]

with get_ib("SNP") as ib:
    virg_puts = ib.run(qualify_me(ib, virg_puts, desc="Qualifying virgin puts"))

if not virg_puts:
    make_virg_puts = False

    print(f"Virgin put for {set(df_virg.symbol.to_list())} is not available! ")
    df_nkd = pd.DataFrame()
else:
    make_virg_puts = True

if make_virg_puts:
    virg_puts = [p for p in virg_puts if p is not None]

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
            make_df_iv(
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
    max_fund_per_symbol = VIRGIN_QTY_MULT * fin.get("net liquidation value", 0)
    df_nkd["qty"] = df_nkd.margin.apply(
        lambda x: max(1, int(max_fund_per_symbol / x)) if x > 0 else 1
    )
    df_nkd['xPrice'] = df_nkd.apply(
        lambda x: get_prec(max(x.price*NAKEDXPMULT, MINNAKEDOPTPRICE / x.qty), 0.01), axis=1)


if nkd_path.exists():
    nkd_path.unlink()

if not df_nkd.empty:

    pickle_me(df_nkd, nkd_path)

    # Analyze naked puts
    premium = (df_nkd.xPrice * 100 * df_nkd.qty).sum()
    print(f"Naked Premiums: $ {premium:,.2f}\n")
else:
    print("No naked puts available!\n")


# %%
# MAKE REAPS
if not REAP_ME:
    print("REAP_ME in config is False. Reaping orders will not be placed\n")

# Extract unreaped contracts from df_unds
df_sowed = df_unds[df_unds.state == "unreaped"].reset_index(drop=True)

# Extract unreaped option contracts from df_pf
df_reap = df_pf[df_pf.symbol.isin(df_sowed.symbol) 
            & (df_pf.secType == "OPT")].reset_index(drop=True)

# # Remove in-the-money options from df_reap
# df_reap = df_reap[~((df_reap.right == 'C') & (df_reap.strike < df_reap.undPrice)) &
#                       ~((df_reap.right == 'P') & (df_reap.undPrice < df_reap.strike))]

# Remove options that are on or below MINREAPDTE. This is to save last day transaction costs.
df_reap = df_reap[df_reap.expiry.apply(get_dte) > MINREAPDTE].reset_index(drop=True)


# Integrate Vy (volatility) into df_sowed_pf from df_unds
df_reap = df_reap.merge(
    df_unds[["symbol", "vy"]], on="symbol", how="left"
)

if df_reap is not None and not df_reap.empty:
    with get_ib("SNP") as ib:
        reaped = {}
        sow_cts = ib.run(qualify_me(ib, df_reap.contract.tolist(), desc="Qualifying reap unds"))
        df_reap = df_reap.assign(contract = sow_cts, expiry=[c.lastTradeDateOrContractMonth for c in sow_cts])

        for c, vy, undPrice in tqdm(zip(df_reap.contract, df_reap.vy, df_reap.undPrice),
                                    desc="reap opt price", total=len(sow_cts)):
            s = ib.calculateOptionPrice(c, vy, undPrice)
            reaped[c.conId] = s

    # correct the option expiry
    if not df_reap.empty:
        _ = [setattr(option, 'lastTradeDateOrContractMonth', "20" + option.localSymbol[6:12]) 
            for option in df_reap.contract.to_list() if option.conId > 0]

    df_reap['optPrice'] = [s.optPrice if s else np.nan for s in df_reap.conId.map(reaped)]
    df_reap["xPrice"] = [get_prec(max(0.01,s),0.01) for s in df_reap['optPrice']]
    df_reap['xPrice'] = df_reap.apply(lambda x: min(x.xPrice, get_prec(abs(x.avgCost*REAPRATIO/100), 0.01)), axis=1)
    df_reap['qty'] = df_reap.position.abs().astype(int)
    
    reaps = (abs(df_reap.mktPrice - df_reap.xPrice)*df_reap.qty*100).sum()

    reap_path = ROOT/'data'/'df_reap.pkl'
    pickle_me(df_reap, reap_path)
    print(f'Have {len(df_reap)} reaping options unlocking US$ {reaps:,.0f}\n')
else:
    print("There are no reaping options\n")

# %%
# EXTRACT ORPHANED CONTRACTS FROM df_pf
df_deorph = df_pf[(df_pf.state == "orphaned") & (df_pf.secType == "OPT")].copy()

if not df_deorph.empty:
   
    with get_ib("SNP") as ib:
        deorph_cts = ib.run(qualify_me(ib, df_deorph.contract.tolist(), desc="Qualifying orphaned unds"))
        df_deorph = df_deorph.assign(contract = deorph_cts)

    df_deorph["qty"] = df_deorph.position.abs().astype(int)
    df_deorph["xPrice"] = df_deorph["mktPrice"].apply(lambda x: max(0.09, get_prec(x, 0.1)))
    
    
    # Calculate total value of orphaned options
    deorph_total = (df_deorph.mktPrice * df_deorph.qty * 100).sum()

    deorph_path = ROOT / 'data' / 'df_deorph.pkl'
    pickle_me(df_deorph, deorph_path)
    print(f'Have {len(df_deorph)} orphaned options with total value US$ {deorph_total:,.0f}\n')
else:
    print("There are no orphaned options to process\n")

# %%
# PRINT TOTAL EXECUTION TIME
end_time = time.time()
execution_time = end_time - start_time
minutes = int(execution_time // 60)
seconds = int(execution_time % 60)
print(f"\n{'='*50}")
print(f"Total execution time: {minutes} minutes and {seconds} seconds")
print(f"{'='*50}")