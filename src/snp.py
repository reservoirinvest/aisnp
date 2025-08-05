from functools import lru_cache
from pathlib import Path

import pandas as pd
from ib_async import Stock
from loguru import logger

from ibfuncs import (make_df_iv, get_ib, get_open_orders,
                     ib_pf, qualify_me)
from utils import (ROOT, classify_pf, clean_ib_util_df, get_pickle, pickle_me,
                   update_unds_status, classify_open_orders)


@lru_cache(maxsize=1)
def get_wiki_snps():
    try:
        snp_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        return pd.read_html(snp_url)[0]["Symbol"]
    except Exception as e:
        logger.error(f"Failed to retrieve S&P 500 symbols: {e}")
        return pd.Series()


@lru_cache(maxsize=1)
def read_weeklys():
    try:
        dls = "http://www.cboe.com/products/weeklys-options/available-weeklys"
        return pd.read_html(dls)[0].iloc[:, 1]
    except Exception as e:
        logger.error(f"Failed to retrieve weekly options: {e}")
        return pd.Series()


def weekly_snps():
    df = read_weeklys()
    df = df[df.isin(get_wiki_snps()) & df.str.isalpha()]
    
    df = pd.concat([df, pd.Series(["QQQ", "SPY"])], ignore_index=True)
    
    return df


def snp_qualified_und_contracts(unds_path: Path, fresh: bool=False) -> pd.Series:
    
    if not fresh:
        output = get_pickle(unds_path)
        if output is not None:
            return output

    unds = weekly_snps()
    
    with get_ib(MARKET='SNP') as ib:
        qualified_contracts = ib.run(
            qualify_me(ib, [Stock(s, exchange='SMART', currency='USD') for s in unds], desc='Qualifying SNP Unds')
        )

    output = pd.Series(qualified_contracts)
    
    # pickle_me(output, unds_path)
    
    return output


def make_snp_unds() -> pd.DataFrame:

    unds_path = ROOT / "data" / "df_unds.pkl"

    with get_ib(MARKET='SNP') as ib:
        qpf = ib_pf(ib)
        df_pf = classify_pf(qpf)
        openords = get_open_orders(ib)
        df_openords = classify_open_orders(openords, df_pf)

    # Get fresh underlying contracts
    unds = snp_qualified_und_contracts(unds_path=unds_path, fresh=True)
    dfu = clean_ib_util_df(unds)

    # Update df_unds undPrice
    dfu["undPrice"] = dfu.merge(
        qpf[qpf.secType == "STK"][["symbol", "mktPrice"]], 
        on="symbol", 
        how="left"
    )["mktPrice"]

    # Get und prices, volatilities
    with get_ib(MARKET='SNP') as ib:
        dfp = ib.run(
            make_df_iv(
                ib=ib,
                stocks=dfu["contract"].tolist(),
                sleep_time=15,
                msg="getting undPrices and vy",
            )
        )

    # Merge price data
    dfu.loc[dfu.undPrice.isnull(), "undPrice"] = dfu.merge(
        dfp[["symbol", "price"]], on="symbol", how="left"
    )["price"]

    # Merge volatility data
    dfu = dfu.merge(dfp[["symbol", "hv", "iv"]], on="symbol", how="left")
    dfu["vy"] = dfu["iv"].combine_first(dfu["hv"])

    # Merge portfolio data
    dfu = pd.concat(
        [
            dfu,
            dfu.merge(
                qpf[qpf.secType == "STK"][
                    ["symbol", "position", "mktPrice", "mktVal", "avgCost"]
                ],
                on="symbol",
                how="left",
            )[["position", "mktPrice", "mktVal", "avgCost"]],
        ],
        axis=1,
    )

    # Compute unPnL and rePnL grouped by symbol
    pnl_summary = qpf.groupby('symbol').agg({
        'unPnL': 'sum',
        'rePnL': 'sum'
    }).reset_index()

    # Merge PnL summary with dfu
    df_unds = dfu.merge(
        pnl_summary, 
        on="symbol", 
        how="left"
    ).merge(
        df_pf[["symbol", "secType", "state"]], 
        on=["symbol", "secType"], 
        how="left"
    )

    # Initialize states
    df_unds.loc[df_unds.state.isna(), "state"] = "tbd"
    
    # Update states for options
    opt_symbols = df_pf[df_pf.secType == "OPT"].symbol
    opt_state_dict = dict(zip(
        df_pf.loc[df_pf.secType == "OPT", "symbol"],
        df_pf.loc[df_pf.secType == "OPT", "state"],
    ))
    
    df_unds.loc[
        (df_unds.symbol.isin(opt_symbols)) & (df_unds.state == "tbd"),
        "state",
    ] = df_unds.loc[
        (df_unds.symbol.isin(opt_symbols)) & (df_unds.state == "tbd"),
        "symbol",
    ].map(opt_state_dict)

    # Update virgin states
    df_unds.loc[~df_unds.symbol.isin(qpf.symbol), "state"] = "virgin"

    # Clean up columns
    df_unds = df_unds.drop(
        columns=["iv", "hv", "expiry", "strike", "right"], 
        errors="ignore"
    )

    # Final status update
    df_unds = update_unds_status(
        df_unds=df_unds, 
        df_pf=df_pf, 
        df_openords=df_openords
    )

    no_undPrice = df_unds[df_unds.undPrice.isna()]
    no_vy = df_unds[df_unds.vy.isna()]
    if not no_undPrice.empty:
        print(f"{len(no_undPrice)} symbols have no undPrice, sample: {no_undPrice.symbol.head().to_list()}")
    if not no_vy.empty:
        print(f"{len(no_vy)} symbols have no vy, sample: {no_vy.symbol.head().to_list()}")

    pickle_me(df_unds, unds_path)

    return df_unds

# %%
if __name__ == "__main__":
    df_unds = make_snp_unds()