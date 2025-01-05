from functools import lru_cache

import pandas as pd
from ib_async import Stock
from loguru import logger

from ibfuncs import get_ib, qualify_me
from utils import ROOT, clean_ib_util_df, get_pickle, pickle_me
from pathlib import Path


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
    
    pickle_me(output, unds_path)
    
    return output


