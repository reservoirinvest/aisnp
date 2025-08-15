import pandas as pd
from ibfuncs import get_ib, ib_pf, get_open_orders, get_financials, df_chains
from utils import (
    load_config, 
    get_pickle, 
    do_i_refresh, 
    ROOT, 
    classify_pf, 
    classify_open_orders, 
    update_unds_status,
    is_market_open,
    pickle_me,
    clean_ib_util_df
)
from snp import make_snp_unds
from loguru import logger

def build_data() -> dict:
    """
    Build and return portfolio data, open orders, underlying data, and option chains.
    
    Returns:
        dict: Dictionary containing:
            - 'df_pf': Portfolio DataFrame
            - 'df_openords': Open orders DataFrame
            - 'df_unds': Underlying data DataFrame
            - 'chains': Option chains DataFrame
    """
    # Load configuration
    config = load_config("SNP")
    MAX_FILE_AGE = config.get("MAX_FILE_AGE")
    
    # Paths
    chains_path = ROOT / "data" / "df_chains.pkl"
    unds_path = ROOT / "data" / "df_unds.pkl"
    
    # Get portfolio, open orders and financials
    with get_ib("SNP") as ib:
        qpf = ib_pf(ib)
        df_pf = classify_pf(qpf)
        openords = get_open_orders(ib)
        fin = ib.run(get_financials(ib))
    
    df_openords = classify_open_orders(openords, df_pf)
    
    # Get unds
    df_unds = get_pickle(unds_path)
    if df_unds is None or do_i_refresh(unds_path, max_days=MAX_FILE_AGE):
        df_unds = make_snp_unds()
    else:
        print(f"Reusing und contracts they are less than MAX_FILE_AGE:{MAX_FILE_AGE} days old")
    
    pickle_me(df_unds, unds_path)
    
    # Update status
    df_unds = update_unds_status(df_unds=df_unds, df_pf=df_pf, df_openords=df_openords).sort_values("symbol").reset_index(drop=True)
    df_pf = df_pf.merge(df_unds[["symbol", "undPrice"]], on="symbol", how="left")
    
    # Get chains
    chains = None
    if do_i_refresh(chains_path, max_days=MAX_FILE_AGE):
        chain_recreate = True
    else:
        print(f"Reusing chains. They are less than MAX_FILE_AGE:{MAX_FILE_AGE} days old")
        chain_recreate = False
    
    if chain_recreate:
        unds = df_unds.contract.to_list()
        
        with get_ib("SNP", LIVE=True) as ib:
            chains = ib.run(df_chains(ib, unds, sleep_time=5.5, msg="raw chains"))
        
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
    
    return {
        'df_pf': df_pf,
        'df_openords': df_openords,
        'df_unds': df_unds,
        'chains': chains,
        'fin': fin
    }
#%%
data = build_data()
