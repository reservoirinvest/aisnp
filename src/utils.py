import asyncio
import math
import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pytz
import yaml
from dateutil import parser
from dotenv import find_dotenv, load_dotenv
from from_root import from_root
from ib_async import util
from loguru import logger
from scipy.stats import norm
from tqdm import tqdm

ROOT = from_root()


class Timediff:
    def __init__(
        self, td: timedelta, days: int, hours: int, minutes: int, seconds: float
    ):
        self.td = td
        self.days = days
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds


def load_config(market: str):
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path=dotenv_path)

    config_path = ROOT / "config" / f"{market.lower()}_config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    for key, value in os.environ.items():
        if key in config:
            config[key] = value

    return config


def configure_logging(module_name: Optional[str] = None, log_level: str = "INFO") -> None:
    """
    Configure loguru logging with module-specific log files.

    Args:
        module_name (Optional[str]): Name of the module for log file naming.
                                     If None, uses the caller's module name.
        log_level (str): Logging level (e.g., 'INFO', 'DEBUG', 'ERROR'). Defaults to 'INFO'.
    """

    # Configure log file path
    log_file = ROOT / "log" / f"{module_name or 'default'}.log"

    # Remove default logger
    logger.remove()

    # Configure console logging
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Configure file logging with rotation
    logger.add(
        log_file,
        level=log_level,
        rotation="10 MB",  # Rotate log files when they reach 10 MB
        retention="10 days",  # Keep log files for 10 days
        compression="zip",  # Compress old log files
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )


def do_in_chunks(
    func, func_params: dict, chunk_size: int = 44, msg: Optional[str] = None
) -> dict:
    """Process payload in chunks using the provided function and its parameters.   
   
    Notes: 
    1. While constructing calling function, payload should be the first agrument of func. 
    Active IB could be included in func_params, if needed.
    
    ...e.g. async def wifs(co:list, ib:IB, ...
    
    2. the output of the core function should be a dictionary
    
    ...e.g. async def wifAsync(..) -> dict: ... """


    if "payload" not in func_params:
        raise ValueError("Missing 'payload' in func_params.")

    items = func_params.pop("payload")  # Extract items from func_params
    all_results = {}

    # Use func.__name__ as default message if msg is None
    msg = msg or func.__name__

    # Initialize tqdm progress bar
    with tqdm(total=len(items), desc=msg, unit="chunk") as pbar:
        for i in range(0, len(items), chunk_size):
            chunk = items[i : i + chunk_size]
            results = asyncio.run(
                func(chunk, **func_params)
            )  # Call the function and collect results
            
            all_results.update(results)  # Combine results from each chunk

            # Update progress bar
            pbar.update(len(chunk))

    return all_results


def pickle_me(obj, file_path: Path):
    with open(str(file_path), "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_pickle(path: Path, print_msg: bool = True):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        if print_msg:
            logger.error(f"File not found: {path}")
        return None


def do_i_refresh(unds_path: Path, max_days: float) -> bool:
    """
    Decides whether to refresh the unds data or not based on how many days old it is.
    """
    days_old = how_many_days_old(unds_path)

    return days_old is None or days_old > max_days


def get_prec(v: float, base: float) -> float:
    try:
        output = round(round((v) / base) * base, -int(math.floor(math.log10(base))))
    except Exception:
        output = None

    return output


def to_list(data):
    if isinstance(data, list):
        return list(flatten(data))

    try:
        return list(data)
    except TypeError:
        return [data]


def flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def clean_ib_util_df(
    contracts: Union[list, pd.Series],
    eod=True,
    ist=False,
) -> Union[pd.DataFrame, None]:
    """Cleans ib_async's util.df to keep only relevant columns"""

    # Ensure contracts is a list
    if isinstance(contracts, pd.Series):
        ct = contracts.to_list()
    elif not isinstance(contracts, list):
        logger.error(
            f"Invalid type for contracts: {type(contracts)}. Must be list or pd.Series."
        )
        return None
    else:
        ct = contracts

    # Try to create DataFrame from contracts
    try:
        udf = util.df(ct)
    except (AttributeError, ValueError) as e:
        logger.error(f"Error creating DataFrame from contracts: {e}")
        return None

    # Check if DataFrame is empty
    if udf.empty:
        return None

    # Select and rename columns
    udf = udf[
        [
            "symbol",
            "conId",
            "secType",
            "lastTradeDateOrContractMonth",
            "strike",
            "right",
        ]
    ]
    udf.rename(columns={"lastTradeDateOrContractMonth": "expiry"}, inplace=True)

    # Convert expiry to UTC datetime, if it exists
    if len(udf.expiry.iloc[0]) != 0:
        udf["expiry"] = udf["expiry"].apply(
            lambda x: convert_to_utc_datetime(x, eod=eod, ist=ist)
        )
    else:
        udf["expiry"] = pd.NaT

    # Assign contracts to DataFrame
    udf["contract"] = ct

    return udf


def convert_to_utc_datetime(date_string, eod=False, ist=True):
    try:
        dt = parser.parse(date_string)
    except ValueError as e:
        logger.error(f"Invalid date string format {e}")
        return np.nan

    if eod:
        if ist:
            timezone = pytz.timezone("Asia/Kolkata")
            dt = dt.replace(hour=15, minute=30, second=0)
        else:
            timezone = pytz.timezone("America/New_York")
            dt = dt.replace(hour=16, minute=0, second=0)

        dt = timezone.localize(dt)

    return dt.astimezone(pytz.UTC)


def yes_or_no(question: str, default="n") -> bool:
    while True:
        answer = input(question + " (y/n): ").lower().strip()
        if not answer:
            return default == "y"
        if answer in ("y", "yes"):
            return True
        elif answer in ("n", "no"):
            return False
        else:
            print("Please answer yes or no.")


def get_file_age(file_path: Path) -> Optional[Timediff]:
    if not file_path.exists():
        logger.info(f"{file_path} file is not found")
        return None

    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
    time_now = datetime.now()
    td = time_now - file_time

    return split_time_difference(td)


def split_time_difference(diff: timedelta) -> Timediff:
    days = diff.days
    hours, remainder = divmod(diff.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    seconds += diff.microseconds / 1e6

    return Timediff(diff, days, hours, minutes, seconds)


def how_many_days_old(file_path: Path) -> float:
    file_age = get_file_age(file_path=file_path)

    seconds_in_a_day = 86400
    file_age_in_days = (
        file_age.td.total_seconds() / seconds_in_a_day if file_age else None
    )

    return file_age_in_days


def get_dte(
    s: Union[pd.Series, datetime], exchange: Optional[str] = None
) -> Union[pd.Series, float]:
    now_utc = datetime.now(timezone.utc)

    if isinstance(s, pd.Series):
        try:
            if isinstance(s.iloc[0], str):
                if exchange == "NSE":
                    s = (
                        pd.to_datetime(s)
                        .dt.tz_localize("Asia/Kolkata")
                        .apply(lambda x: x.replace(hour=15, minute=30, second=0))
                    )
                else:
                    s = (
                        pd.to_datetime(s)
                        .dt.tz_localize("US/Eastern")
                        .apply(lambda x: x.replace(hour=16, minute=0, second=0))
                    )
            return (s - now_utc).dt.total_seconds() / (24 * 60 * 60)
        except (TypeError, ValueError):
            return pd.Series([np.nan] * len(s))
    elif isinstance(s, datetime):
        return (s - now_utc).total_seconds() / (24 * 60 * 60)
    else:
        raise TypeError("Input must be a pandas Series or a datetime.datetime object")


def us_repo_rate():
    """Risk free US interest rate

    Returns:
        _type_: float (5.51)

    """
    tbill_yield = web.DataReader(
        "DGS1MO", "fred", start=datetime.now() - timedelta(days=365), end=datetime.now()
    )["DGS1MO"].iloc[-1]
    return tbill_yield


def black_scholes(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """
    Calculate the Black-Scholes option price.

    Parameters:
    -----------
    S : float
        Current underlying price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Implied volatility (annualized)
    option_type : str
        Option type ('C' for Call, 'P' for Put)

    Returns:
    --------
    float
        Option price according to Black-Scholes formula
    """

    # Input validation
    if not isinstance(option_type, str) or option_type not in ["C", "P"]:
        raise ValueError("option_type must be either 'C' for Call or 'P' for Put")
    if any(x <= 0 for x in [S, K, T, sigma]):
        raise ValueError("S, K, T, and sigma must be positive")

    # Handle edge case of very small time to expiration
    if T < 1e-10:
        if option_type == "C":
            return max(0, S - K)
        else:
            return max(0, K - S)

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate Call or Put price
    if option_type == "C":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put option
        price = (
            S * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
            + K * np.exp(-r * T)
            - S
        )

    return price


def classify_pf(pf):
    """
    Classifies trading strategies in a portfolio based on option and stock positions.

    Parameters:
    pf (pd.DataFrame): Portfolio DataFrame containing columns:
        - symbol: Ticker symbol
        - secType: Security type ('STK' or 'OPT')
        - right: Option right ('C', 'P', or '0' for stocks)
        - expiry: Option expiration date
        - strike: Option strike price
        - position: Position size (positive or negative)

    Returns:
    pd.DataFrame: Original DataFrame with added 'state' column containing classifications
    """
    # Create a copy to avoid modifying the original DataFrame
    # pf = pf.copy()

    # Sort by covered with protection pairs
    right_order = {"C": 0, "0": 1, "P": 2}

    pf = pf.sort_values(
        by=["symbol", "right"],
        key=lambda x: x.map(right_order) if x.name == "right" else x,
    )

    # Initialize status field with blank underscore
    pf["state"] = "tbd"

    # Filter for options only
    opt_pf = pf[pf.secType == "OPT"]

    # Group by symbol and expiry to find matching calls and puts
    straddled = opt_pf.groupby(["symbol", "expiry", "strike"]).filter(
        lambda x: (
            # Must have exactly 2 rows (call and put)
            len(x) == 2
            and
            # Must have both C and P
            set(x["right"]) == {"C", "P"}
            and
            # Position signs must match
            np.sign(x["position"].iloc[0]) == np.sign(x["position"].iloc[1])
        )
    )

    # Update status field for straddles
    pf.loc[pf.index.isin(straddled.index), "state"] = "straddled"

    # Filter for stocks and their associated options
    solid = pf.groupby("symbol").filter(
        lambda x: (
            # Must have exactly one STK row
            (x.secType == "STK").sum() == 1
            and
            # Must have 1 or 2 OPT rows
            (x.secType == "OPT").sum() in [1, 2]
        )
    )

    # Update status field for covered calls/puts
    pf.loc[solid.index, "state"] = solid.apply(
        lambda x: (
            "covering"
            if (x["right"] == "C" and x["position"] < 0)
            or (x["right"] == "P" and x["position"] < 0)
            else "protecting"
        ),
        axis=1,
    )

    # Update status field for stocks with both covering and protecting
    stocks_with_both = pf[
        (pf.secType == "STK")
        & pf.symbol.isin(pf[(pf.state == "covering")].symbol)
        & pf.symbol.isin(pf[(pf.state == "protecting")].symbol)
    ]
    pf.loc[stocks_with_both.index, "state"] = "solid"

    # Update status field for stocks with covering but no protecting
    stocks_covered_only = pf[
        (pf.secType == "STK")
        & pf.symbol.isin(pf[(pf.state == "covering")].symbol)
        & ~pf.symbol.isin(pf[(pf.state == "protecting")].symbol)
    ]
    pf.loc[stocks_covered_only.index, "state"] = "unprotected"

    # Update status field for stocks with protecting but no covering
    stocks_protected_only = pf[
        (pf.secType == "STK")
        & ~pf.symbol.isin(pf[(pf.state == "covering")].symbol)
        & pf.symbol.isin(pf[(pf.state == "protecting")].symbol)
    ]
    pf.loc[stocks_protected_only.index, "state"] = "uncovered"

    # Update status field for orphaned options
    pf.loc[(pf.state == "tbd") & (pf.secType == "OPT") & (pf.position > 0), "state"] = (
        "orphaned"
    )

    # Update status field for sowed options
    pf.loc[(pf.state == "tbd") & (pf.secType == "OPT") & (pf.position < 0), "state"] = (
        "sowed"
    )

    # Update status field for exposed stock positions
    pf.loc[(pf.state == "tbd") & (pf.secType == "STK"), "state"] = "exposed"

    return pf


def classify_open_orders(df_openords, pf):
    """
    Classify open orders based on their characteristics and portfolio context.

    Parameters:
    df_openords (pd.DataFrame): DataFrame of open orders
    pf (pd.DataFrame): Portfolio DataFrame

    Returns:
    pd.DataFrame: Open orders DataFrame with added 'state' column
    """
    if df_openords is None or df_openords.empty:
        return df_openords

    # Create a copy to avoid modifying the original DataFrame
    df = df_openords.copy()

    # Initialize status column
    df["state"] = "unclassified"

    # Identify option orders
    opt_orders = df[df.secType == "OPT"]

    # 'covering' - option SELL order with underlying stock position
    covering_mask = (opt_orders.action == "SELL") & (
        # Call option with positive stock position
        (
            (opt_orders.right == "C")
            & (
                opt_orders.symbol.isin(
                    pf[(pf.secType == "STK") & (pf.position > 0)].symbol
                )
            )
        )
        |
        # Put option with negative stock position
        (
            (opt_orders.right == "P")
            & (
                opt_orders.symbol.isin(
                    pf[(pf.secType == "STK") & (pf.position < 0)].symbol
                )
            )
        )
    )

    df.loc[covering_mask, "state"] = "covering"

    # 'protecting' - option BUY order with underlying stock position
    protecting_mask = (
        (opt_orders.action == "BUY")
        & (opt_orders.right == "P")
        & (opt_orders.symbol.isin(pf[(pf.secType == "STK") & (pf.position > 0)].symbol))
    )
    df.loc[protecting_mask, "state"] = "protecting"

    # 'sowing' - option SELL order without underlying stock position
    sowing_mask = (opt_orders.action == "SELL") & (
        ~opt_orders.symbol.isin(pf[(pf.secType == "STK")].symbol)
    )
    df.loc[sowing_mask, "state"] = "sowing"

    # 'reaping' - option BUY order with matching existing option position
    reaping_mask = opt_orders.apply(
        lambda row: (
            row.action == "BUY"
            and not pf[
                (pf.secType == "OPT")
                & (pf.symbol == row.symbol)
                & (pf.right == row.right)
                & (pf.strike == row.strike)
            ].empty
        ),
        axis=1,
    )
    df.loc[reaping_mask, "state"] = "reaping"

    # 'straddling' - two option BUY orders for same symbol not in portfolio
    # Group by symbol and count BUY actions
    straddle_symbols = (
        opt_orders[(opt_orders.action == "BUY")]
        .groupby("symbol")
        .filter(lambda x: len(x) >= 2)["symbol"]
        .unique()
    )

    straddle_mask = (
        (opt_orders.action == "BUY")
        & (opt_orders.symbol.isin(straddle_symbols))
        & (~opt_orders.symbol.isin(pf.symbol))
    )
    df.loc[straddle_mask, "state"] = "straddling"

    # 'de-orphaning' - option SELL order with only an underlying option position and no stock position
    deorphaning_mask = opt_orders.apply(
        lambda row: (
            row.action == "SELL"
            and
            # No stock position for the symbol
            row.symbol not in pf[(pf.secType == "STK")].symbol
            and
            # Has an option position for the symbol
            not pf[(pf.secType == "OPT") & (pf.symbol == row.symbol)].empty
        ),
        axis=1,
    )
    df.loc[deorphaning_mask, "state"] = "de-orphaning"

    return df


def update_unds_status(df_unds, df_pf, df_openords):
    """
    Update underlying symbols status based on portfolio and open orders.

    Parameters:
    df_unds (pd.DataFrame): Underlying symbols DataFrame
    df_pf (pd.DataFrame): Portfolio DataFrame
    df_openords (pd.DataFrame): Open orders DataFrame

    Returns:
    pd.DataFrame: Updated underlying symbols DataFrame with 'state' column
    """
    # Initialize status column if not exists
    if "state" not in df_unds.columns:
        df_unds["state"] = df_pf.set_index("symbol")["state"].reindex(
            df_unds.symbol, fill_value="unknown"
        )
    if df_openords is None or df_openords.empty:
        return df_unds

    # Zen conditions
    zen_symbols = set()

    # 1. Symbols with both covering and protecting positions
    for symbol, group in df_openords.groupby("symbol"):
        if len(group) == 2 and {"covering", "protecting"}.issubset(set(group.state)):
            zen_symbols.add(symbol)
        else:
            group = df_pf[df_pf.symbol == symbol]
            if len(group) == 2 and {"covering", "protecting"}.issubset(
                set(group.state)
            ):
                zen_symbols.add(symbol)

    # 2. Symbols with 'straddled' portfolio state
    straddled_symbols = df_pf[df_pf.state == "straddled"].symbol
    zen_symbols.update(straddled_symbols)

    # 3. Symbols with short 'sowing' order
    sowing_symbols = df_openords[df_openords.status == "sowing"].symbol
    zen_symbols.update(sowing_symbols)

    # 4. Unprotected with protecting order
    unprotected_with_protect = df_pf[
        (df_pf.state == "unprotected")
        & df_pf.symbol.isin(df_openords[df_openords.status == "protecting"].symbol)
    ].symbol
    zen_symbols.update(unprotected_with_protect)

    # 5. Uncovered with covering order
    uncovered_with_cover = df_pf[
        (df_pf.state == "uncovered")
        & df_pf.symbol.isin(df_openords[df_openords.status == "covering"].symbol)
    ].symbol
    zen_symbols.update(uncovered_with_cover)

    # 6. Long 'orphaned' position with 'de-orphaning' order
    orphaned_with_deorphan = df_pf[
        (df_pf.state == "orphaned")
        & df_pf.symbol.isin(df_openords[df_openords.status == "de-orphaning"].symbol)
    ].symbol
    zen_symbols.update(orphaned_with_deorphan)

    # 7. Short 'sowed' position with 'reaping' order
    sowed_with_reap = df_pf[
        (df_pf.state == "sowed")
        & df_pf.symbol.isin(df_openords[df_openords.status == "reaping"].symbol)
    ].symbol
    zen_symbols.update(sowed_with_reap)

    # Update status for zen symbols
    df_unds.loc[df_unds.symbol.isin(zen_symbols), "state"] = "zen"

    # Unreaped: Symbol has a short option position with no open 'reaping' order
    unreaped_symbols = df_pf[
        (df_pf.state == "sowed")
        & ~df_pf.symbol.isin(df_openords[df_openords.status == "reaping"].symbol)
    ].symbol

    # Update status for unreaped symbols
    df_unds.loc[df_unds.symbol.isin(unreaped_symbols), "state"] = "unreaped"

    return df_unds
