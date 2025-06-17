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
import pandas_market_calendars as mcal
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
            output = pickle.load(f)
            print(f"Loaded {path}")
            return output
    except FileNotFoundError:
        if print_msg:
            print(f"File not found: {path}")
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
        # udf["expiry"] = udf["expiry"].apply(lambda x: convert_to_utc_datetime(x, eod=eod, ist=ist))
        udf["expiry"] = udf["expiry"].apply(util.formatIBDatetime)
    else:
        udf["expiry"] = pd.NaT

    # Assign contracts to DataFrame
    udf["contract"] = ct

    return udf


def convert_to_utc_datetime(date_string, eod=False, ist=True):
    try:
        dt = parser.parse(date_string)
        date_obj = datetime.strptime(date_string, '%Y%m%d')
    except ValueError as e:
        logger.error(f"Invalid date string format {e}")
        return np.nan

    if eod:
        if ist:
            timezone = pytz.timezone("Asia/Kolkata")
            dt = dt.replace(hour=15, minute=30, second=0)
            dt = timezone.localize(dt)
        else:
            timezone = pytz.timezone('US/Eastern')
            dt = timezone.localize(date_obj.replace(hour=16, minute=0))
            # dt = est_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')

    return dt


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


def get_dte(date_input):
    """
    Calculate days to expiration from a date string or pandas Series of date strings.
    
    Args:
        date_input (str or pd.Series): Date string(s) in 'YYYYMMDD' format
    
    Returns:
        float or pd.Series: Number of days from option closing time to current time in UTC
    """
    # If input is a pandas Series, apply the function to each element
    if isinstance(date_input, pd.Series):
        return date_input.apply(get_dte)
    
    # Take first 8 characters if string is longer
    date_str = str(date_input)[:8]
    
    # Parse the date
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    
    # Create datetime object at option closing time (4 PM market close)
    expiry_datetime = datetime(year, month, day, 16, 0, 0, tzinfo=timezone.utc)
    
    # Get current time in UTC
    current_time = datetime.now(timezone.utc)
    
    # Calculate time difference and convert to days
    time_diff = expiry_datetime - current_time
    days_to_expiry = time_diff.total_seconds() / (24 * 3600)
    
    return days_to_expiry

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
    pf = pf.copy() # set the input
    pf["state"] = "tbd"

    states_classification = [
            # 1. Straddle Positions (Highest Priority)
            {
                "name": "straddled",
                "mask": pf.groupby(["symbol", "expiry", "strike"]).apply(
                    lambda x: (
                        len(x) == 2
                        and set(x["right"]) == {"C", "P"}
                        and np.sign(x["position"].iloc[0]) == np.sign(x["position"].iloc[1])
                    ),
                    include_groups=False
                ).reset_index()[0],
            },
            # 2. Option Covering Positions
            {
                "name": "covering",
                "mask": (
                    (pf.secType == "OPT") & 
                    ((pf.right == "C") & (pf.position < 0) | ((pf.right == "P") & (pf.position < 0)))
                ),
            },
            # 3. Option Protecting Positions
            {
                "name": "protecting",
                "mask": (
                    (pf.secType == "OPT") & 
                    ((pf.right == "C") & (pf.position > 0) | ((pf.right == "P") & (pf.position > 0)))
                ),
            },
            # 4. Orphaned Option Positions
            {
                "name": "orphaned",
                "mask": (
                    (pf.secType == "OPT") &
                    (pf.position > 0) &
                    (~pf.symbol.isin(pf[pf.secType == "STK"].symbol))
                ),
            },
            # 5. Sowed Option Positions
            {
                "name": "sowed",
                "mask": (
                    (pf.secType == "OPT") &
                    (pf.position < 0) &
                    (~pf.symbol.isin(pf[pf.secType == "STK"].symbol))
                ),
            },
            # 6. Stock Positions with Both Covering and Protecting
            {
                "name": "solid",
                "mask": (
                    (pf.secType == "STK") &
                    pf.symbol.isin(
                        pf[(pf.secType == "OPT") & (pf.state == "covering")].symbol
                    ) &
                    pf.symbol.isin(
                        pf[(pf.secType == "OPT") & (pf.state == "protecting")].symbol
                    )
                ),
            },
            # 7. Stock Positions with Covering but No Protecting
            {
                "name": "unprotected",
                "mask": (
                    (pf.secType == "STK") &
                    pf.symbol.isin(
                        pf[(pf.secType == "OPT") & (pf.state == "covering")].symbol
                    ) &
                    ~pf.symbol.isin(
                        pf[(pf.secType == "OPT") & (pf.state == "protecting")].symbol
                    )
                ),
            },
            # 8. Stock Positions with Protecting but No Covering
            {
                "name": "uncovered",
                "mask": (
                    (pf.secType == "STK") &
                    ~pf.symbol.isin(
                        pf[(pf.secType == "OPT") & (pf.state == "covering")].symbol
                    ) &
                    pf.symbol.isin(
                        pf[(pf.secType == "OPT") & (pf.state == "protecting")].symbol
                    )
                ),
            },
            # 9. Exposed Stock Positions (Lowest Priority)
            {
                "name": "exposed",
                "mask": (
                    (pf.secType == "STK") &
                    (pf.position > 0) &
                    ~pf.symbol.isin(
                        pf[(pf.secType == "OPT") & (pf.state == "covering")].symbol
                    ) &
                    ~pf.symbol.isin(
                        pf[(pf.secType == "OPT") & (pf.state == "protecting")].symbol
                    )
                ),
            },
        ]

    # Apply states in order of priority
    for state_config in states_classification:
        mask = state_config["mask"]
        pf.loc[mask, "state"] = state_config["name"]

    # Fallback for any remaining 'tbd' states
    pf.loc[pf.state == "tbd", "state"] = "unclassified"

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

    df.loc[covering_mask[covering_mask].index, "state"] = "covering"

    # 'protecting' - option BUY order with underlying stock position
    protecting_mask = (
        (opt_orders.action == "BUY")
        & (
            # Put option protecting long stock position
            ((opt_orders.right == "P") & (opt_orders.symbol.isin(pf[(pf.secType == "STK") & (pf.position > 0)].symbol)))
            |
            # Call option protecting short stock position
            ((opt_orders.right == "C") & (opt_orders.symbol.isin(pf[(pf.secType == "STK") & (pf.position < 0)].symbol)))
        )
    )
    df.loc[protecting_mask[protecting_mask].index, "state"] = "protecting"

    # 'sowing' - option SELL order without underlying stock position
    sowing_mask = (opt_orders.action == "SELL") & (
        ~opt_orders.symbol.isin(pf[(pf.secType == "STK")].symbol)
    )
    df.loc[sowing_mask[sowing_mask].index, "state"] = "sowing"

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
    df.loc[reaping_mask[reaping_mask].index, "state"] = "reaping"

    # 'de-orphaning' - option SELL order with matching existing option position
    de_orphaning_mask = opt_orders.apply(
        lambda row: (
            row.action == "SELL"
            and not pf[
                (pf.secType == "OPT")
                & (pf.symbol == row.symbol)
                & (pf.right == row.right)
                & (pf.strike == row.strike)
            ].empty
        ),
        axis=1,
    )
    df.loc[de_orphaning_mask[de_orphaning_mask].index, "state"] = "de-orphaning"

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
    df.loc[straddle_mask[straddle_mask].index, "state"] = "straddling"

    return df


def update_unds_status(df_unds:pd.DataFrame, 
                    df_pf:pd.DataFrame, 
                    df_openords: pd.DataFrame) -> pd.DataFrame:
    """
    Update underlying symbols status based on portfolio and open orders.

    Parameters:
    df_unds (pd.DataFrame): Underlying symbols DataFrame
    df_pf (pd.DataFrame): Portfolio DataFrame

    Returns:
    pd.DataFrame: Updated underlying symbols DataFrame with 'state' column
    """

    df_unds = df_unds.drop(columns=['mktPrice', 'state', ], errors='ignore').merge(
        df_pf[df_pf["secType"] == "STK"][["symbol", "mktPrice", 'state']],
        on="symbol",
        how="left",
        suffixes=("", "_new"),
    )

    # update status from df_pf for stock symbols
    stk_symbols = df_pf[df_pf.secType == "STK"].symbol
    stk_state_dict = dict(
        zip(
            df_pf.loc[df_pf.secType == "STK", "symbol"],
            df_pf.loc[df_pf.secType == "STK", "state"],
        )
    )

    df_unds.loc[df_unds.symbol.isin(stk_symbols), "state"] = \
            df_unds.loc[df_unds.symbol.isin(stk_symbols)].symbol.map(stk_state_dict)

    # ..update status for symbols not in df_pf
    df_unds.loc[~df_unds.symbol.isin(df_pf.symbol.unique()), "state"] = "virgin"

    # Zen conditions
    zen_symbols = set()

    # 1. Symbols with both covering and protecting positions are zen
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
    sowing_symbols = df_openords[df_openords.state == "sowing"].symbol
    zen_symbols.update(sowing_symbols)

    # 4. Unprotected with protecting order
    unprotected_with_protect = df_pf[
        (df_pf.state == "unprotected")
        & df_pf.symbol.isin(df_openords[df_openords.state == "protecting"].symbol)
    ].symbol
    zen_symbols.update(unprotected_with_protect)

    # 5. Uncovered with covering order
    uncovered_with_cover = df_pf[
        (df_pf.state == "uncovered")
        & df_pf.symbol.isin(df_openords[df_openords.state == "covering"].symbol)
    ].symbol
    zen_symbols.update(uncovered_with_cover)

    # 6. Long 'orphaned' position with 'de-orphaning' order
    orphaned_with_deorphan = df_pf[
        (df_pf.state == "orphaned")
        & df_pf.symbol.isin(df_openords[df_openords.state == "de-orphaning"].symbol)
    ].symbol
    zen_symbols.update(orphaned_with_deorphan)

    # 7. Short 'sowed' position with 'reaping' order
    sowed_with_reap = df_pf[
        (df_pf.state == "sowed")
        & df_pf.symbol.isin(df_openords[df_openords.state == "reaping"].symbol)
    ].symbol
    zen_symbols.update(sowed_with_reap)

    # 8. Short 'orphaned' position with 'virgin' order
    orphaned_with_virgin = df_pf[
        (df_pf.state == "orphaned")
        & ~df_pf.symbol.isin(df_openords[df_openords.state == "virgin"].symbol)
    ].symbol
    zen_symbols.update(orphaned_with_virgin)

    # Update status for zen symbols
    df_unds.loc[df_unds.symbol.isin(zen_symbols), "state"] = "zen"

    # Unreaped: Symbol has a short option position with no open 'reaping' order
    unreaped_symbols = df_pf[
        (df_pf.state == "sowed")
        & ~df_pf.symbol.isin(df_openords[df_openords.state == "reaping"].symbol)
    ].symbol

    # Update status for unreaped symbols
    df_unds.loc[df_unds.symbol.isin(unreaped_symbols), "state"] = "unreaped"

    # Unprotected: Symbol has an exposed state with only one 'covering' order
    unprotected_symbols = []
    for symbol in df_pf[df_pf.state == "unprotected"].symbol:
        openord_group = df_openords[df_openords.symbol == symbol]
        if len(openord_group) == 1 and openord_group.iloc[0].state == "covering":
            unprotected_symbols.append(symbol)

    # Update status for unprotected symbols
    df_unds.loc[df_unds.symbol.isin(unprotected_symbols), "state"] = "unprotected"

    # Uncovered: Symbol has an exposed state with only one 'protecting' order
    uncovered_symbols = []
    for symbol in df_unds[df_unds.state == "exposed"].symbol:
        openord_group = df_openords[df_openords.symbol == symbol]
        if len(openord_group) == 1 and openord_group.iloc[0].state == "protecting":
            uncovered_symbols.append(symbol)

    # Update status for uncovered symbols
    df_unds.loc[df_unds.symbol.isin(uncovered_symbols), "state"] = "uncovered"

    # Orphaned: Symbol has an 'orphaned' state with no open orders
    orphaned_symbols = df_pf[(df_pf.state == "orphaned") & ~df_pf.symbol.isin(df_openords.symbol)].symbol

    # Update status for orphaned symbols
    df_unds.loc[df_unds.symbol.isin(orphaned_symbols), "state"] = "orphaned"

    return df_unds

def atm_margin(strike, undPrice, dte, vy):
    """
    Calculates the margin for an at-the-money put sale.
    
    Parameters:
    strike (float): The strike price of the put option.
    undPrice (float): The underlying asset price.
    dte (int): The number of days to expiration.
    vy (float): The volatility of the underlying asset.
    
    Returns:
    float: The margin for the put sale.
    """
    
    # Calculate the time to expiration in years
    t = dte / 365
    
    # Calculate the delta of the put option
    d1 = (np.log(undPrice / strike) + (vy**2 / 2) * t) / (vy * np.sqrt(t))
    delta = -norm.cdf(d1)
    
    # Calculate the margin
    margin = strike * 100 * abs(delta)
    
    return margin

def is_market_open(date: Union[None, datetime]=datetime.now()) -> bool:
    """
    Checks if the US stock market (NYSE) is currently open.

    Parameters:
    date (str): The date to check. If None, the current date is used.

    Returns:
    bool: True if the market is open, False otherwise.
    """
    if date:
        result = mcal.get_calendar("NYSE").schedule(start_date=date, end_date=date)
    else:
        result = mcal.get_calendar("NYSE").schedule(start_date=datetime.now(), end_date=datetime.now())
    
    return result.empty is False

def delete_files(files_to_delete):
    """
    Delete specified files from the filesystem.
    
    Args:
        files_to_delete (list): List of Path objects to delete
    """
    for file_path in files_to_delete:
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"Deleted {file_path}")
            else:
                print(f"File {file_path} does not exist.")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def delete_pkl_files(files_to_delete, root=None):
    """
    Delete specified .pkl files from the data directory.
    
    Args:
        files_to_delete (list): List of filenames to delete
        root (Path, optional): Root directory. Defaults to project root/data.
    """
    from pathlib import Path
    
    # Use provided root or default to project root/data
    if root is None:
        root = ROOT / 'data'
    else:
        root = Path(root)
    
    for filename in files_to_delete:
        # Ensure filename ends with .pkl
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        file_path = root / filename
        delete_files([file_path])

def is_running_in_regular_terminal() -> bool:
    """
    This function determines if the code is running in a regular terminal.
    
    Returns:
        bool: True if running in a regular terminal, False otherwise.
    """
    # Check for VSCode
    if 'VSCODE_PID' in os.environ:
        return False
    # Check for PyCharm
    elif 'PYCHARM_HOSTED' in os.environ:
        return False
    # Check for generic VSCode
    elif 'TERM_PROGRAM' in os.environ and 'vscode' in os.environ['TERM_PROGRAM'].lower():
        return False
    # Check for Jupyter Notebook
    elif 'ipykernel' in sys.modules:
        return False
    # Check for Streamlit
    elif 'streamlit' in sys.argv:
        return False
    else:
        return True
