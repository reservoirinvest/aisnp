from utils import load_config, get_pickle, ROOT
from ib_async import Option

# Load configuration
config = load_config("SNP")

# Load data from pickle files
chains_path = ROOT / "data" / "df_chains.pkl"
unds_path = ROOT / "data" / "df_unds.pkl"

chains = get_pickle(chains_path)
df_unds = get_pickle(unds_path)

# Get virgin and orphaned symbols
df_v = df_unds[df_unds.state.isin(['virgin', 'orphaned'])].reset_index(drop=True)

# Filter chains for virgin and orphaned symbols
df_virg_calls = chains.loc[chains.symbol.isin(df_v.symbol.to_list())]

# Merge with underlying prices and volatilities
df_virg_calls = df_virg_calls.merge(
    df_unds[['symbol', 'undPrice', 'vy']], 
    on='symbol', 
    how='left'
)

# Calculate standard deviation based on implied volatility and days to expiration
df_virg_calls['sdev'] = df_virg_calls.undPrice * df_virg_calls.vy * (df_virg_calls.dte / 365) ** 0.5

# Get VIRGIN_CALL_STD_MULT from config
v_std = config.get("VIRGIN_CALL_STD_MULT", 3)  # Default to 3 if not specified

# Get call shorts for virgin and orphaned symbols
virg_short_calls = (
    df_virg_calls
    .assign(diff=lambda x: abs(x['strike'] - (x['undPrice'] + v_std * x['sdev'])))
    .query('strike > undPrice + @v_std * sdev')
    .sort_values(['symbol', 'expiry', 'diff'])
    .groupby('symbol', group_keys=False)
    .head(1)  # Take the first row for each symbol
)

# Print results for inspection
print("Virgin and Orphaned Symbols:\n", df_v)
print("\nVirgin Call Options:\n", df_virg_calls)
print("\nShort Virgin Calls:\n", virg_short_calls)

# Create Option contracts for the selected strikes
virg_call_contracts = [
    Option(s, e, k, "C", "SMART")
    for s, e, k in zip(virg_short_calls.symbol, virg_short_calls.expiry, virg_short_calls.strike)
]

# Print the contracts created
print("\nContracts Created:")
for contract in virg_call_contracts:
    print(f"Symbol: {contract.symbol}, Expiry: {contract.lastTradeDateOrContractMonth}, Strike: {contract.strike}")