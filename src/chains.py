
# %%
# MAKE QUALIFIED CHAINS

from utils import get_pickle, ROOT, get_dte, clean_ib_util_df
import pandas as pd
from ibfuncs import get_ib, qualify_me
from ib_async import Option, IB
from tqdm.asyncio import tqdm

ch = get_pickle(ROOT / "data" / "chains.pkl", print_msg=False)
ch['dte'] = ch.expiry.apply(get_dte)
ch = ch[ch.dte >= 0]

ch['right'] = 'C'
ch = pd.concat([ch, ch.assign(right='P')], ignore_index=True).reset_index(drop=True)

cts = [Option(s, e, k, "C", "SMART") for s, e, k in zip(ch.symbol, ch.expiry, ch.strike)]

async def qualify_me_in_chunks(
    ib: IB, 
    data: list, 
    chunk_size: int = 200, 
    desc: str = "Qualifying contracts"
) -> list:
    """Process contracts in chunks using qualify_me() function.
    
    Args:
        ib (IB): Interactive Brokers connection
        data (list): List of contracts to qualify
        chunk_size (int, optional): Number of contracts to process in each chunk. Defaults to 200.
        desc (str, optional): Description for progress bar. Defaults to "Qualifying contracts".
    
    Returns:
        list: List of qualified contracts
    """
    if not data:
        return []

    qualified = []

    with tqdm(total=len(data), desc=desc, unit="contract") as pbar:
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            
            qualified_chunk = await qualify_me(ib, chunk, desc=f"{desc} (chunk)")
            qualified.extend(qualified_chunk)

            # Update progress bar
            pbar.update(len(chunk))

    return qualified

# In an async context
with get_ib('SNP') as ib:
    qcts = ib.run(
        qualify_me_in_chunks(ib, cts, desc='Qualifying SNP Chains')
    )

df_ch = clean_ib_util_df([q for q in qcts if q])

df_ch.to_pickle(ROOT / "data" / "df_ch.pkl")