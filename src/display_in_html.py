# %%
# OUTLINES BREACHES

import os
import webbrowser
from pathlib import Path
from utils import ROOT

with open(ROOT / 'src' / 'styles.css', 'r') as f:
    css = f.read()

def style_rows(df, rows_index, message=None, calc=None):
    def _style_rows(row):
        if row.name in rows_index:
            return ['background-color: black; color: white'] * len(row)
        elif row['source'] == 'und':
            if row['unPnL'] > 0:
                return ['background-color: green; color: white'] * len(row)
            elif row['unPnL'] < 0:
                return ['background-color: red; color: white'] * len(row)
        return ['background-color: #313131; color: #b4b1b1'] * len(row)
    
    # DRY approach to formatting
    int_columns = ['qty', 'position']
    float_columns = ['undPrice', 'strike', 'avgCost', 'unPnL', 'dte', 'mktVal']
    format_dict = {col: '{:.0f}' for col in int_columns}
    format_dict.update({col: '{:.2f}' for col in float_columns})
    
    # Prepare caption
    caption = f'<font size=4>{message}</font>' if message else None
    if message and calc is not None:
        # Convert calc to a numeric value if it's a string
        try:
            calc_numeric = float(calc.replace(',', ''))
            caption += f'<font size=4> US${calc_numeric:,.0f}</font>'
        except (ValueError, AttributeError):
            caption += f'<font size=4> US${calc}</font>'
    
    # Styling of option breaches
    dfs = (
        df.style
        .format(format_dict)
        .apply(_style_rows, axis=1)
        .hide(axis='index')
        .hide(axis='columns', subset=df.columns[df.isna().all()])
    )
    
    # Add caption if provided
    if caption:
        dfs = dfs.set_caption(caption)
    
    return dfs

# Colour rows for option breaches
calls_lt_und = df[(df.right == 'C') & (df.strike < df.undPrice)].index.tolist()
puts_gt_und = df[(df.right == 'P') & (df.strike > df.undPrice)].index.tolist()
option_breach_index = list(set(calls_lt_und).union(set(puts_gt_und)))

# Initialize variables
opt_breached_df = pd.DataFrame()
total_breach_pnl = "0"  # Default value if no breaches
dfs = None  # Initialize dfs as None

# Data Manipulation
if option_breach_index:
    # Get the breach PnL for the relevant symbols
    breach_pnl = df[(df.source == "und") &
                  (df.symbol.isin(df.loc[option_breach_index, 'symbol'].unique())) &
                  (df.secType == 'STK')].unPnL

    # Calculate the total breach PnL for the caption
    total_breach_pnl = format(breach_pnl.sum(), ",.0f")

    # Filter the DataFrame for display
    opt_breached_df = df[df.symbol.isin(df.loc[option_breach_index, 'symbol'])]

# Only show the table if we have breached options
if not opt_breached_df.empty:
    dfs = style_rows(
        opt_breached_df, 
        rows_index=option_breach_index, 
        message='Option cover breaches are generating', 
        calc=total_breach_pnl
    )

def is_running_in_bare_terminal() -> bool:
    """
    Determines if the code is running in a bare terminal.
    """
    # Check for common IDE and notebook environments
    if any(env in os.environ for env in ['VSCODE_PID', 'PYCHARM_HOSTED', 'JUPYTER_RUNTIME_DIR', 'SPYDER_ARGS']):
        return False
    
    # Check for VSCode's integrated terminal
    if 'TERM_PROGRAM' in os.environ and 'vscode' in os.environ['TERM_PROGRAM'].lower():
        return False
    
    # Consider all other environments as "bare terminal" for this purpose
    return True

def webbrowser_styled_df(styled_df):
    # Capture print statements in a message pad
    msg_pad = []
    
    # Capture existing print statements
    msg_pad.append("FINANCIALS")
    msg_pad.append('==========')
    for k, v in fin.items():
        if v:
            if v > 1:
                msg_pad.append(f"{k}: {format(v, ',.0f')}")
            else:
                msg_pad.append(f"{k}: {format(v, ',.2f')}")
    
    msg_pad.append('\nRisk Analysis')
    msg_pad.append('-------------')
    risk_msg = []
    risk_msg.append(f'Our max risk is ${df_risk.unprot_val.sum():,.0f} from ${fin.get("stocks", 0):,.0f} stock for {df_risk.dte.mean():.1f} days at risk premium of ${df_risk.cost.sum():,.0f}')
    unprotected_stocks = df[(df.source == "und") & (df.state == "unprotected")].symbol.unique()
    if unprotected_stocks.size > 0:
        risk_msg.append(f'Unprotected stocks: {", ".join(unprotected_stocks)}')
    else:
        risk_msg.append('All stock positions are protected!')
    msg_pad.extend(risk_msg)
    
    msg_pad.append('\nRewards')
    msg_pad.append('-------')
    reward_msg = (
        f'Total reward this month is expected to be ${total_reward:,.0f} '
        f'Our maximum cover reward in {df_reward.dte.mean():.1f} days is '
        f'${df_reward.max_reward.sum():,.0f}, if all covers get blown.\n\n'
        f'Our cover premiums from covering options is ${abs(df_reward.premium.sum()):,.0f} this week from our stock positions\n'
        f' ...this can be projected to give us ${cover_projection:,.0f} for the protected period\n' 
    )
    msg_pad.append(reward_msg)
    
    sow_msg = (
        f'Our sowed reward in about {df_sowed.dte.mean():.1f} dte days is ${df_sowed.avgCost.sum():,.0f}'
    )
    msg_pad.append(sow_msg)
    
    # Add premiums and profit section if applicable
    if cov_premium > 0 or nkd_premium > 0:
        msg_pad.append('\nPREMIUMS AND PROFIT from df_cov and df_nkd')
        msg_pad.append('==========================================')
        msg_pad.append(f"Total Premium available is {format(cov_premium + nkd_premium, ',.0f')}")
        msg_pad.append(f"...Cover Premium: {format(cov_premium, ',.0f')}")
        msg_pad.append(f"...Naked Premiums: {format(nkd_premium, ',.0f')}")
        msg_pad.append(f"Max possible profit from covers: {format(maxProfit, ',.0f')}")
    
    msg_pad.append('\nSYMBOL COUNT BY STATE')
    msg_pad.append('=====================')
    msg_pad.append(', '.join(f"{state}: {len(df.symbol.unique())}" for state, df in df_unds.groupby('state')))
    
    msg_pad.append('\nCOUNT OF SYMBOLS IN EACH DATAFRAME')
    msg_pad.append('==================================')
    msg_pad.append(', '.join(f"{k}: {len(v) if v is not None else 0}" for k, v in {'df_cov': df_cov, 'df_protect': df_protect, 'df_reap': df_reap, 'df_nkd': df_nkd, }.items()))
    
    # Convert the styled DataFrame to HTML using to_html()
    html = styled_df.to_html()

    # Print the message pad contents
    print("\n".join(msg_pad))

    # Read the contents of the CSS file
    try:
        css_path = Path(ROOT / 'src' / 'styles.css')
        css_contents = css_path.read_text()
    except Exception as e:
        print(f"Error reading CSS file: {e}")
        css_contents = ""

    # Add custom CSS to improve the appearance
    custom_css = f"""
    <style>
        {css_contents}
        .msg-pad {{ 
            font-family: monospace; 
            white-space: pre-wrap; 
            background-color: #f4f4f4; 
            padding: 10px; 
            margin-top: 20px; 
            border: 1px solid #ddd; 
        }}
    </style>
    """

    # Create message pad HTML
    msg_pad_html = '<div class="msg-pad">' + '\n'.join(msg_pad) + '</div>'

    # Create a timestamp for the top of the HTML
    from datetime import datetime
    current_time = datetime.now().strftime('%d-%b-%Y %H:%M:%S')
    timestamp_html = f'<div class="timestamp">Report Generated: {current_time}</div>'

    # Modify the full_html to include the timestamp
    full_html = f"""
    <html>
    <head>{custom_css}
        <style>
            .timestamp {{
                font-family: Arial, sans-serif;
                text-align: right;
                color: #666;
                padding: 10px;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        {timestamp_html}
        {msg_pad_html}
        {html}
    </body>
    </html>"""

    # Ensure the report directory exists
    report_dir = ROOT / 'report'
    report_dir.mkdir(exist_ok=True)

    # Create a filename with current timestamp in yyyymmdd_hh format
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H')
    report_path = report_dir / f'report_{timestamp}.html'

    # Write the full HTML to the file
    with report_path.open('w') as f:
        f.write(full_html)

    # Set read permissions for the file
    report_path.chmod(0o644)

    # Open the HTML file in the default web browser
    try:
        webbrowser.open(f'file://{report_path}')
    except Exception as e:
        print(f"Error opening browser: {e}")
        print(f"Try opening this file manually: {report_path}")

    # Explicitly prevent the Styler object from being printed in the terminal
    return

# Main logic
if is_running_in_bare_terminal():
    # If in a bare terminal, render the styled DataFrame in the browser
    webbrowser_styled_df(dfs)
else:
    # If in an IDE or notebook, use IPython.display
    from IPython.display import display, HTML
    display(HTML('<style>pre { white-space: pre-wrap; }</style>'))
    display(dfs)