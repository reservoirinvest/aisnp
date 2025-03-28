import re

import pandas as pd
import pytz
import yaml
from datetime import datetime

from utils import ROOT

# %%

FILENAME = 'U8898867_20190102_20191231.csv'

a_file = ROOT/'report'/'history'/FILENAME

# Read the CSV file
df = pd.read_csv(a_file,
                 on_bad_lines='warn',
                 sep='\t',
                 encoding='utf-8',
                 quotechar='"',
                 engine='python',
                 quoting=3)

# Make header as the first row
df.loc[-1] = df.columns
df.index = df.index + 1
df.sort_index(inplace=True)

# Reset column names
df.columns = [f'Column_{i+1}' for i in range(len(df.columns))]

# Timezone for conversion
est = pytz.timezone('US/Eastern')

# Regex pattern to find datetime strings
datetime_pattern = r'"(\d{4}-\d{2}-\d{2}, \d{2}:\d{2}:\d{2})"'

# Regex patterns for datetime and numeric strings
datetime_pattern = r'"(\d{4}-\d{2}-\d{2}, \d{2}:\d{2}:\d{2})"'
numeric_pattern = r'"([+-]?[\d,]+(?:\.\d+)?)"'

# Iterate through rows and replace datetime and numeric strings
for i in range(len(df)):
    row = df.loc[i, 'Column_1']
    
    # Convert datetime strings to epoch
    datetime_match = re.search(datetime_pattern, row)
    if datetime_match:
        datetime_str = datetime_match.group(1)
        try:
            dt = pd.to_datetime(datetime_str, format='%Y-%m-%d, %H:%M:%S')
            epoch = int(est.localize(dt).timestamp())
            row = row.replace(f'"{datetime_str}"', str(epoch))
        except Exception as e:
            print(f"Error converting datetime {datetime_str}: {e}")
    
    # Convert numeric strings to integers
    numeric_matches = re.findall(numeric_pattern, row)
    for numeric_str in numeric_matches:
        try:
            # Remove commas and convert to integer
            cleaned_num = numeric_str.replace(',', '')
            int_value = int(float(cleaned_num))
            row = row.replace(f'"{numeric_str}"', str(int_value))
        except Exception as e:
            print(f"Error converting numeric {numeric_str}: {e}")
    
    # Update the row
    df.loc[i, 'Column_1'] = row

# Calculate max columns
max_cols = df.apply(lambda x: x.str.count(',').max()).max()+1

# Convert all values to strings and pad with commas
for col in df.columns:
    df[col] = df[col].astype(str).apply(
        lambda x: x + ',' * (max_cols - x.count(',') - 1) 
        if x.count(',') + 1 < max_cols else x
    )

# Split columns
df_split = df[df.columns[0]].str.split(',', expand=True)
for col in df.columns[1:]:
    temp_split = df[col].str.split(',', expand=True)
    df_split = pd.concat([df_split, temp_split], axis=1)

# Reset column names
df_split.columns = [f'Column_{i+1}' for i in range(len(df_split.columns))]

# Remove completely empty columns
df_split = df_split.dropna(axis=1, how='all')

# %%
# Split df_split into multiple DataFrames
dataframes = {}

# --- Configuration ---
DATE_TIME_COLUMN = 'Date/Time'  # Name of your date/time column
TIMEZONE = "US/Eastern"  # Your desired timezone

# Find indices of 'Header' rows
header_indices = df_split[df_split['Column_2'] == 'Header'].index

for i in range(len(header_indices)):
    # Get the start and end indices for each section
    start_idx = header_indices[i]
    end_idx = header_indices[i+1] if i+1 < len(header_indices) else len(df_split)
    
    # Get the section DataFrame
    section_df = df_split.loc[start_idx:end_idx-1].copy()
    
    # Get the first column value to create DataFrame name
    first_col_value = section_df['Column_1'].iloc[0]
    df_name = 'df_' + first_col_value[:4].lower() + str(i)
    
    # Use the header row's columns as column names
    if not section_df.empty:
        header_row = section_df[section_df['Column_2'] == 'Header']
        if not header_row.empty:
            # Extract column names from the header row
            new_columns = header_row.iloc[0, 2:].tolist()
            
            # Filter out data rows
            data_df = section_df[section_df['Column_2'] == 'Data'].copy()
            
            # Rename columns and dropo Column_2
            if len(new_columns) <= len(data_df.columns):
                data_df.columns = ['Type', 'Column_2'] + new_columns[:len(data_df.columns)-2]
                data_df.drop('Column_2', axis=1, inplace=True)

                # Filter out 'Summary' rows
                if 'DataDiscriminator' in data_df.columns:
                    try:
                        data_df = data_df[data_df['DataDiscriminator'] != 'Summary'].reset_index(drop=True)
                    except:
                        pass

                if 'Open' in data_df.columns:
                    est = pytz.timezone("US/Eastern")
                    data_df['Open'] = data_df['Open'].apply(
                        lambda x: datetime.fromtimestamp(int(x), tz=pytz.utc)
                            .astimezone(est)
                            .strftime("%Y-%m-%d, %H:%M:%S")
                    )

                # --- Process the Date/Time column ---
                if DATE_TIME_COLUMN in data_df.columns:
                    est = pytz.timezone(TIMEZONE)

                    for index, x in data_df[DATE_TIME_COLUMN].items():
                        if isinstance(x, str):
                            try:
                                dt = datetime.strptime(x, "%Y-%m-%d")  # Use datetime.strptime
                                dt = dt.replace(hour=16, minute=0, second=0)
                                dt_est = est.localize(dt)
                                data_df.loc[index, DATE_TIME_COLUMN] = dt_est.strftime("%Y-%m-%d, %H:%M:%S")
                            except ValueError:
                                try:
                                    epoch = int(x)
                                    utc_datetime = datetime.fromtimestamp(epoch, tz=pytz.utc)  # Use datetime.fromtimestamp
                                    dt_est = utc_datetime.astimezone(est)
                                    data_df.loc[index, DATE_TIME_COLUMN] = dt_est.strftime("%Y-%m-%d, %H:%M:%S")
                                except ValueError:
                                    data_df.loc[index, DATE_TIME_COLUMN] = pd.NaT
                        else:
                            data_df.loc[index, DATE_TIME_COLUMN] = pd.NaT
                
                data_df.insert(0, 'Year', int(FILENAME[len(FILENAME)-8-4:-8]))
                
                # Store the DataFrame
                dataframes[df_name] = data_df
                print(f"Created {df_name} with shape: {data_df.shape}")



# Concatenate dataframes that have the same first 7 characters in their name
dfs = {}
for key in dataframes.keys():
    if len(key) > 7:
        key_group = key[:7]
        if key_group not in dfs:
            dfs[key_group] = pd.DataFrame()
        dfs[key_group] = pd.concat([dfs[key_group], dataframes[key]], ignore_index=True)

# Remove the concatenated dataframes from dataframes
for key in list(dfs.keys()):
    for key_group in list(dataframes.keys()):
        if key_group.startswith(key):
            del dataframes[key_group]

# Add the concatenated dataframes to dataframes
dataframes.update(dfs)

# Print out the names of created DataFrames
print("\nCreated DataFrames:")
for name in dataframes.keys():
    print(name)

# %%
# Read the YAML file
with open(ROOT/'config'/'ib_trade_codes.yml', 'r') as file:
    code_dict = yaml.safe_load(file)

# Function to expand codes
def expand_codes(code_string):
    if pd.isna(code_string):
        return ''
    codes = code_string.split(';')
    return '; '.join([f"{code}: {code_dict.get(code, '')}" for code in codes])

# Create separate dataframes
df_td = dataframes.get('df_trad', pd.DataFrame())
df_div = dataframes.get('df_divi', pd.DataFrame())
df_xae = dataframes.get('df_opti', pd.DataFrame())
df_open = dataframes.get('df_open', pd.DataFrame())
df_header = dataframes.get('df_stat', pd.DataFrame())

# Remove Header, Total, and SubTotal rows
for df in [df_td, df_div, df_xae, df_open]:
    df.drop(df[df['Column_2'].isin(['Header', 'Total', 'SubTotal'])].index, inplace=True)

# Expand the 'Code' column
df_td['Column_3'] = df_td['Column_3'].apply(expand_codes)
df_xae['Column_3'] = df_xae['Column_3'].apply(expand_codes)

# Split the 'Symbol' column in df_xae
df_xae[['Column_4', 'Column_5', 'Column_6', 'Column_7']] = df_xae['Column_4'].str.extract(r'(\w+)\s+(\d+\w+\d+)\s+(\d+(?:\.\d+)?)\s+(\w)')
df_xae['Column_5'] = pd.to_datetime(df_xae['Column_5'], format='%d%b%y').dt.strftime('%d-%b-%Y')

# Reset index for all dataframes
df_td.reset_index(drop=True, inplace=True)
df_div.reset_index(drop=True, inplace=True)
df_xae.reset_index(drop=True, inplace=True)
df_open.reset_index(drop=True, inplace=True)
df_header.reset_index(drop=True, inplace=True)

# Print the first few rows of each dataframe to verify
print(df_td.head())
print(df_div.head())
print(df_xae.head())
print(df_open.head())
print(df_header.head())
