# %%
# RENDER IN WEBBROWSER
import webbrowser
import tempfile
import os

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

def display_styled_df_in_browser(styled_df):
    # Convert the styled DataFrame to HTML using to_html()
    html = styled_df.to_html()
    
    # Add custom CSS to improve the appearance
    custom_css = """
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            margin: 20px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        th, td {
            padding: 10px;
            text-align: left;
            border: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #ddd;
        }
        caption {
            font-size: 1.5em;
            margin: 10px;
        }
    </style>
    """
    # Combine the custom CSS with the HTML
    full_html = f"<html><head>{custom_css}</head><body>{html}</body></html>"
    
    # Create a temporary HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        f.write(full_html)
        temp_file_path = f.name

    # Set read permissions for the file
    os.chmod(temp_file_path, 0o644)

    print(f"Temporary file created: {temp_file_path}")

    # Open the HTML file in the default web browser
    try:
        webbrowser.open('file://' + temp_file_path)
    except Exception as e:
        print(f"Error opening browser: {e}")
        print(f"Try opening this file manually: {temp_file_path}")

print(f"Is running in bare terminal: {is_running_in_bare_terminal()}")
if is_running_in_bare_terminal():
    display_styled_df_in_browser(dfs)
else:
    display(dfs)