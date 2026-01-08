"""Utils Module used for Data Analysis and EDA"""

from IPython.display import display


# Summary Function
def summarize_df(df, df_name="df"):
    """
    Display key information about a DataFrame:
    - info()
    - describe()
    - duplicated rows
    - count of missing values
    """
    print(f"===== DataFrame ({df_name.upper()}) Summary =====")
    print("===== DataFrame Index =====")
    display(df.index)
    print("===== DataFrame Info =====")
    df.info()
    print("\n===== DataFrame Description =====")
    display(
        df.describe(include="all")
    )  # include='all' to describe non-numeric columns too
    print("\n===== Duplicate Rows =====")
    duplicates = df[df.duplicated(keep=False)]
    if not duplicates.empty:
        display(duplicates)
    else:
        print("No duplicate rows found.")
    print("\n===== Missing Values per Column =====")
    print(df.isna().sum())
