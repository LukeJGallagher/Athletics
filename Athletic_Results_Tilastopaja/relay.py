import streamlit as st  
import sqlite3  
import pandas as pd  
import os  
  
# You must define or import these helper functions and dtype dictionaries:  
# clean_columns, parse_result, coerce_dtypes, SAUDI_COLUMNS_DTYPE, and MAJOR_COLUMNS_DTYPE  
  
# Example dummy definitions (replace with your actual ones)  
def clean_columns(df):  
    df.columns = [c.strip() for c in df.columns]  
    return df  
  
def parse_result(result, event):  
    try:  
        # Convert result to float if possible  
        return float(result)  
    except:  
        return None  
  
def coerce_dtypes(df, dtype_dict):  
    for col, dtype in dtype_dict.items():  
        df[col] = df[col].astype(dtype, errors='ignore')  
    return df  
  
SAUDI_COLUMNS_DTYPE = {}  # Define according to your dataset  
MAJOR_COLUMNS_DTYPE = {}  # Define according to your dataset  
  
@st.cache_data  
def load_db(db_filename: str):  
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    db_path = os.path.join(base_dir, "SQL", db_filename)  
    if not os.path.exists(db_path):  
        st.warning(db_filename + " not found in 'SQL' folder.")  
        return pd.DataFrame()  
    conn = sqlite3.connect(db_path)  
    df = pd.read_sql_query("SELECT * FROM athletics_data", conn)  
    conn.close()  
    df = clean_columns(df)  
    # Filter out rows where Event is missing  
    if "Event" in df.columns:  
        df = df[df["Event"].notnull()]  
    if 'Result' in df.columns and 'Event' in df.columns:  
        df['Result_numeric'] = df.apply(lambda row: parse_result(row['Result'], row['Event']), axis=1)  
    if db_filename == "saudi_athletes.db":  
        df = coerce_dtypes(df, SAUDI_COLUMNS_DTYPE)  
    elif db_filename == "major_championships.db":  
        df = coerce_dtypes(df, MAJOR_COLUMNS_DTYPE)  
    if 'Year' not in df.columns and 'Start_Date' in df.columns:  
        df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')  
        df['Year'] = df['Start_Date'].dt.year  
    return df  
  
# Load major database and print debug info  
df_major = load_db("major_championships.db")  
if df_major.empty:  
    st.error("No data loaded from major_championships.db")  
else:  
    df_major["Event_clean"] = df_major["Event"].str.strip().str.lower()  
    # Filter for relay events based on a cleaned list of events  
    relay_events_clean = ["4 x 100m", "4 x 400m", "4 x 400m mixed relay"]  
    relay_rows = df_major[df_major["Event_clean"].isin([s.lower() for s in relay_events_clean])]  
      
    st.write("Relay events sample:", relay_rows[["Event", "Start_Date", "Year"]].head(10))  
    st.write("Unique Start_Date values:", df_major["Start_Date"].unique())  
    st.write("Unique Result values:", df_major["Result"].unique())  
    st.write("Unique Competition values:", df_major["Competition"].unique())  
    st.write("Unique Round values:", df_major["Round"].unique())  
    st.write("Cleaned Start_Date sample:", df_major["Start_Date"].head(10))  
    st.write("Cleaned Year sample:", df_major["Year"].head(10))  
    st.write("Result_numeric stats:", df_major["Result_numeric"].describe())  
      
st.write("Done.")  