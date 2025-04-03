import os
import sqlite3
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Check Table Structure", layout="wide")
st.title("Review major_championships.db Tables & Columns")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "SQL", "major_championships.db")

if not os.path.exists(db_path):
    st.error("major_championships.db not found in the SQL folder.")
    st.stop()

conn = sqlite3.connect(db_path)

# 1) Show a list of all tables
tables_list = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
tables = [t[0] for t in tables_list]

if not tables:
    st.error("No tables found in major_championships.db.")
    conn.close()
    st.stop()

chosen_table = st.selectbox("Select a table to inspect", options=tables)
st.write(f"**Selected Table:** {chosen_table}")

try:
    # 2) Get the column info
    col_info = conn.execute(f"PRAGMA table_info({chosen_table});").fetchall()
    st.write("**Columns in this table:**")
    col_details = pd.DataFrame(col_info, columns=["cid", "name", "type", "notnull", "default_value", "pk"])
    st.dataframe(col_details)

    # 3) Show first few rows
    df_preview = pd.read_sql_query(f"SELECT * FROM {chosen_table} LIMIT 10;", conn)
    st.subheader(f"First 10 Rows of {chosen_table}")
    st.dataframe(df_preview)

except Exception as e:
    st.error(f"Error reading {chosen_table}: {e}")

conn.close()
