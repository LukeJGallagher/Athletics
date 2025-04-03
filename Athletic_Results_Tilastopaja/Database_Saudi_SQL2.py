import os
import sqlite3

def create_saudi_database(source_db_path, target_db_path):
    if os.path.exists(target_db_path):
        print(f"Removing existing database at {target_db_path}")
        os.remove(target_db_path)

    # Connect to source DB
    source_conn = sqlite3.connect(source_db_path)
    source_cursor = source_conn.cursor()

    # Get table schema
    source_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='athletics_data'")
    schema_result = source_cursor.fetchone()
    if schema_result is None:
        raise ValueError("The table 'athletics_data' was not found in the source database.")
    table_schema = schema_result[0]

    # Create target DB + table
    target_conn = sqlite3.connect(target_db_path)
    target_cursor = target_conn.cursor()
    target_cursor.execute(table_schema)

    # ✅ New query to include both fields
    query = """
        SELECT * FROM athletics_data
        WHERE Athlete_Country = 'Saudi Arabia'
        OR Athlete_CountryCode = 'Saudi Arabia'
    """
    source_cursor.execute(query)
    rows = source_cursor.fetchall()

    # Get column info for insert
    source_cursor.execute("PRAGMA table_info(athletics_data)")
    columns_info = source_cursor.fetchall()
    num_columns = len(columns_info)
    placeholders = ",".join(["?"] * num_columns)

    insert_query = f"INSERT INTO athletics_data VALUES ({placeholders})"
    target_cursor.executemany(insert_query, rows)
    target_conn.commit()

    print(f"Created new database at {target_db_path} with {len(rows)} Saudi athletes.")

    # Close
    source_conn.close()
    target_conn.close()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    source_db_path = os.path.join(BASE_DIR, "SQL", "athletics.db")
    target_db_path = os.path.join(BASE_DIR, "SQL", "saudi_athletes.db")
    create_saudi_database(source_db_path, target_db_path)
