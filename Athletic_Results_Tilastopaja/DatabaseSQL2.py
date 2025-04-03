import os
import pandas as pd
import sqlite3

def update_database(data_folder='Data', db_path='athletics.db'):
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_files (
            file_name TEXT PRIMARY KEY
        )
    ''')
    conn.commit()

    main_table = 'athletics_data'

    for file in os.listdir(data_folder):
        if file.endswith('.parquet'):
            cursor.execute("SELECT 1 FROM processed_files WHERE file_name = ?", (file,))
            if cursor.fetchone():
                print(f"Skipping {file} (already processed).")
                continue

            file_path = os.path.join(data_folder, file)
            try:
                df = pd.read_parquet(file_path)

                # ✅ Normalize date format
                if 'Start_Date' in df.columns:
                    df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce').dt.strftime('%Y-%m-%d')

                # ✅ Fill missing Athlete_Country from Athlete_CountryCode
                if 'Athlete_Country' in df.columns and 'Athlete_CountryCode' in df.columns:
                    df['Athlete_Country'] = df['Athlete_Country'].fillna(df['Athlete_CountryCode'])

                # ✅ Clean both fields for consistency
                for col in ['Athlete_Country', 'Athlete_CountryCode']:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.strip().str.title()

                # Save to database
                df.to_sql(main_table, conn, if_exists='append', index=False)

                cursor.execute("INSERT INTO processed_files (file_name) VALUES (?)", (file,))
                conn.commit()
                print(f"Processed and added {file} to the database.")
            except Exception as e:
                print(f"Error processing {file}: {e}")

    conn.close()

if __name__ == "__main__":
    update_database()
