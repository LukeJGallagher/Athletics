import os
import sqlite3

# Dictionary mapping competition names to their Competition_IDs
competition_cids = {
    "Olympics 2021": "12992925",
    "Olympics 2016": "12877460",
    "Olympics 2012": "12825110",
    "Olympics 2008": "12042259",
    "Olympics 2004": "8232064",
    "Olympics 2000": "8257021",
    "Olympics 1996": "12828534",
    "Olympics 1992": "12828528",
    "Olympics 1988": "12828533",
    "Olympics 1984": "12828557",
    "World Champs 2023": "13046619",
    "World Champs 2022": "13002354",
    "World Champs 2019": "12935526",
    "World Champs 2017": "12898707",
    "World Champs 2013": "12844203",
    "World Champs 2011": "12814135",
    "World Champs 2009": "12789100",
    "World Champs 2007": "10626603",
    "World Champs 2005": "8906660",
    "World Champs 2003": "7993620",
    "World Champs 2001": "8257083",
    "World Champs 1999": "8256922",
    "World Champs 1997": "12996366",
    "World Champs 1995": "12828581",
    "World Champs 1993": "12828580",
    "World Champs 1991": "12996365",
    "World Champs 1987": "12996362",
    "World Champs 1983": "8255184",
    "World Champs 1980": "13015297",
    "World Junior 2022": "13002364",
    "World Junior 2021": "12993802",
    "World Junior 2018": "12910467",
    "World Junior 2016": "12876812",
    "World Junior 2014": "12853328",
    "World Junior 2012": "12824526",
    "World Junior 2008": "11909738",
    "World Junior 2006": "9238748",
    "World Junior 2004": "8196283",
    "World Junior 2000": "8256856",
    "Asian Games 2018": "12911586",
    "Asian Games 2014": "12854365",
    "Asian Junior 2018": "12908650",
    "Asian Junior 2016": "12875239",
    "Asian Junior 2014": "12852006",
    "Asian Junior 2012": "12823993",
    "Asian Junior 2010": "12799144",
    "Asian Junior 2008": "11800375",
    "Asian Youth 2019": "12924200",
    "Asian Youth 2017": "12896175",
    "Asian Youth 2015": "12860083",
    "Youth Olympics 2018": "12912645",
    "Youth Olympics 2014": "12853759",
    "Youth Olympics 2010": "12800536",
    "Asian Athletics Champs 2019": "12927085",
    "Asian Athletics Champs 2017": "12897142",
    "Asian Athletics Champs 2015": "12861120",
    "Asian Athletics Champs 2013": "12843333",
    "Asian Athletics Champs 2011": "12812847",
    "Asian Athletics Champs 2007": "10571413",
    "Asian Athletics Champs 2005": "8923929",
    "Asian Athletics Champs 2003": "7999347",
    "World Youth 2009": "12788414",
    "World Youth 2007": "10290642",
    "World Youth 2005": "8889520",
    "World Youth 2003": "7958195",
    "World Youth 2001": "8257029",
    "World Youth 1999": "12888228",
    "World Indoor 2022": "13002200",
    "World Indoor 2018": "12904540",
    "World Indoor 2016": "12871065",
    "World Indoor 2014": "12848482",
    "World Indoor 2012": "12821019",
    "World Indoor 2010": "12794620",
    "World Indoor 2008": "11465020",
    "World Indoor 2006": "9050779",
    "World Indoor 2004": "8066697",
    "World Indoor 2003": "7863109",
    "World Indoor 2001": "8257397",
    "World Indoor 1999": "8588390",
    "World Indoor 1997": "12828553",
    "World Indoor 1995": "12828567",
    "World Indoor 1993": "12828599",
    "World Indoor 1991": "12828568",
    "World Indoor 1989": "12829172",
    "World Indoor 1987": "12828628",
    "World Indoor 1985": "13092013"
}

def create_major_championships_database(source_db_path, target_db_path):
    if os.path.exists(target_db_path):
        print(f"Removing existing database at {target_db_path}")
        os.remove(target_db_path)

    source_conn = sqlite3.connect(source_db_path)
    source_cursor = source_conn.cursor()

    source_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='athletics_data'")
    schema_result = source_cursor.fetchone()
    if schema_result is None:
        raise ValueError("The table 'athletics_data' was not found in the source database.")
    table_schema = schema_result[0]

    target_conn = sqlite3.connect(target_db_path)
    target_cursor = target_conn.cursor()
    target_cursor.execute(table_schema)

    competition_ids = list(competition_cids.values())
    placeholders = ",".join("?" for _ in competition_ids)
    query = f"SELECT * FROM athletics_data WHERE Competition_ID IN ({placeholders})"

    source_cursor.execute(query, competition_ids)
    rows = source_cursor.fetchall()

    source_cursor.execute("PRAGMA table_info(athletics_data)")
    columns_info = source_cursor.fetchall()
    num_columns = len(columns_info)
    column_names = [col[1] for col in columns_info]

    # Find the index of the Competition_ID column
    try:
        comp_id_index = column_names.index("Competition_ID")
    except ValueError:
        raise ValueError("Column 'Competition_ID' not found in athletics_data table.")

    placeholders = ",".join("?" for _ in range(num_columns))
    insert_query = f"INSERT INTO athletics_data VALUES ({placeholders})"
    target_cursor.executemany(insert_query, rows)
    target_conn.commit()

    print(f"Created major championships database at {target_db_path} with {len(rows)} records.")

    # Error checking: Find missing competition IDs
    found_ids = {str(row[comp_id_index]) for row in rows}
    missing = [name for name, cid in competition_cids.items() if cid not in found_ids]
    if missing:
        print("Warning: The following competitions were not found in the source DB:")
        for name in missing:
            print(f" - {name}")

    source_conn.close()
    target_conn.close()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    source_db_path = os.path.join(BASE_DIR, "SQL", "athletics.db")
    target_db_path = os.path.join(BASE_DIR, "SQL", "major_championships.db")

    create_major_championships_database(source_db_path, target_db_path)