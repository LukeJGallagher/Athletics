import os
import re
import sqlite3
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
import base64
import datetime
import math

# Increase the maximum number of cells allowed by Pandas Styler.
pd.set_option("styler.render.max_elements", 1000000)

##########################################################
# 0) Our dictionary of Competition_ID-based filter
##########################################################
MAJOR_COMPETITIONS_CID = {
    "Olympics": {
        "2021": {"CID": "12992925", "Season": "2021"},
        "2016": {"CID": "12877460", "Season": "2016"},
        "2012": {"CID": "12825110", "Season": "2012"},
        "2008": {"CID": "12042259", "Season": "2008"},
        "2004": {"CID": "8232064",  "Season": "2004"},
        "2000": {"CID": "8257021",  "Season": "2000"},
        "1996": {"CID": "12828534", "Season": "1996"},
        "1992": {"CID": "12828528", "Season": "1992"},
        "1988": {"CID": "12828533", "Season": "1988"},
        "1984": {"CID": "12828557", "Season": "1984"}
    },
    "World Championships": {
        "2023": {"CID": "13046619", "Season": "2023"},
        "2022": {"CID": "13002354", "Season": "2022"},
        "2019": {"CID": "12935526", "Season": "2019"},
        "2017": {"CID": "12898707", "Season": "2017"},
        "2013": {"CID": "12844203", "Season": "2013"},
        "2011": {"CID": "12814135", "Season": "2011"},
        "2009": {"CID": "12789100", "Season": "2009"},
        "2007": {"CID": "10626603", "Season": "2007"},
        "2005": {"CID": "8906660",  "Season": "2005"},
        "2003": {"CID": "7993620",  "Season": "2003"},
        "2001": {"CID": "8257083",  "Season": "2001"},
        "1999": {"CID": "8256922",  "Season": "1999"},
        "1997": {"CID": "12996366", "Season": "1997"},
        "1995": {"CID": "12828581", "Season": "1995"},
        "1993": {"CID": "12828580", "Season": "1993"},
        "1991": {"CID": "12996365", "Season": "1991"},
        "1987": {"CID": "12996362", "Season": "1987"},
        "1983": {"CID": "8255184",  "Season": "1983"},
        "1980": {"CID": "13015297", "Season": "1980"}
    },
    "World Junior Championships": {
        "2022": {"CID": "13002364", "Season": "2022"},
        "2021": {"CID": "12993802", "Season": "2021"},
        "2018": {"CID": "12910467", "Season": "2018"},
        "2016": {"CID": "12876812", "Season": "2016"},
        "2014": {"CID": "12853328", "Season": "2014"},
        "2012": {"CID": "12824526", "Season": "2012"},
        "2008": {"CID": "11909738", "Season": "2008"},
        "2006": {"CID": "9238748",  "Season": "2006"},
        "2004": {"CID": "8196283",  "Season": "2004"},
        "2000": {"CID": "8256856",  "Season": "2000"},
        "1998": {"CID": "12995279", "Season": "1998"},
        "1996": {"CID": "12995266", "Season": "1996"},
        "1994": {"CID": "12995254", "Season": "1994"},
        "1992": {"CID": "12995180", "Season": "1992"},
        "1990": {"CID": "12994817", "Season": "1990"},
        "1988": {"CID": "12994701", "Season": "1988"},
        "1986": {"CID": "12994430", "Season": "1986"}
    },
    "World Youth Championships": {
        "2017": {"CID": "12898087", "Season": "2017"},
        "2015": {"CID": "12862619", "Season": "2015"},
        "2013": {"CID": "12843556", "Season": "2013"},
        "2011": {"CID": "12812827", "Season": "2011"},
        "2009": {"CID": "12788414", "Season": "2009"},
        "2007": {"CID": "10290642", "Season": "2007"},
        "2005": {"CID": "8889520",  "Season": "2005"},
        "2003": {"CID": "7958195",  "Season": "2003"},
        "2001": {"CID": "8257029",  "Season": "2001"},
        "1999": {"CID": "12888228", "Season": "1999"}
    },
    "World Athletics Indoor Championships": {
        "2022": {"CID": "13002200", "Season": "2022"},
        "2018": {"CID": "12904540", "Season": "2018"},
        "2016": {"CID": "12871065", "Season": "2016"},
        "2014": {"CID": "12848482", "Season": "2014"},
        "2012": {"CID": "12821019", "Season": "2012"},
        "2010": {"CID": "12794620", "Season": "2010"},
        "2008": {"CID": "11465020", "Season": "2008"},
        "2006": {"CID": "9050779",  "Season": "2006"},
        "2004": {"CID": "8066697",  "Season": "2004"},
        "2003": {"CID": "7863109",  "Season": "2003"},
        "2001": {"CID": "8257397",  "Season": "2001"},
        "1999": {"CID": "8588390",  "Season": "1999"},
        "1997": {"CID": "12828553", "Season": "1997"},
        "1995": {"CID": "12828567", "Season": "1995"},
        "1993": {"CID": "12828599", "Season": "1993"},
        "1991": {"CID": "12828568", "Season": "1991"},
        "1989": {"CID": "12829172", "Season": "1989"},
        "1987": {"CID": "12828628", "Season": "1987"},
        "1985": {"CID": "13092013", "Season": "1985"}
    },
    "Asian Games": {
        "2018": {"CID": "12911586", "Season": "2018"},
        "2014": {"CID": "12854365", "Season": "2014"}
    },
    "Asian Athletics Championships": {
        "2019": {"CID": "12927085", "Season": "2019"},
        "2017": {"CID": "12897142", "Season": "2017"},
        "2015": {"CID": "12861120", "Season": "2015"},
        "2013": {"CID": "12843333", "Season": "2013"},
        "2011": {"CID": "12812847", "Season": "2011"},
        "2007": {"CID": "10571413", "Season": "2007"},
        "2005": {"CID": "8923929",  "Season": "2005"},
        "2003": {"CID": "7999347",  "Season": "2003"}
    },
    "Asian Athletics Junior": {
        "2018": {"CID": "12908650", "Season": "2018"},
        "2016": {"CID": "12875239", "Season": "2016"},
        "2014": {"CID": "12852006", "Season": "2014"},
        "2012": {"CID": "12823993", "Season": "2012"},
        "2010": {"CID": "12799144", "Season": "2010"},
        "2008": {"CID": "11800375", "Season": "2008"}
    },
    "Asian Youth Championships": {
        "2019": {"CID": "12924200", "Season": "2019"},
        "2017": {"CID": "12896175", "Season": "2017"},
        "2015": {"CID": "12860083", "Season": "2015"}
    },
    "Youth Olympics": {
        "2018": {"CID": "12912645", "Season": "2018"},
        "2014": {"CID": "12853759", "Season": "2014"},
        "2010": {"CID": "12800536", "Season": "2010"}
    },
    "Asian Indoor Championships": {
        "2018": {"CID": "12903393", "Season": "2018"},
        "2016": {"CID": "12870068", "Season": "2016"},
        "2014": {"CID": "12847970", "Season": "2014"},
        "2012": {"CID": "12820522", "Season": "2012"},
        "2010": {"CID": "12793955", "Season": "2010"},
        "2008": {"CID": "11377718", "Season": "2008"},
        "2004": {"CID": "8037718",  "Season": "2004"}
    }
}

###################################
# 1) Streamlit Setup
###################################
st.set_page_config(
    page_title="Athletics Analysis Dashboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

###################################
# 2) Dark Background
###################################
def set_background_from_url(url):
    css = f"""
    <style>
    .stApp {{
        background-image: url("{url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(0,0,0,0.6);
        padding: 2rem;
        border-radius: 12px;
        color: white;
    }}
    h1, h2, h3, h4 {{
        color: #00FF7F;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background_from_url("https://raw.githubusercontent.com/LukeJGallagher/Athletics/main/Athletic_Results_Tilastopaja/Background2.PNG")




###################################
# 3) DataFrame / Chart Helpers
###################################
def ensure_json_safe(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].apply(lambda x: int(x) if pd.notnull(x) else None)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].apply(lambda x: float(x) if pd.notnull(x) else None)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%Y-%m-%d').fillna("")
        else:
            df[col] = df[col].astype(str).replace("None", "")
    return df

def style_dark_df(df):
    return df.style.set_properties(
        **{'background-color': '#222', 'color': 'white', 'border-color': 'gray'}
    ).hide(axis='index')

###################################
# 4) Basic Data Cleaning + Parsing
###################################
def clean_date(date_str):
    if isinstance(date_str, str):
        parts = date_str.split('_')
        return parts[0]
    return date_str

def clean_columns(df):
    str_cols = [
        "Athlete_ID", "Athlete_Name", "Athlete_Country", "Gender",
        "Round", "Position", "Personal_Best", "Competition", "City", "Stadium"
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": None}).str.strip()
    for dc in ["Start_Date", "End_Date", "Date_of_Birth"]:
        if dc in df.columns:
            df[dc] = df[dc].apply(clean_date)
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
    return df

SAUDI_COLUMNS_DTYPE = {
    "Position": "numeric",
    "Result_numeric": "numeric",
    "Start_Date": "date",
    "End_Date": "date",
    "Date_of_Birth": "date"
}
MAJOR_COLUMNS_DTYPE = {
    "Position": "numeric",
    "Result_numeric": "numeric",
    "Start_Date": "date",
    "End_Date": "date",
    "Date_of_Birth": "date"
}

def coerce_dtypes(df, dtype_map):
    for col, ctype in dtype_map.items():
        if col not in df.columns:
            continue
        if ctype == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif ctype == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

###################################
# 4) Basic Data Cleaning + Parsing (Event Results)
###################################

# 1) Event type map (must be first)
event_type_map = {
    '60m': 'time', '100m': 'time', '200m': 'time', '400m': 'time', '800m': 'time',
    '1000m': 'time', '1500m': 'time', '3000m': 'time', '5000m': 'time', '10000m': 'time', 'Marathon': 'time',
    '60m Hurdles': 'time', '60m Hurdles (Women)': 'time', '60m Hurdles (Men)': 'time',
    '80m Hurdles': 'time', '100m Hurdles': 'time', '110m Hurdles': 'time',
    '110m Hurdles (Youth)': 'time', '100m Hurdles (Youth)': 'time', '400m Hurdles': 'time',
    '4x100m Relay': 'time', '4x400m Relay': 'time', '4x400m Mixed Relay': 'time',
    'Long Jump': 'distance', 'Triple Jump': 'distance', 'High Jump': 'distance', 'Pole Vault': 'distance',
    'Long Jump Indoor': 'distance', 'Triple Jump Indoor': 'distance', 'High Jump Indoor': 'distance', 'Pole Vault Indoor': 'distance',
    'Shot Put': 'distance', 'Discus Throw': 'distance', 'Hammer Throw': 'distance', 'Javelin Throw': 'distance',
    'Shot Put Indoor': 'distance',
    'Decathlon': 'points', 'Heptathlon': 'points', 'Pentathlon': 'points',
    'Heptathlon Indoor': 'points', 'Pentathlon Indoor': 'points'
}

# 2) Helper to list them
def get_event_list():
    return sorted(event_type_map.keys())

# 3) Normalize any relay‐style raw names into your canonical keys
def normalize_relay_events(df):
    if 'Event' not in df.columns:
        return df

    # lowercase for matching
    df['Event'] = df['Event'].astype(str).str.strip().str.lower()

    relay_map = {
        r'4\s*[x×*]\s*100':      '4x100m Relay',
        r'4\s*[x×*]\s*400\s*mixed': '4x400m Mixed Relay',
        r'4\s*[x×*]\s*400':      '4x400m Relay',
    }

    for pattern, standard in relay_map.items():
        mask = df['Event'].str.contains(pattern, regex=True, na=False)
        df.loc[mask, 'Event'] = standard

    return df

# 4) Robust parser that strips commas/letters and handles h:mm:ss, m:ss, plain floats
import re

def parse_result(value, event):
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().upper()

    # keep only digits, colon, and dot
    cleaned = re.sub(r"[^\d:\.]", "", raw)
    # drop stray A/H tags
    cleaned = cleaned.replace("A", "").replace("H", "").strip()
    if cleaned in {"", "DNF", "DNS", "DQ", "NM"}:
        return None

    # look up event type
    evt_key = event.strip().replace("Indoor", "").strip()
    e_type = event_type_map.get(event,
             event_type_map.get(evt_key, "other"))

    # fallback: if still 'other' but contains a colon, treat as time
    if e_type == "other" and ":" in cleaned:
        e_type = "time"

    try:
        if e_type == "time":
            parts = cleaned.split(":")
            if len(parts) == 3:
                h, m, s = parts
                return float(h)*3600 + float(m)*60 + float(s)
            if len(parts) == 2:
                m, s = parts
                return float(m)*60 + float(s)
            return float(parts[0])
        if e_type in {"distance", "points"}:
            return float(cleaned)
        # final fallback
        return float(cleaned)
    except:
        return None

# 5) Loader: cleans, normalizes, strips commas from Event, then parses
@st.cache_data
def load_db(db_filename: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path  = os.path.join(base_dir, "SQL", db_filename)
    if not os.path.exists(db_path):
        st.warning(f"{db_filename} not found in 'SQL' folder.")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    df   = pd.read_sql_query("SELECT * FROM athletics_data", conn)
    conn.close()

    # your existing cleaning
    df = clean_columns(df)

    # normalize relay names
    df = normalize_relay_events(df)

    # strip commas so "10,000m" → "10000m"
    df['Event'] = df['Event'].astype(str).str.replace(',', '').str.strip()

    # parse every result
    if 'Result' in df.columns and 'Event' in df.columns:
        df['Result_numeric'] = df.apply(
            lambda r: parse_result(r['Result'], r['Event']), axis=1
        )

    # coerce Saudi vs Major DB types
    if db_filename == "saudi_athletes.db":
        df = coerce_dtypes(df, SAUDI_COLUMNS_DTYPE)
    elif db_filename == "major_championships.db":
        df = coerce_dtypes(df, MAJOR_COLUMNS_DTYPE)

    # derive Year if missing
    if 'Year' not in df.columns and 'Start_Date' in df.columns:
        df['Year'] = df['Start_Date'].dt.year

    return df


#conn = sqlite3.connect("SQL/major_championships.db")
#df_raw = pd.read_sql_query("SELECT Event FROM athletics_data", conn)
#conn.close()

#st.write("🧪 RAW Event values from DB (sample):", df_raw["Event"].dropna().unique())




# Load major DB and print debug info
df_major = load_db("major_championships.db")
df_major["Event_clean"] = df_major["Event"].str.strip().str.lower()
relay_events_clean = ["4 x 100m", "4 x 400m", "4 x 400m mixed relay"]
relay_rows = df_major[df_major["Event_clean"].isin(relay_events_clean)]
#st.write("Relay events sample:", relay_rows[["Event", "Start_Date", "Year"]].head(10))
#st.write("Unique Start_Date values:", df_major["Start_Date"].unique())
#st.write("Unique Result values:", df_major["Result"].unique())
#st.write("Unique Competition values:", df_major["Competition"].unique())
#st.write("Unique Round values:", df_major["Round"].unique())
#st.write("Cleaned Start_Date sample:", df_major["Start_Date"].head(10))
#st.write("Cleaned Year sample:", df_major["Year"].head(10))
#st.write("Result_numeric stats:", df_major["Result_numeric"].describe())
#df = load_db("major_championships.db")
#st.write("🚨 Blank or missing Events sample:", df[df["Event"].isnull() | (df["Event"].str.strip() == "")].head(10))
#st.write("🧪 RAW Event values from DB (sample):", df_raw["Event"].dropna().unique())

# 2. Assign country as Athlete_Name fallback for relays
df_major['Athlete_Name'] = df_major.apply(
    lambda row: row['Athlete_Name'] if pd.notna(row['Athlete_Name']) and row['Athlete_Name'].strip() != '' else row.get('Athlete_Country', 'Team'),
    axis=1
)







###################################
# 6) Athlete Expansions
###################################
def show_single_athlete_profile(profile, db_label):
    name = profile['Athlete_Name'].iloc[0] if 'Athlete_Name' in profile.columns else "Unknown"
    country = profile['Athlete_Country'].iloc[0] if 'Athlete_Country' in profile.columns else "N/A"
    dob = profile['Date_of_Birth'].iloc[0] if 'Date_of_Birth' in profile.columns else None
    def position_medal(pos):
        if pd.isna(pos):
            return ""
        try:
            p_ = int(pos)
            if p_ == 1:
                return "🥇"
            elif p_ == 2:
                return "🥈"
            elif p_ == 3:
                return "🥉"
        except:
            return ""
        return ""
    grouped = profile.copy()
    if 'Start_Date' in grouped.columns:
        grouped['Year'] = grouped['Start_Date'].dt.year.astype('Int64')
    if pd.notna(dob) and 'Start_Date' in grouped.columns:
        grouped['Age'] = ((grouped['Start_Date'] - dob).dt.days / 365.25).astype(int)
    else:
        grouped['Age'] = np.nan
    if 'Round' in grouped.columns:
        grouped['Round'] = grouped['Round'].fillna('Final')
        grouped['Round'] = grouped['Round'].replace({"": "Final", "None": "Final", "nan": "Final"})
    events_ = ", ".join(grouped['Event'].dropna().unique()) if 'Event' in grouped.columns else "N/A"
    with st.expander(f"{name} ({country})", expanded=False):
        st.subheader("Athlete Profile Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {name}")
            st.markdown(f"**Country:** {country}")
            if pd.notna(dob):
                st.markdown(f"**Date of Birth:** {dob.strftime('%Y-%m-%d')}")
            else:
                st.markdown("**Date of Birth:** N/A")
            st.markdown(f"**Events:** {events_}")
        with col2:
            if 'Result_numeric' in grouped.columns and not grouped['Result_numeric'].isna().all():
                best_idx = grouped['Result_numeric'].idxmin()
                best_row = grouped.loc[best_idx]
                st.markdown("**Personal Best:**")
                st.markdown(
                    f"{best_row.get('Event','N/A')} — {best_row.get('Result','N/A')} "
                    f"@ {best_row.get('Competition','N/A')} on {best_row.get('Start_Date','N/A')} "
                    f"in {best_row.get('Stadium','N/A')}"
                )
        st.markdown("### Notable Performances (Last 2 Years)")
        recent = grouped.copy()
        if 'Year' in recent.columns:
            cur_year = datetime.datetime.now().year
            recent = recent[recent['Year'] >= cur_year - 2]
        if 'Result_numeric' in recent.columns:
            recent = recent.sort_values('Result_numeric', ascending=True)
            recent['Highlight'] = (recent['Result_numeric'] == recent['Result_numeric'].min())
            recent['Highlight'] = recent['Highlight'].apply(lambda x: '🏅' if x else '')
        if 'Position' in recent.columns:
            recent['Medal'] = recent['Position'].apply(position_medal)
        show_cols = ['Result', 'Event', 'Competition', 'Competition_ID', 'Stadium', 'Start_Date', 'Round', 'Position', 'Medal', 'Age', 'Highlight']
        st.dataframe(style_dark_df(ensure_json_safe(recent[[c for c in show_cols if c in recent.columns]].head(5))))
        st.markdown("### Performance Progression Chart")
        if 'Result_numeric' in grouped.columns and 'Event' in grouped.columns:
            for ev_ in grouped['Event'].dropna().unique():
                sub_ev = grouped[(grouped['Event'] == ev_) & grouped['Result_numeric'].notna()]
                if sub_ev.empty:
                    continue
                q1 = sub_ev['Result_numeric'].quantile(0.25)
                q3 = sub_ev['Result_numeric'].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                sub_ev_filtered = sub_ev[(sub_ev['Result_numeric'] >= lower) & (sub_ev['Result_numeric'] <= upper)]
                y_min = sub_ev_filtered['Result_numeric'].min()
                y_max = sub_ev_filtered['Result_numeric'].max()
                y_pad = (y_max - y_min) * 0.1 if y_max > y_min else 1
                y_axis = alt.Y('Result_numeric:Q', title='Performance', scale=alt.Scale(domain=[y_min - y_pad, y_max + y_pad]))
                chart = alt.Chart(sub_ev_filtered).mark_line(
                    interpolate='monotone',
                    point=alt.OverlayMarkDef(filled=True, size=60)
                ).encode(
                    x=alt.X('Start_Date:T', title='Date'),
                    y=y_axis,
                    tooltip=['Start_Date:T', 'Event', 'Result', 'Competition', 'Round', 'Position', 'Age'],
                    color=alt.value('#00FF7F')
                ).properties(
                    title=f"{ev_} Progression",
                    width=800,
                    height=300
                ).configure_axis(
                    labelColor='white',
                    titleColor='white',
                    labelFontSize=12,
                    titleFontSize=14,
                    gridColor='gray',
                    domainColor='white'
                ).configure_view(
                    strokeWidth=0,
                    fill='black'
                ).configure_title(
                    color='white',
                    fontSize=16
                )
                st.altair_chart(chart, use_container_width=True)
        st.markdown("### 🗕️ Current Season Results")
        if 'Year' in grouped.columns:
            cyr = datetime.datetime.now().year
            cseason = grouped[grouped['Year'] == cyr].sort_values('Start_Date', ascending=False)
            if cseason.empty:
                st.warning("No results found for the current season.")
            else:
                cseason['Season_Best'] = (cseason['Result_numeric'] == cseason['Result_numeric'].min())
                cseason['Season_Best'] = cseason['Season_Best'].apply(lambda x: '🌟' if x else '')
                show_cols2 = ['Start_Date', 'Event', 'Result', 'Season_Best', 'Competition', 'Round', 'Position']
                st.dataframe(style_dark_df(ensure_json_safe(cseason[[c for c in show_cols2 if c in cseason.columns]])))
        def recent_avg(dfx, year_):
            sub_df = dfx[dfx['Year'] == year_].sort_values('Start_Date', ascending=False)
            avg_ = sub_df['Result_numeric'].mean()
            sub_df['Season_Best'] = (sub_df['Result_numeric'] == sub_df['Result_numeric'].min())
            sub_df['Season_Best'] = sub_df['Season_Best'].apply(lambda x: '🌟' if x else '')
            return sub_df, avg_
        st.markdown("### Seasonal Averages (Last 3 Results)")
        if 'Year' in grouped.columns:
            cyr = datetime.datetime.now().year
            this_season_df, this_avg = recent_avg(grouped, cyr)
            last_season_df, last_avg = recent_avg(grouped, cyr - 1)
            if len(this_season_df) < 3:
                st.info("🔄 Early season: fewer than 3 results.")
            if not this_season_df.empty:
                valid_cnt = this_season_df['Result_numeric'].notna().sum()
                if valid_cnt == 0:
                    st.warning("📬 This season has results, but none numeric for averaging.")
                elif np.isnan(this_avg):
                    st.markdown("**This Season Avg:** Not available (missing numeric results)")
                else:
                    st.markdown(f"**This Season Avg (Last 3):** {this_avg:.2f}")
            else:
                st.markdown("No data for this season")
            st.dataframe(style_dark_df(ensure_json_safe(
                this_season_df[[c for c in ['Start_Date', 'Event', 'Result', 'Season_Best', 'Competition'] if c in this_season_df.columns]]
            )))
            if not np.isnan(last_avg):
                st.markdown(f"**Last Season Avg (Last 3):** {last_avg:.2f}")
            else:
                st.markdown("No data for last season")
            st.dataframe(style_dark_df(ensure_json_safe(
                last_season_df[[c for c in ['Start_Date', 'Event', 'Result', 'Season_Best', 'Competition'] if c in last_season_df.columns]]
            )))
        st.markdown("### Top Results")
        if 'Result_numeric' in grouped.columns:
            top10 = grouped.dropna(subset=['Result_numeric']).sort_values('Result_numeric').head(10)
            top10['PB'] = (top10['Result_numeric'] == top10['Result_numeric'].min())
            top10['PB'] = top10['PB'].apply(lambda x: '🏅' if x else '')
            if 'Position' in top10.columns:
                top10['Medal'] = top10['Position'].apply(position_medal)
            cshow = ['Result', 'Is_Hand_Timed', 'Is_Altitude', 'PB', 'Competition', 'Competition_ID', 'Stadium', 'Round', 'Position', 'Medal', 'Age', 'Start_Date']
            st.dataframe(style_dark_df(ensure_json_safe(
                top10[[c for c in cshow if c in top10.columns]]
            )))

###################################
# 7) Athlete Profiles Container
###################################
def show_athlete_profiles(filtered_df, db_label):
    st.subheader(f"{db_label}: Athlete Profiles")
    if 'Athlete_Name' not in filtered_df.columns:
        st.warning("No 'Athlete_Name' column in data.")
        return
    names_ = [n.strip() for n in filtered_df['Athlete_Name'].dropna().unique()]
    default_name = ["Abdulaziz Abdou Atafi"] if "Abdulaziz Abdou Atafi" in names_ else names_[:1]
    chosen_names = st.multiselect(
    f"{db_label} Athlete(s)",
    names_,
    default=default_name,
    key=f"{db_label}_athlete"
)

    for athlete_name in chosen_names:
        profile = filtered_df[filtered_df['Athlete_Name'] == athlete_name]
        if profile.empty:
            continue
        show_single_athlete_profile(profile, db_label)

###################################
# 8) Qualification Stage
###################################
def get_flag(country_code):
    if not isinstance(country_code, str) or len(country_code) != 3:
        return ""
    offset = 127397
    try:
        return ''.join([chr(ord(c.upper()) + offset) for c in country_code[:2]])
    except:
        return ""

import math
import pandas as pd
import altair as alt
import streamlit as st

# … (other imports and helper functions above) …

def show_qualification_stage(df):
    st.subheader("Qualification Stage")
    # 1) Keep only valid rounds
    df = df[df['Round'].notna()]
    df = df[~df['Round'].isin(["None", "", "nan", None])]

    # 2) Normalize round names
    round_clean_map = {
        "Preliminary round": "Prelims",
        "Preliminary": "Prelims",
        "Qualification": "Heats",
        "Qualifying": "Heats",
        "Heats": "Heats",
        "Quarterfinals": "QF",
        "Semifinals": "SF",
        "Final": "Final",
        "F": "Final"
    }
    df['Round'] = df['Round'].map(round_clean_map).fillna(df['Round'])

    # 3) Fallback for relay teams
    if 'Athlete_Name' in df.columns and 'Athlete_Country' in df.columns:
        df['Athlete_Name'] = df.apply(
            lambda row: row['Athlete_Name']
                if pd.notna(row['Athlete_Name']) and row['Athlete_Name'].strip() != ''
                else row.get('Athlete_Country', 'Team'),
            axis=1
        )

    # 4) Add country flags
    if 'Athlete_Country' in df.columns:
        df['Country_Flag'] = df['Athlete_Country'].apply(get_flag)
        df['Athlete_Country'] = df['Country_Flag'] + ' ' + df['Athlete_Country']

    # 5) Outlier removal helper
    def remove_outliers(df_in, field='Result_numeric'):
        q1 = df_in[field].quantile(0.25)
        q3 = df_in[field].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return df_in[(df_in[field] >= lower) & (df_in[field] <= upper)]

    # 6) If we have numeric results, proceed
    if {'Round', 'Result_numeric', 'Year'}.issubset(df.columns):
        # keep only numeric
        df_num = df[df['Result_numeric'].notna()].copy()

        # remove outliers
        df_filtered = remove_outliers(df_num, 'Result_numeric')
        removed = df_num[~df_num.index.isin(df_filtered.index)]
        if not removed.empty:
            with st.expander("📛 Removed Outliers (IQR method)", expanded=False):
                st.dataframe(
                    style_dark_df(
                        ensure_json_safe(
                            removed[['Event', 'Round', 'Year', 'Result', 'Result_numeric']]
                        )
                    )
                )

        # aggregate stats
        round_year_stats = (
            df_filtered
            .groupby(['Round', 'Year'], as_index=False)
            .agg(Avg=('Result_numeric', 'mean'),
                 Min=('Result_numeric', 'min'),
                 Max=('Result_numeric', 'max'))
        )

        # qualifier lines (fastest_q = 2nd best, slowest_q = 8th best)
        def get_qualifier_stats(sub):
            s = sub.sort_values('Result_numeric')['Result_numeric'].dropna()
            return pd.Series({
                'Fastest_Q': s.iloc[1] if len(s) > 1 else pd.NA,
                'Slowest_Q': s.iloc[7] if len(s) > 7 else pd.NA
            })

        qualifier_stats = (
            df_filtered
            .groupby(['Round', 'Year'], as_index=False)
            .apply(get_qualifier_stats)
            .reset_index(drop=True)
        )

        # merge full stats
        full_stats = pd.merge(
            round_year_stats,
            qualifier_stats,
            on=['Round', 'Year'],
            how='outer'
        )

        st.markdown("**Min / Avg / Max / Fastest Q / Slowest Q by Round & Year**")
        st.dataframe(style_dark_df(ensure_json_safe(full_stats)))

        # 7) Faceted line chart
        melted = full_stats.melt(
            id_vars=['Round', 'Year'],
            var_name='Metric',
            value_name='Value'
        )
        order = ["Prelims", "Heats", "QF", "SF", "Final"]
        melted['Round'] = pd.Categorical(melted['Round'], categories=order, ordered=True)

        y_min, y_max = melted['Value'].min(skipna=True), melted['Value'].max(skipna=True)
        if math.isnan(y_min) or math.isnan(y_max):
            st.info("Insufficient numeric data available for charting qualification performance.")
        else:
            pad = (y_max - y_min) * 0.1 if y_max > y_min else 1
            y_axis = alt.Y('Value:Q', scale=alt.Scale(domain=[y_min - pad, y_max + pad]))

            qualifier_lines = ['Fastest_Q', 'Slowest_Q']
            chart = (
                alt.Chart(melted)
                .mark_line(interpolate='monotone', point=alt.OverlayMarkDef(filled=True, size=60))
                .encode(
                    x=alt.X('Year:O', title='Year'),
                    y=y_axis,
                    color=alt.Color('Round:N', sort=order, scale=alt.Scale(scheme='dark2')),
                    strokeDash=alt.condition(
                        alt.FieldOneOfPredicate(field='Metric', oneOf=qualifier_lines),
                        alt.value([4,4]),
                        alt.value([1])
                    ),
                    tooltip=['Year','Round','Metric','Value']
                )
                .properties(width=220, height=250)
                .facet(facet='Metric:N', columns=3)
                .resolve_scale(y='shared')
                .configure_axis(labelColor='white', titleColor='white', gridColor='gray')
                .configure_view(strokeWidth=0, fill='black')
                .configure_title(color='white', fontSize=16)
            )
            st.markdown("### Rounds Over Years (Min/Avg/Max + Qualifier Lines)")
            st.altair_chart(chart, use_container_width=True)

    else:
        st.info("Need 'Round', 'Result_numeric', and 'Year' columns for qualification chart.")
        st.dataframe(style_dark_df(ensure_json_safe(df.head(10))))


def show_final_performances(df):
    st.subheader("Final Performances")
    # 1) Ensure final round
    if 'Round' in df.columns:
        df['Round'] = df['Round'].fillna("Final").replace({"F":"Final","None":"Final","": "Final"})
        df = df[df['Round'] == "Final"]

    # 2) Must have numeric & position
    if 'Position' not in df.columns or 'Result_numeric' not in df.columns:
        st.info("Need 'Position' & 'Result_numeric' for final performances.")
        st.dataframe(style_dark_df(ensure_json_safe(df.head(10))))
        return

    # 3) Assign medals
    df = df.copy()
    df['Medal'] = df['Position'].apply(lambda p: "🥇" if p==1 else ("🥈" if p==2 else ("🥉" if p==3 else "")))

    # 4) Add flags
    if 'Athlete_Country' in df.columns:
        df['Country_Flag'] = df['Athlete_Country'].apply(get_flag)
        df['Athlete_Country'] = df['Country_Flag'] + ' ' + df['Athlete_Country']

    # 5) Top-12 preview
    final_12 = df[df['Position'].between(1,12)]
    if final_12.empty:
        st.info("No final round results in positions 1–12.")
        return

    # 6) Outlier removal for top-8 chart
    top8 = final_12[final_12['Position'] <= 8].copy()
    top8_filtered = top8[top8['Result_numeric'].notna()]
    # outlier removal
    q1, q3 = top8_filtered['Result_numeric'].quantile(0.25), top8_filtered['Result_numeric'].quantile(0.75)
    iqr = q3 - q1
    top8_filtered = top8_filtered[(top8_filtered['Result_numeric'] >= (q1 - 1.5*iqr)) & (top8_filtered['Result_numeric'] <= (q3 + 1.5*iqr))]

    # 7) Chart if valid
    if 'Year' in top8_filtered.columns:
        y_min, y_max = top8_filtered['Result_numeric'].min(skipna=True), top8_filtered['Result_numeric'].max(skipna=True)
        if math.isnan(y_min) or math.isnan(y_max):
            st.info("Insufficient numeric data available for charting final top‑8 performance.")
        else:
            pad = (y_max - y_min)*0.1 if y_max > y_min else 1
            y_axis = alt.Y('Result_numeric:Q', scale=alt.Scale(domain=[y_min - pad, y_max + pad]), title='Performance')
            chart = (
                alt.Chart(top8_filtered)
                .mark_line(interpolate='monotone', point=alt.OverlayMarkDef(filled=True, size=60))
                .encode(
                    x=alt.X('Year:O', title='Year'),
                    y=y_axis,
                    color=alt.Color('Position:N', scale=alt.Scale(scheme='tableau10')),
                    tooltip=['Year','Position','Medal','Athlete_Name','Result','Competition']
                )
                .properties(width=800, height=400)
                .configure_axis(labelColor='white', titleColor='white', gridColor='gray')
                .configure_view(strokeWidth=0, fill='black')
                .configure_title(color='white', fontSize=18)
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No 'Year' column available for final top‑8 chart.")



###################################
# 9) Final Performances
###################################
def show_final_performances(df):
    st.subheader("Final Performances")
    if 'Round' in df.columns:
        df['Round'] = df['Round'].fillna("Final")
        df['Round'] = df['Round'].replace({"F": "Final", "None": "Final", "": "Final"})
        df = df[df['Round'] == "Final"]
    if 'Position' not in df.columns or 'Result_numeric' not in df.columns:
        st.info("Need 'Position' & 'Result_numeric' for final performances.")
        st.dataframe(style_dark_df(ensure_json_safe(df.head(10))))
        return
    def medal_emoji(pos):
        if pd.isna(pos):
            return ""
        try:
            pos = int(pos)
            if pos == 1:
                return "🥇"
            elif pos == 2:
                return "🥈"
            elif pos == 3:
                return "🥉"
        except:
            return ""
        return ""
    df = df.copy()
    df['Medal'] = df['Position'].apply(medal_emoji)
    if 'Athlete_Country' in df.columns:
        df['Country_Flag'] = df['Athlete_Country'].apply(get_flag)
        df['Athlete_Country'] = df['Country_Flag'] + ' ' + df['Athlete_Country']
    final_12 = df[df['Position'].between(1, 12, inclusive='both')].copy()
    st.write("Final round preview (before year conversion):", 
             final_12[['Event', 'Start_Date', 'Result', 'Result_numeric']].head(10))
    if final_12.empty:
        st.info("No final round results in positions 1–12.")
        return
    relay_events = ['4x100m Relay', '4x400m Relay', '4x400m Mixed Relay']
    current_event = final_12['Event'].iloc[0] if 'Event' in final_12.columns else None
    if current_event in relay_events:
        st.info("Relay event detected. Displaying results without outlier removal.")
        top8_filtered = final_12[final_12['Result_numeric'].notna()].copy()
    else:
        def remove_outliers(df_inner, field='Result_numeric'):
            q1 = df_inner[field].quantile(0.25)
            q3 = df_inner[field].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return df_inner[(df_inner[field] >= lower) & (df_inner[field] <= upper)]
        top8 = final_12[final_12['Position'] <= 8]
        top8_filtered = top8[top8['Result_numeric'].notna()].copy()
        top8_filtered = remove_outliers(top8_filtered)
        removed_outliers = top8[~top8.index.isin(top8_filtered.index)]
        if not removed_outliers.empty:
            with st.expander("📛 Removed Outliers (IQR method)", expanded=False):
                st.dataframe(style_dark_df(ensure_json_safe(
                    removed_outliers[['Event', 'Year', 'Result', 'Result_numeric']]
                )))
    import math
    if 'Year' not in top8_filtered.columns:
        if 'Start_Date' in top8_filtered.columns:
            top8_filtered['Year'] = pd.to_datetime(top8_filtered['Start_Date'], errors='coerce').dt.year
        else:
            st.info("No 'Year' or 'Start_Date' column available; cannot chart Final round top 8.")
            return
    if top8_filtered['Year'].isnull().all():
        st.info("Final round top 8 data do not contain valid year information. Attempting fallback extraction from 'Competition'.")
        def fallback_year(row):
            competition = row['Competition'] if 'Competition' in row and pd.notnull(row['Competition']) else ''
            match = re.search(r'(\d{4})', competition)
            if match:
                return int(match.group(1))
            return None
        top8_filtered = top8_filtered.reset_index(drop=True)
        fallback_series = top8_filtered.apply(lambda row: fallback_year(row), axis=1)
        top8_filtered['Year'] = fallback_series
        st.write("After fallback extraction, final round preview:", 
                 top8_filtered[['Event', 'Start_Date', 'Year', 'Result_numeric']].head(10))
        if top8_filtered['Year'].isnull().all():
            st.info("Fallback extraction did not yield any valid year values. Setting default year of 2021.")
            top8_filtered['Year'] = 2021
    st.write("Final round top 8 data preview:", top8_filtered[['Event', 'Start_Date', 'Year', 'Result_numeric']].head(10))
    if 'Year' in top8_filtered.columns and not top8_filtered.empty:
        y_min = top8_filtered['Result_numeric'].min()
        y_max = top8_filtered['Result_numeric'].max()
        if math.isnan(y_min) or math.isnan(y_max):
            st.info("Insufficient numeric data available for charting performance.")
        else:
            y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 1
            y_axis = alt.Y(
                'Result_numeric:Q',
                title='Performance',
                scale=alt.Scale(domain=[y_min - y_padding, y_max + y_padding])
            )
            chart = alt.Chart(top8_filtered).mark_line(
                interpolate='monotone',
                point=alt.OverlayMarkDef(filled=True, size=60)
            ).encode(
                x=alt.X('Year:O', title='Year'),
                y=y_axis,
                color=alt.Color('Position:N', scale=alt.Scale(scheme='tableau10')),
                tooltip=['Year', 'Position', 'Medal', 'Athlete_Name', 'Athlete_Country', 'Event', 'Result', 'Competition']
            ).properties(
                title="Top 8 Finalists Over Years",
                width=800,
                height=400
            ).configure_axis(
                labelColor='white',
                titleColor='white',
                labelFontSize=12,
                titleFontSize=14,
                gridColor='gray',
                domainColor='white'
            ).configure_view(
                strokeWidth=0,
                fill='black'
            ).configure_title(
                color='white',
                fontSize=18
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No 'Year' column or data available for charting Final round top 8.")
###################################
# 9) Relay Chart
###################################

def show_relay_charts(df):
    st.subheader("Relay Event Analysis")

    # Normalize relay names
    df = normalize_relay_events(df)

    # Filter for relay events only
    relay_df = df[df['Event'].str.contains('relay', case=False, na=False)].copy()
    if relay_df.empty:
        st.info("No relay event data found.")
        return

    # Normalize event column for filtering
    relay_df['Event_clean'] = relay_df['Event'].str.strip().str.lower()

    # Build clean master list
    relay_events_master = [e for e in event_type_map if 'relay' in e.lower()]
    relay_events_master_normalized = [e.lower() for e in relay_events_master]
    event_display_map = {e.lower(): e for e in relay_events_master}

    # Optional filters
    col1, col2, col3 = st.columns(3)
    with col1:
        gender_opts = sorted(relay_df['Gender'].dropna().unique())
        chosen_gender = st.selectbox("Gender", ["All"] + gender_opts, index=0, key="relay_gender")
        if chosen_gender != "All":
            relay_df = relay_df[relay_df['Gender'] == chosen_gender]

    with col2:
        chosen_events = st.multiselect(
            "Relay Events",
            relay_events_master,
            default=relay_events_master,
            key="relay_event_filter"
        )
        chosen_events_normalized = [e.lower() for e in chosen_events]
        relay_df = relay_df[relay_df['Event_clean'].isin(chosen_events_normalized)]

    with col3:
        year_opts = sorted(relay_df['Year'].dropna().unique())
        chosen_years = st.multiselect("Years", year_opts, default=year_opts, key="relay_year_filter")
        if chosen_years:
            relay_df = relay_df[relay_df['Year'].isin(chosen_years)]

    if relay_df.empty:
        st.warning("No data after applying filters.")
        return

    # Re-map event names to display-friendly format
    relay_df['Event'] = relay_df['Event_clean'].map(event_display_map).fillna(relay_df['Event'])

    # Drop missing or outlier results
    relay_df = relay_df[relay_df['Result_numeric'].notna()]
    q1 = relay_df['Result_numeric'].quantile(0.25)
    q3 = relay_df['Result_numeric'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    relay_df = relay_df[(relay_df['Result_numeric'] >= lower) & (relay_df['Result_numeric'] <= upper)]

    # Relay progression faceted by country
    st.markdown("### 📈 Relay Progression by Country")
    base_chart = alt.Chart(relay_df).mark_line(
        interpolate='monotone',
        point=alt.OverlayMarkDef(filled=True, size=50)
    ).encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Result_numeric:Q', title='Time (s)'),
        color=alt.Color('Event:N', title='Event'),
        tooltip=['Year', 'Event', 'Result', 'Athlete_Country', 'Competition']
    ).properties(
        width=250,
        height=200
    )

    line_chart = base_chart.facet(
        facet=alt.Facet('Athlete_Country:N', title='Country'),
        columns=3
    ).configure_axis(
        labelColor='white',
        titleColor='white',
        gridColor='gray'
    ).configure_view(
        strokeWidth=0,
        fill='black'
    ).configure_title(
        color='white'
    )

    st.altair_chart(line_chart, use_container_width=True)

    # Best results table
    st.markdown("### 🏅 Best Relay Performances per Country & Event")
    best_results = (
        relay_df.sort_values('Result_numeric')
        .groupby(['Athlete_Country', 'Event'], as_index=False)
        .first()
    )
    show_cols = ['Athlete_Country', 'Event', 'Result', 'Year', 'Competition']
    st.dataframe(style_dark_df(ensure_json_safe(best_results[show_cols])))


###################################
# 10) Main
###################################
def main():
    tab_names = [
    "Athlete Profiles (Saudi DB)",
    "Athlete Profiles (Major DB)",
    "Major Championships: Qualification vs Final",
    "Relay Analytics"  # 🆕 New tab
]

    tabs = st.tabs(tab_names)
    with tabs[0]:
        st.title("Athlete Profiles – Saudi DB")
        df_saudi = load_db("saudi_athletes.db")
        if df_saudi.empty:
            st.warning("No data from 'saudi_athletes.db'")
        else:
            st.header("Filters – Saudi Athletes")
            filtered_df = df_saudi.copy()
            if 'Gender' in filtered_df.columns:
                gender_options = sorted(filtered_df['Gender'].dropna().unique())
                chosen_gender = st.selectbox("Saudi Gender", ["All"] + gender_options, key="saudi_gender")
                if chosen_gender != "All":
                    filtered_df = filtered_df[filtered_df['Gender'] == chosen_gender]
            if 'Event' in filtered_df.columns:
                event_options = sorted(filtered_df['Event'].dropna().unique())
                default_event = ["100m"] if "100m" in event_options else event_options[:1]
                chosen_events = st.multiselect("Saudi Events", event_options, default=default_event, key="saudi_events")
            if chosen_events:
                filtered_df = filtered_df[filtered_df['Event'].isin(chosen_events)]

            if 'Year' in filtered_df.columns:
                year_options = sorted(filtered_df['Year'].dropna().unique())
                chosen_years = st.multiselect("Saudi Years", year_options, default=year_options, key="saudi_years")
                if chosen_years:
                    filtered_df = filtered_df[filtered_df['Year'].isin(chosen_years)]
            show_athlete_profiles(filtered_df, "Saudi DB")
    with tabs[1]:
        st.title("Athlete Profiles – Major DB")
        df_major = load_db("major_championships.db")
        if df_major.empty:
            st.warning("No data from 'major_championships.db'")
        else:
            st.header("Filters – Major Championships")
            filtered_df = df_major.copy()
            if 'Gender' in filtered_df.columns:
                gender_options = sorted(filtered_df['Gender'].dropna().unique())
                chosen_gender = st.selectbox("Major Gender", ["All"] + gender_options, key="major_gender")
                if chosen_gender != "All":
                    filtered_df = filtered_df[filtered_df['Gender'] == chosen_gender]
            if 'Event' in filtered_df.columns:
                event_options = sorted(filtered_df['Event'].dropna().unique())
                chosen_events = st.multiselect("Major Events", event_options, default=event_options[:1], key="major_events")
                if chosen_events:
                    filtered_df = filtered_df[filtered_df['Event'].isin(chosen_events)]
            if 'Year' in filtered_df.columns:
                year_options = sorted(filtered_df['Year'].dropna().unique())
                chosen_years = st.multiselect("Major Years", year_options, default=year_options, key="major_years")
                if chosen_years:
                    filtered_df = filtered_df[filtered_df['Year'].isin(chosen_years)]
            if 'Athlete_Country' in filtered_df.columns:
                country_options = sorted(filtered_df['Athlete_Country'].dropna().unique())
                chosen_country = st.selectbox("Major Country", ["All"] + country_options, key="major_country")
                if chosen_country != "All":
                    filtered_df = filtered_df[filtered_df['Athlete_Country'] == chosen_country]
            st.markdown("#### Filter by Championship")
            comp_names = ["All"] + sorted(MAJOR_COMPETITIONS_CID.keys())
            chosen_comp = st.selectbox("Major Championship", comp_names, key="major_comp_name")
            if chosen_comp != "All":
                edition_years = list(MAJOR_COMPETITIONS_CID[chosen_comp].keys())
                chosen_edition_year = st.selectbox("Edition Year", ["All"] + sorted(edition_years, reverse=True), key="major_comp_year")
                if chosen_edition_year != "All":
                    cid = MAJOR_COMPETITIONS_CID[chosen_comp][chosen_edition_year]["CID"]
                    if "Competition_ID" in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df["Competition_ID"] == cid]
            show_athlete_profiles(filtered_df, "Major DB")
    with tabs[2]:
        st.title("Major Championships: Qualification vs Final")
        df_maj = load_db("major_championships.db")
        if df_maj.empty:
            st.warning("No data from 'major_championships.db'")
        else:
            if 'Competition_ID' in df_maj.columns:
                df_maj['Competition_ID'] = df_maj['Competition_ID'].astype(str)
            st.header("Filters: Qualification vs Final")
            if 'Gender' in df_maj.columns:
                g_opts = sorted(df_maj['Gender'].dropna().unique())
                default_gender = "Men" if "Men" in g_opts else g_opts[0]
                chosen_g = st.selectbox("Gender", g_opts, index=g_opts.index(default_gender), key="qualfinal_gender")
                df_maj = df_maj[df_maj['Gender'] == chosen_g]
            if 'Event' in df_maj.columns:
                e_opts = sorted(df_maj['Event'].dropna().unique())
                default_event = [e for e in ['100m'] if e in e_opts] or e_opts[:1]
                chosen_e = st.multiselect("Event", e_opts, default=default_event, key="qualfinal_events")
                if chosen_e:
                    df_maj = df_maj[df_maj['Event'].isin(chosen_e)]
            st.markdown("#### Filter by Championship")
            comp_names = sorted(MAJOR_COMPETITIONS_CID.keys())
            default_comp = "Olympics" if "Olympics" in comp_names else comp_names[0]
            chosen_comp = st.selectbox("Championship", ["All"] + comp_names, index=0, key="qualfinal_champ_name")
            if chosen_comp != "All":
                cids = [v["CID"] for v in MAJOR_COMPETITIONS_CID[chosen_comp].values()]
                if "Competition_ID" in df_maj.columns:
                    df_maj = df_maj[df_maj["Competition_ID"].isin(cids)]
                else:
                    st.warning("'Competition_ID' column not found in data.")
            if df_maj.empty:
                st.info("No data after filters.")
            else:
                sub_tabs = st.tabs(["Qualification Stage", "Final Performances"])
                with sub_tabs[0]:
                    show_qualification_stage(df_maj)
                with sub_tabs[1]:
                    show_final_performances(df_maj)

        with tabs[3]:
            st.title("Relay Event Analytics")
            df_maj_relay = load_db("major_championships.db")
            if df_maj_relay.empty:
                st.warning("No data from 'major_championships.db'")
            else:
                show_relay_charts(df_maj_relay)

if __name__ == "__main__":
    main()

# Footer
st.markdown("""
    <hr style='margin-top: 30px; border: 1px solid #444;'>
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
        Athletics Analysis Dashboard — Created by <strong>Luke Gallagher</strong>
    </div>
    """, unsafe_allow_html=True)
