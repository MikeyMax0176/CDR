# --- CGI/Cell Tower Schema Upload and Mapping ---
st.header("🗺️ Cell Tower Schema Upload & Call Mapping")
cell_file = st.file_uploader(
    "Upload cell tower schema (CSV/XLSX with CGI, latitude, longitude)",
    type=["csv", "xlsx"],
    key="cell_schema"
)

cell_df = None
if cell_file is not None:
    file_type = cell_file.name.split('.')[-1].lower()
    if file_type == "csv":
        cell_df = pd.read_csv(cell_file)
    elif file_type == "xlsx":
        cell_df = pd.read_excel(cell_file)
    else:
        st.error("Unsupported cell schema file type.")
        cell_df = None
    if cell_df is not None:
        st.write("Preview of cell tower schema:")
        st.dataframe(cell_df.head())

# --- Map Visualization ---
if cell_df is not None and 'cdr_data' in st.session_state:
    cdr_df = st.session_state['cdr_data']
    cgi_cols = [col for col in ['cgi', 'ecgi', 'nr-cgi'] if col in cdr_df.columns]
    cell_cgi_cols = [col for col in ['cgi', 'ecgi', 'nr-cgi', 'CGI', 'ECGI', 'NR-CGI'] if col in cell_df.columns]
    if cgi_cols and cell_cgi_cols:
        cgi_col = cgi_cols[0]
        cell_cgi_col = cell_cgi_cols[0]
        # Ensure both columns are string type for merge
        cdr_df[cgi_col] = cdr_df[cgi_col].astype(str)
        cell_df[cell_cgi_col] = cell_df[cell_cgi_col].astype(str)
        merged = pd.merge(cdr_df, cell_df, left_on=cgi_col, right_on=cell_cgi_col, how='inner')
        if not merged.empty and 'latitude' in merged.columns and 'longitude' in merged.columns:
            # Drop rows with missing lat/lon
            merged = merged.dropna(subset=['latitude', 'longitude'])
            if not merged.empty:
                lat_mean = merged['latitude'].mean()
                lon_mean = merged['longitude'].mean()
                if pd.notna(lat_mean) and pd.notna(lon_mean):
                    st.success(f"Mapped {len(merged)} calls to cell locations.")
                    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=10)
                    for _, row in merged.iterrows():
                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=4,
                            popup=f"A: {row.get('a_number', '')}<br>B: {row.get('b_number', '')}<br>Time: {row.get('start_time', '')}",
                            color='blue', fill=True, fill_opacity=0.6
                        ).add_to(m)
                    st_folium(m, width=700, height=500)
                else:
                    st.warning("No valid latitude/longitude values to plot after merging.")
            else:
                st.warning("No valid latitude/longitude values to plot after merging.")
        else:
            st.warning("No latitude/longitude columns found in merged data.")
    else:
        st.info("No matching CGI/ECGI/NR-CGI columns found for join.")

# --- Imports ---
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import folium
from streamlit_folium import st_folium
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
st.title("📞 CDR Analyzer")
st.write("Upload your Call Detail Record (CDR) files in CSV or Excel format.")

# --- File Upload: CDR ---
uploaded_file = st.file_uploader(
    "Choose a CDR file (CSV or XLSX)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_type == "xlsx":
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type.")
        df = None

    @st.cache_data(show_spinner=False)
    def standardize_cdr_columns(df):
        # Mapping of possible CDR column names to standard schema
        column_map = {
            'a_number': ['a_number', 'calling party number', 'msisdn-a', 'caller', 'calling', 'number', 'a_number_msisdn'],
            'b_number': ['b_number', 'called party number', 'msisdn-b', 'callee', 'called', 'number b', 'b_number_msisdn'],
            'imsi': ['imsi', 'subscriber id', 'subscriberid'],
            'imei': ['imei'],
            'start_time': ['start_time', 'call start time', 'timestamp', 'date', 'datetime', 'start'],
            'duration': ['duration', 'call duration', 'length', 'call end time', 'end time'],
            'call_type': ['call_type', 'type', 'service type', 'bearer type'],
            'direction': ['direction'],
            'disposition': ['disposition', 'termination code', 'call disposition'],
            'cell_id': ['cell id', 'cid', 'eci', 'ecgi', 'cellid'],
            'lac': ['lac', 'tac'],
            'mcc': ['mcc'],
            'mnc': ['mnc'],
            'cgi': ['cgi', 'ecgi', 'nr-cgi'],
            'roaming': ['roaming', 'roaming indicator'],
            'msc_id': ['msc id', 'switch', 'switch id'],
            'charge': ['charge', 'rate', 'tariff', 'cost'],
            'data_volume': ['data volume', 'volume', 'kb', 'mb', 'bytes'],
            'service_type': ['service type', 'bearer type'],
            'record_id': ['call reference id', 'record sequence number', 'record id', 'sequence number'],
        }

        # Normalize columns for matching: lowercase, remove spaces and underscores
        def normalize(col):

            # --- File Upload: CDR ---
            st.title("📞 CDR Analyzer")
            st.write("Upload your Call Detail Record (CDR) files in CSV or Excel format.")
            uploaded_file = st.file_uploader(
                "Choose a CDR file (CSV or XLSX)",
                type=["csv", "xlsx"]
            )

            @st.cache_data(show_spinner=False)
            def standardize_cdr_columns(df):
                # Mapping of possible CDR column names to standard schema
                column_map = {
                    'a_number': ['a_number', 'calling party number', 'msisdn-a', 'caller', 'calling', 'number', 'a_number_msisdn'],
                    'b_number': ['b_number', 'called party number', 'msisdn-b', 'callee', 'called', 'number b', 'b_number_msisdn'],
                    'imsi': ['imsi', 'subscriber id', 'subscriberid'],
                    'imei': ['imei'],
                    'start_time': ['start_time', 'call start time', 'timestamp', 'date', 'datetime', 'start'],
                    'duration': ['duration', 'call duration', 'length', 'call end time', 'end time'],
                    'call_type': ['call_type', 'type', 'service type', 'bearer type'],
                    'direction': ['direction'],
                    'disposition': ['disposition', 'termination code', 'call disposition'],
                    'cell_id': ['cell id', 'cid', 'eci', 'ecgi', 'cellid'],
                    'lac': ['lac', 'tac'],
                    'mcc': ['mcc'],
                    'mnc': ['mnc'],
                    'cgi': ['cgi', 'ecgi', 'nr-cgi'],
                    'roaming': ['roaming', 'roaming indicator'],
                    'msc_id': ['msc id', 'switch', 'switch id'],
                    'charge': ['charge', 'rate', 'tariff', 'cost'],
                    'data_volume': ['data volume', 'volume', 'kb', 'mb', 'bytes'],
                    'service_type': ['service type', 'bearer type'],
                    'record_id': ['call reference id', 'record sequence number', 'record id', 'sequence number'],
                }

                # Normalize columns for matching: lowercase, remove spaces and underscores
                def normalize(col):
                    return col.lower().replace(' ', '').replace('_', '')
                df_columns = {normalize(col): col for col in df.columns}
                standardized = {}
                for std_col, aliases in column_map.items():
                    found = None
                    for alias in aliases:
                        alias_norm = normalize(alias)
                        for col_norm, orig_col in df_columns.items():
                            if alias_norm == col_norm:
                                found = orig_col
                                break
                        if found:
                            break
                    if found:
                        standardized[std_col] = df[found]
                    else:
                        standardized[std_col] = None
                return pd.DataFrame(standardized)

            if uploaded_file is not None:
                file_type = uploaded_file.name.split('.')[-1].lower()
                if file_type == "csv":
                    df = pd.read_csv(uploaded_file)
                elif file_type == "xlsx":
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file type.")
                    df = None

                if df is not None:
                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())

                    std_df = standardize_cdr_columns(df)
                    st.write("\nStandardized CDR data (first 10 rows):")
                    st.dataframe(std_df.head(10))

                    # Store standardized data in session state for later analysis
                    st.session_state['cdr_data'] = std_df

                    # --- Basic Summary Statistics ---
                    st.header("📊 CDR Summary Statistics")
                    total_records = len(std_df)
                    total_duration = None
                    call_type_counts = std_df['call_type'].value_counts(dropna=True) if 'call_type' in std_df else None

                    # Handle duration: only sum numeric values
                    if 'duration' in std_df:
                        duration_numeric = pd.to_numeric(std_df['duration'], errors='coerce')
                        non_numeric_count = std_df['duration'].dropna().shape[0] - duration_numeric.dropna().shape[0]
                        total_duration = duration_numeric.dropna().sum()
                        if non_numeric_count > 0:
                            st.warning(f"{non_numeric_count} non-numeric duration values were ignored in the total duration calculation.")

                    st.markdown(f"**Total records:** {total_records}")
                    if total_duration is not None:
                        st.markdown(f"**Total duration (seconds):** {total_duration:.0f}")
                    if call_type_counts is not None:
                        st.markdown("**Call Type Counts:**")
                        st.dataframe(call_type_counts)

                    # --- Visualizations ---
                    st.header("📈 Visualizations")
                    # Bar chart: Call type distribution
                    if call_type_counts is not None and len(call_type_counts) > 0:
                        fig = px.bar(call_type_counts, x=call_type_counts.index, y=call_type_counts.values,
                                     labels={'x': 'Call Type', 'y': 'Count'}, title='Call Type Distribution')
                        st.plotly_chart(fig, use_container_width=True)

                    # Bar chart: Duration distribution (if available)
                    if 'duration' in std_df and std_df['duration'].notna().any():
                        fig2 = px.histogram(std_df, x='duration', nbins=30, title='Call Duration Distribution (seconds)')
                        st.plotly_chart(fig2, use_container_width=True)

                    # --- Call Relationship Network Visualization ---
                    st.header("🔗 Call Relationship Network")
                    if 'a_number' in std_df and 'b_number' in std_df:
                        rel_df = std_df[['a_number', 'b_number']].dropna()
                        rel_df = rel_df.head(500)
                        G = nx.from_pandas_edgelist(rel_df, 'a_number', 'b_number', create_using=nx.DiGraph())
                        net = Network(height="500px", width="100%", directed=True, notebook=False)
                        net.from_nx(G)
                        net.repulsion(node_distance=200, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.95)
                        net.show_buttons(filter_=['physics'])
                        net.save_graph("call_network.html")
                        with open("call_network.html", "r", encoding="utf-8") as f:
                            html = f.read()
                        components.html(html, height=550, scrolling=True)
                    else:
                        st.info("A- and B-number columns are required for relationship visualization.")
                            location=[row['latitude'], row['longitude']],
                            radius=4,
                            popup=f"A: {row.get('a_number', '')}<br>B: {row.get('b_number', '')}<br>Time: {row.get('start_time', '')}",
                            color='blue', fill=True, fill_opacity=0.6
                        ).add_to(m)
                    st_folium(m, width=700, height=500)
                else:
                    st.warning("No valid latitude/longitude values to plot after merging.")
            else:
                st.warning("No valid latitude/longitude values to plot after merging.")
        else:
            st.warning("No latitude/longitude columns found in merged data.")
    else:
        st.info("No matching CGI/ECGI/NR-CGI columns found for join.")

# --- AI Assistant Chat ---
st.header("🤖 AI Assistant")
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def ask_llm(prompt, model="llama2"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"[Error] LLM server returned status {response.status_code}"
    except Exception as e:
        return f"[Error] Could not connect to LLM server: {e}"

with st.form("ai_chat_form", clear_on_submit=True):
    user_input = st.text_area("Ask a question about your CDR data:", height=60)
    submitted = st.form_submit_button("Send")
    if submitted and user_input:
        st.session_state['chat_history'].append(("user", user_input))
        cdr_sample = ""
        if 'cdr_data' in st.session_state:
            cdr_sample = st.session_state['cdr_data'].head(5).to_csv(index=False)
        prompt = f"CDR data sample:\n{cdr_sample}\nUser: {user_input}\nAssistant:"
        ai_response = ask_llm(prompt)
        st.session_state['chat_history'].append(("assistant", ai_response))

if st.session_state['chat_history']:
    for role, msg in st.session_state['chat_history']:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")

st.title("📞 CDR Analyzer")
st.write("Upload your Call Detail Record (CDR) files in CSV or Excel format.")

uploaded_file = st.file_uploader(
    "Choose a CDR file (CSV or XLSX)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_type == "xlsx":
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type.")
        df = None


    @st.cache_data(show_spinner=False)
    def standardize_cdr_columns(df):
        # Mapping of possible CDR column names to standard schema
        column_map = {
            'a_number': ['a_number', 'calling party number', 'msisdn-a', 'caller', 'calling'],
            'b_number': ['b_number', 'called party number', 'msisdn-b', 'callee', 'called'],
            'imsi': ['imsi', 'subscriber id', 'subscriberid'],
            'imei': ['imei'],
            'start_time': ['start_time', 'call start time', 'timestamp', 'date', 'datetime', 'start'],
            'duration': ['duration', 'call duration', 'length', 'call end time', 'end time'],
            'call_type': ['call_type', 'type', 'service type', 'bearer type'],
            'direction': ['direction'],
            'disposition': ['disposition', 'termination code', 'call disposition'],
            'cell_id': ['cell id', 'cid', 'eci', 'ecgi', 'cellid'],
            'lac': ['lac', 'tac'],
            'mcc': ['mcc'],
            'mnc': ['mnc'],
            'cgi': ['cgi', 'ecgi', 'nr-cgi'],
            'roaming': ['roaming', 'roaming indicator'],
            'msc_id': ['msc id', 'switch', 'switch id'],
            'charge': ['charge', 'rate', 'tariff', 'cost'],
            'data_volume': ['data volume', 'volume', 'kb', 'mb', 'bytes'],
            'service_type': ['service type', 'bearer type'],
            'record_id': ['call reference id', 'record sequence number', 'record id', 'sequence number'],
        }

        # Lowercase columns for matching
        df_columns = {col.lower(): col for col in df.columns}
        standardized = {}
        for std_col, aliases in column_map.items():
            found = None
            for alias in aliases:
                alias = alias.lower()
                for col in df_columns:
                    if alias == col or alias.replace(' ', '') == col.replace(' ', ''):
                        found = df_columns[col]
                        break
                if found:
                    break
            if found:
                standardized[std_col] = df[found]
            else:
                standardized[std_col] = None
        return pd.DataFrame(standardized)

    if df is not None:
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        std_df = standardize_cdr_columns(df)
        st.write("\nStandardized CDR data (first 10 rows):")
        st.dataframe(std_df.head(10))

        # Store standardized data in session state for later analysis
        st.session_state['cdr_data'] = std_df

        # --- Basic Summary Statistics ---
        st.header("📊 CDR Summary Statistics")
        total_records = len(std_df)
        total_duration = std_df['duration'].dropna().astype(float).sum() if 'duration' in std_df else None
        call_type_counts = std_df['call_type'].value_counts(dropna=True) if 'call_type' in std_df else None

        st.markdown(f"**Total records:** {total_records}")
        if total_duration is not None:
            st.markdown(f"**Total duration (seconds):** {total_duration:.0f}")
        if call_type_counts is not None:
            st.markdown("**Call Type Counts:**")
            st.dataframe(call_type_counts)

        # --- Visualizations ---
        st.header("📈 Visualizations")
        # Bar chart: Call type distribution
        if call_type_counts is not None and len(call_type_counts) > 0:
            fig = px.bar(call_type_counts, x=call_type_counts.index, y=call_type_counts.values,
                         labels={'x': 'Call Type', 'y': 'Count'}, title='Call Type Distribution')
            st.plotly_chart(fig, use_container_width=True)

        # Bar chart: Duration distribution (if available)
        if 'duration' in std_df and std_df['duration'].notna().any():
            fig2 = px.histogram(std_df, x='duration', nbins=30, title='Call Duration Distribution (seconds)')
            st.plotly_chart(fig2, use_container_width=True)
