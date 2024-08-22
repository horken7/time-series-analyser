import pandas as pd
import streamlit as st
from darts import TimeSeries

@st.cache_data
def load_data(uploaded_file):
    """Load CSV data into a Pandas DataFrame."""
    return pd.read_csv(uploaded_file)


def select_columns(df):
    """Allow user to select timestamp and machine ID columns."""
    columns = df.columns.tolist()

    timestamp_col = st.selectbox(
        'Select Timestamp Column',
        columns,
        help="This is the column containing the timestamps for the time series."
    )

    machine_id_options = [col for col in columns if col != timestamp_col]
    machine_id_options.append(None)
    machine_id_col = st.selectbox(
        'Select Machine ID Column (optional)',
        machine_id_options,
        help="This column contains machine IDs. Leave as 'None' if not applicable."
    )

    return timestamp_col, machine_id_col


def select_machine_id(df, machine_id_col):
    """Allow user to select a specific machine ID (if applicable)."""
    if machine_id_col:
        machine_ids = df[machine_id_col].unique()
        selected_machine_id = st.selectbox(
            'Select Machine ID',
            list(machine_ids) + [None],
            help="Select a specific machine ID to filter the data, or 'None' to use all IDs."
        )
        return selected_machine_id
    return None

def filter_and_sort_data(df, timestamp_col, machine_id_col, selected_machine_id):
    """Filter the DataFrame by selected machine ID and sort by timestamp."""
    if selected_machine_id is not None and machine_id_col:
        df = df[df[machine_id_col] == selected_machine_id]

    return df.sort_values(by=timestamp_col)


def select_feature(df, timestamp_col, machine_id_col):
    """Allow user to select the feature column."""
    feature_options = [col for col in df.columns if col != timestamp_col and col != machine_id_col]
    selected_feature = st.selectbox(
        'Select Feature',
        feature_options,
        help="This is the column you want to analyze."
    )
    return selected_feature

def create_timeseries(df, timestamp_col, feature_col):
    """Create a TimeSeries object from the DataFrame."""
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    ts = TimeSeries.from_dataframe(df, time_col=timestamp_col, value_cols=feature_col)
    return ts
