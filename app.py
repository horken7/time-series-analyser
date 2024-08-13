import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest


# Function to detect anomalies using moving average and standard deviation
def detect_anomalies_moving_avg(series, window_size=5, sigma=2):
    rolling_mean = series.rolling(window=window_size).mean()
    rolling_std = series.rolling(window=window_size).std()
    anomalies = series[(series - rolling_mean).abs() > sigma * rolling_std]
    return anomalies.dropna()


# Function to detect anomalies using IsolationForest
def detect_anomalies_isolation_forest(series, contamination='auto'):
    original_index = series.index  # Store the original index
    series = series.values.reshape(-1, 1)
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(series)
    preds = model.predict(series)
    # Re-create the Series with the original index
    return pd.Series(series.flatten(), index=original_index)[preds == -1]


# Streamlit app title
st.title('Time Series Analyzer')

# Explanation text with example dataset reference
st.write("""
    **Welcome to the Time Series Analyzer!**

    This tool allows you to upload a CSV file containing time series data. You can then select a column 
    representing the timestamp, choose a feature to analyze, and apply anomaly detection methods 
    like Moving Average or Isolation Forest. The results will be visualized for easy interpretation.

    Please start by uploading your CSV file from the sidebar. An example dataset can be found [here](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance?select=PdM_maint.csv).
""")

# File upload section
st.sidebar.header('Upload your CSV file')
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV file
    df = pd.read_csv(uploaded_file)

    # List of columns
    columns = df.columns.tolist()

    # Prompt user to select the timestamp column
    st.sidebar.subheader('Select Timestamp Column')
    timestamp_col = st.sidebar.selectbox('Select Timestamp Column', columns)

    # Prepare options for machine ID column
    machine_id_options = [col for col in columns if col != timestamp_col]
    machine_id_options.append(None)  # Add "None" as the last option for machine ID column

    # Prompt user to select the machine id column
    st.sidebar.subheader('Select Machine ID Column (optional)')
    machine_id_col = st.sidebar.selectbox('Select Machine ID Column', machine_id_options)

    # Prepare options for specific machine ID selection
    if machine_id_col:
        machine_ids = df[machine_id_col].unique()
        machine_id_options = [None] + list(machine_ids)  # Add "None" at the beginning
    else:
        machine_id_options = [None]  # Only "None" if no machine ID column is selected

    # Prompt user to select the specific machine ID (if machine ID column is selected)
    st.sidebar.subheader('Select Machine ID')
    selected_machine_id = st.sidebar.selectbox('Select Machine ID', machine_id_options)

    if timestamp_col:
        # Filter data by selected machine ID
        if selected_machine_id is not None and machine_id_col:
            df = df[df[machine_id_col] == selected_machine_id]

        # Sort the DataFrame by the timestamp column
        df = df.sort_values(by=timestamp_col)

        # Convert to more memory-efficient types (optional)
        df = df.astype({col: 'float32' for col in df.columns if df[col].dtype == 'float64'})

        # Prepare options for feature column
        feature_options = [col for col in df.columns if col != timestamp_col and (col != machine_id_col)]

        # Sidebar options to select feature and method, and input for anomaly detection parameters
        st.sidebar.subheader('Anomaly Detection Parameters')
        selected_feature = st.sidebar.selectbox('Select Feature', feature_options)
        detection_method = st.sidebar.selectbox('Select Detection Method', ['Isolation Forest', 'Moving Average'])

        if detection_method == 'Moving Average':
            window_size = st.sidebar.number_input('Window Size', min_value=1, value=5)
            sigma = st.sidebar.number_input('Sigma Threshold', min_value=0.1, value=2.0, step=0.1)
        elif detection_method == 'Isolation Forest':
            contamination = st.sidebar.text_input(
                'Contamination (auto for automatic, or a float between 0.01 and 0.5)', 'auto')
            # Validate and convert contamination input
            try:
                contamination_value = float(contamination) if contamination.lower() != 'auto' else 'auto'
                if contamination_value != 'auto' and (contamination_value < 0.01 or contamination_value > 0.5):
                    st.error('Contamination must be between 0.01 and 0.5.')
                    contamination_value = 'auto'
            except ValueError:
                st.error('Invalid input for contamination. Please enter a float or "auto".')
                contamination_value = 'auto'

        # Plot the time series data and anomalies for the selected feature
        st.subheader(f'Time Series & Anomaly Detection for {selected_feature} using {detection_method}:')

        # Detect anomalies based on selected method
        if detection_method == 'Moving Average':
            anomalies = detect_anomalies_moving_avg(df[selected_feature], window_size=window_size, sigma=sigma)
        elif detection_method == 'Isolation Forest':
            anomalies = detect_anomalies_isolation_forest(df[selected_feature], contamination=contamination_value)

        # Plot data with anomalies
        fig = px.line(df, x=timestamp_col, y=selected_feature,
                      title=f'Time Series Data with Anomalies in {selected_feature}')

        # Add anomalies as red dots on the same figure
        if not anomalies.empty:
            fig.add_scatter(x=df[timestamp_col], y=anomalies, mode='markers', name='Anomalies',
                            marker=dict(color='red', size=8, line=dict(color='red', width=2)))

        st.plotly_chart(fig)

        # Display the detected anomalies data
        if not anomalies.empty:
            st.subheader('Detected Anomalies:')
            st.write(anomalies)
        else:
            st.write('No anomalies detected.')
