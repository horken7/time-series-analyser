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
def detect_anomalies_isolation_forest(series):
    original_index = series.index  # Store the original index
    series = series.values.reshape(-1, 1)
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(series)
    preds = model.predict(series)
    # Re-create the Series with the original index
    return pd.Series(series.flatten(), index=original_index)[preds == -1]


# Function to downsample the data if it's too large
def downsample_data(df, max_points=10000):
    if len(df) > max_points:
        return df.iloc[::len(df) // max_points]
    return df


# Streamlit app title
st.title('Time Series Analyzer')

# Explanation text
st.write("""
    **Welcome to the Time Series Analyzer!**

    This tool allows you to upload a CSV file containing time series data. You can then select a column 
    representing the timestamp, choose a feature to analyze, and apply anomaly detection methods 
    like Moving Average or Isolation Forest. The results will be visualized for easy interpretation.

    Please start by uploading your CSV file from the sidebar.
""")

# File upload section
st.sidebar.header('Upload your CSV file')
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV file
    df = pd.read_csv(uploaded_file)

    # Prompt user to select the timestamp column
    st.sidebar.subheader('Select Timestamp Column')
    timestamp_col = st.sidebar.selectbox('Select Timestamp Column', df.columns)

    if timestamp_col:
        # Downsample the data if it's too large
        df = downsample_data(df)

        # Convert to more memory-efficient types (optional)
        df = df.astype({col: 'float32' for col in df.columns if df[col].dtype == 'float64'})

        # Sidebar options to select feature and method, and input for anomaly detection parameters
        st.sidebar.subheader('Anomaly Detection Parameters')
        selected_feature = st.sidebar.selectbox('Select Feature', [col for col in df.columns if col != timestamp_col])
        detection_method = st.sidebar.selectbox('Select Detection Method', ['Isolation Forest', 'Moving Average'])

        if detection_method == 'Moving Average':
            window_size = st.sidebar.number_input('Window Size', min_value=1, value=5)
            sigma = st.sidebar.number_input('Sigma Threshold', min_value=0.1, value=2.0, step=0.1)

        # Plot the time series data and anomalies for the selected feature
        st.subheader(f'Time Series & Anomaly Detection for {selected_feature} using {detection_method}:')

        # Detect anomalies based on selected method
        if detection_method == 'Moving Average':
            anomalies = detect_anomalies_moving_avg(df[selected_feature], window_size=window_size, sigma=sigma)
        elif detection_method == 'Isolation Forest':
            anomalies = detect_anomalies_isolation_forest(df[selected_feature])

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
