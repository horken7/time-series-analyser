Time Series Analyzer
====================

This Streamlit app visualizes time series data and detects anomalies using two methods: Moving Average and Isolation Forest. 

# Features

- **Upload CSV**: Upload your time series data in CSV format.
- **Select Feature**: Choose the feature to analyze from your dataset.
- **Choose Detection Method**: Select between Moving Average or Isolation Forest for anomaly detection.
- **Visualize Anomalies**: View time series data with detected anomalies highlighted.

# How to Use

1. **Upload Data**:
   - Go to the sidebar and upload a CSV file containing your time series data.

2. **Select Parameters**:
   - **Feature**: Choose the column to analyze.
   - **Detection Method**: Select either "Moving Average" or "Isolation Forest."
     - **Moving Average**: Adjust the window size and sigma threshold as needed.
     - **Isolation Forest**: No additional parameters are required.

3. **View Results**:
   - The main panel will display a plot of the time series data with detected anomalies marked.
   - Detected anomalies are listed below the plot if any are found.

## Running

```bash
streamlit run app.py
```

This will open the app in your default web browser.