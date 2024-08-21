import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest
from utils import load_data, select_columns, select_machine_id, filter_and_sort_data, select_feature

@st.cache_data
def detect_anomalies_moving_avg(df, feature_col, window_size=5, sigma=1.75):
    rolling_mean = df[feature_col].rolling(window=window_size).mean()
    rolling_std = df[feature_col].rolling(window=window_size).std()
    anomalies = (df[feature_col] - rolling_mean).abs() > sigma * rolling_std
    df['Anomaly'] = anomalies
    return df

@st.cache_data
def detect_anomalies_isolation_forest(df, feature_col, contamination='auto'):
    series = df[feature_col].values.reshape(-1, 1)
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(series)
    preds = model.predict(series)
    df['Anomaly'] = preds == -1
    return df

st.title('Time Series Analyzer')

st.write("""
    **Welcome to the Anomaly Detection page!**

    Start by uploading your CSV file from the sidebar. An example dataset can be found [here](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance?select=PdM_maint.csv).
""")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"], help="The file must contain one column with timestamps and one column with target values, and optionally a column for machineID")

if uploaded_file:
    df = load_data(uploaded_file)

    with st.sidebar.expander("Columns", expanded=True):
        # Use the utility functions to select columns and filter data
        timestamp_col, machine_id_col = select_columns(df)
        selected_machine_id = select_machine_id(df, machine_id_col)
        df = filter_and_sort_data(df, timestamp_col, machine_id_col, selected_machine_id)
        selected_feature = select_feature(df, timestamp_col, machine_id_col)

    with st.sidebar.expander("Anomaly Detection", expanded=True):
        detection_method = st.selectbox(
            'Select Detection Method',
            ['Moving Average', 'Isolation Forest'],
            help="Choose the method for detecting anomalies in the selected feature."
        )

        if detection_method == 'Moving Average':
            window_size = st.number_input(
                'Window Size',
                min_value=1,
                value=5,
                help="Specify the window size for the moving average."
            )
            sigma = st.number_input(
                'Sigma Threshold',
                min_value=0.1,
                value=1.75,
                step=0.1,
                help="Set the sigma threshold for anomaly detection using the moving average."
            )
            df_anomalies = detect_anomalies_moving_avg(df, selected_feature, window_size=window_size, sigma=sigma)
        else:
            contamination = st.text_input(
                'Contamination',
                value='auto',
                help="Set the contamination level for the Isolation Forest method (auto or a float between 0.01 and 0.5)."
            )
            df_anomalies = detect_anomalies_isolation_forest(df, selected_feature, contamination=contamination)

    # Create the main time series plot
    fig = px.line(df_anomalies, x=timestamp_col, y=selected_feature, title=f'Time Series Data with Anomalies in {selected_feature}')

    # Add the anomalies scatter plot on top
    fig.add_scatter(
        x=df_anomalies.loc[df_anomalies['Anomaly'], timestamp_col],
        y=df_anomalies.loc[df_anomalies['Anomaly'], selected_feature],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=8, line=dict(color='red', width=2)),
        showlegend=True  # Ensures the legend is visible
    )

    st.plotly_chart(fig)

    if not df_anomalies[df_anomalies['Anomaly']].empty:
        st.subheader('Detected Anomalies:')
        st.write(df_anomalies[df_anomalies['Anomaly']])
    else:
        st.write('No anomalies detected.')
