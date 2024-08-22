import streamlit as st
import plotly.express as px
import pandas as pd
from darts.models import ARIMA, AutoARIMA, NaiveDrift
from darts import TimeSeries
from darts.metrics import mape, rmse
from darts.datasets import AirPassengersDataset, AusBeerDataset
from utils import load_data, select_columns, select_machine_id, filter_and_sort_data, select_feature, create_timeseries, load_pdmt_telemetry


def configure_dataframe(df):
    """Helper function to configure DataFrame for time series analysis."""
    with st.sidebar.expander("Columns", expanded=True):
        timestamp_col, machine_id_col = select_columns(df)
        selected_machine_id = select_machine_id(df, machine_id_col)
        df = filter_and_sort_data(df, timestamp_col, machine_id_col, selected_machine_id)
        selected_feature = select_feature(df, timestamp_col, machine_id_col)

    return df, timestamp_col, selected_feature


st.title('Time Series Forecasting')

# Checkbox for loading example dataset
use_example_data = st.sidebar.checkbox("Load Example Dataset")

if use_example_data:
    dataset_name = st.sidebar.selectbox(
        "Select Example Dataset",
        [
            "AirPassengers",
            "AusBeer",
            "PdMTelemetry"
        ]
    )

    # Load the selected example dataset
    if dataset_name == "AirPassengers":
        ts = AirPassengersDataset().load()
    elif dataset_name == "AusBeer":
        ts = AusBeerDataset().load()
    elif dataset_name == "PdMTelemetry":
        df = load_pdmt_telemetry()
        df, timestamp_col, selected_feature = configure_dataframe(df)
        ts = create_timeseries(df, timestamp_col, selected_feature)

else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)

        # Configure DataFrame
        df, timestamp_col, selected_feature = configure_dataframe(df)
        ts = create_timeseries(df, timestamp_col, selected_feature)

    else:
        st.warning("Please upload a CSV file or select an example dataset.")
        ts = None  # Set ts to None if no data is provided

if ts is not None:
    # Determine the split index for 10% of the data
    split_index = int(len(ts) * 0.1)
    train_ts, test_ts = ts[:-split_index], ts[-split_index:]

    # Move model selection before forecast horizon
    with st.sidebar.expander("Forecasting Parameters", expanded=True):
        model_type = st.selectbox(
            'Select Forecasting Model',
            ['ARIMA', 'AutoARIMA', 'NaiveDrift'],
            index=1,  # Set AutoARIMA as the default model
            help="Choose the forecasting model to use."
        )

        forecast_horizon = int(len(test_ts) * 1.5)  # Default forecast horizon to 1.5 times the test set length

        forecast_horizon = st.number_input(
            'Forecast Horizon (in timesteps)',
            min_value=1,
            value=forecast_horizon,
            help=f"Specify the number of timesteps to forecast into the future. Default is 1.5 times the length of the test set."
        )

        # Additional hyperparameters for model tuning
        if model_type == 'ARIMA':
            p = st.slider('ARIMA p', 0, 10, 1,
                help="Number of lag observations included in the model (AR term).")
            d = st.slider('ARIMA d', 0, 2, 1,
                help="Number of times that the raw observations are differenced (I term).")
            q = st.slider('ARIMA q', 0, 10, 1,
                help="Size of the moving average window (MA term).")

        elif model_type == 'AutoARIMA':
            seasonal = st.selectbox(
                'Seasonal Period',
                [None, 24, 168, 730],  # Values for hourly data
                help="Specify the seasonal period if applicable. None for no seasonality, or specify periods like 24 (daily), 168 (weekly), 730 (yearly)."
            )
            seasonal_periods = st.slider(
                'Seasonal Period Length',
                1, 730, 24,
                help="Specify the length of the seasonal period (in timesteps)."
            )

    # Initialize and fit the model
    if model_type == 'ARIMA':
        model = ARIMA(p=p, d=d, q=q)
    elif model_type == 'AutoARIMA':
        model = AutoARIMA(seasonal=seasonal, seasonal_periods=seasonal_periods)
    elif model_type == 'NaiveDrift':
        model = NaiveDrift()

    # Train the model
    model.fit(train_ts)

    # Make predictions
    forecast = model.predict(forecast_horizon)

    # Convert time series to DataFrames for Plotly
    train_df = pd.DataFrame({
        'timestamp': train_ts.time_index,
        'value': train_ts.values().flatten(),
        'type': 'Train Data'
    })

    test_df = pd.DataFrame({
        'timestamp': test_ts.time_index,
        'value': test_ts.values().flatten(),
        'type': 'Test Data'
    })

    forecast_test_df = pd.DataFrame({
        'timestamp': test_ts.time_index,
        'value': forecast.values()[:len(test_ts)].flatten(),
        'type': 'Test Data Prediction'
    })

    forecast_df = pd.DataFrame({
        'timestamp': forecast.time_index[len(test_ts):],
        'value': forecast.values()[len(test_ts):].flatten(),
        'type': 'Forecast'
    })

    # Combine all data for plotting
    combined_df = pd.concat([train_df, test_df, forecast_test_df, forecast_df])

    # Plot using Plotly Express
    fig = px.line(
        combined_df,
        x='timestamp',
        y='value',
        color='type',
        labels={'value': 'Value', 'timestamp': 'Time'},
        color_discrete_map={
            'Train Data': 'blue',
            'Test Data': 'blue',
            'Test Data Prediction': 'green',
            'Forecast': 'red'
        }
    )

    # Show the plot
    st.plotly_chart(fig)

    # Convert DataFrames to TimeSeries for metric calculations
    test_ts = TimeSeries.from_dataframe(pd.DataFrame({
        'timestamp': test_df['timestamp'],
        'value': test_df['value']
    }), time_col='timestamp', value_cols='value')
    forecast_ts = TimeSeries.from_dataframe(pd.DataFrame({
        'timestamp': forecast_test_df['timestamp'],
        'value': forecast_test_df['value']
    }), time_col='timestamp', value_cols='value')

    # Calculate performance metrics
    mape_value = mape(test_ts, forecast_ts)
    rmse_value = rmse(test_ts, forecast_ts)

    st.subheader('Performance Metrics')
    st.write(f"MAPE: {mape_value:.2f}%")
    st.write(f"RMSE: {rmse_value:.2f}")

    with st.expander("Info"):
        st.write(
            f"**Mean Absolute Percentage Error (MAPE)** measures the accuracy as a percentage. Lower values are better, with values under 10% considered good.\n\n")
        st.write(
            f"**Root Mean Squared Error (RMSE)** measures the average error magnitude. Lower values are better, but the interpretation depends on the scale of your data.")
