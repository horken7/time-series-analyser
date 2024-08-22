import streamlit as st

anomaly_detection_page = st.Page("anomaly_detection.py", title="Anomaly Detection", icon=":material/analytics:")
forecasting_page = st.Page("forecasting.py", title="Forecasting", icon=":material/calendar_month:")

pg = st.navigation([anomaly_detection_page, forecasting_page])
st.set_page_config(page_title="Time Series Analyzer", page_icon=":material/analytics:")
pg.run()
