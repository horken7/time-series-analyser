import streamlit as st

anomaly_detection_page = st.Page("anomaly_detection.py", title="Anomaly detection", icon=":material/scanner:")

pg = st.navigation([anomaly_detection_page])
st.set_page_config(page_title="Data manager", page_icon=":material/edit:")
pg.run()
