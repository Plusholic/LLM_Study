import streamlit as st
from streamlit_file_browser import st_file_browser

st.header('Default Options')
event = st_file_browser(key='A',
                        path='/Users/jeonjunhwi/문서/Projects/GNN_Covid/',
                        show_choose_file=True,
                        show_download_file=False)
st.write(event)
st.link_button("Go to gallery", "/Users/jeonjunhwi/문서/Projects/GNN_Covid/refference/GNN논문/A deep spatio-temporal meta-learning model for urban traffic revitalization index prediction in the COVID-19 pandemic.pdf")
# st.header('With Artifacts Server, Allow choose file, disable download')
# event = st_file_browser("example_artifacts", artifacts_site="http://localhost:1024", show_choose_file=True, show_download_file=False, key='B')
# st.write(event)

# st.header('Show only molecule files')
# event = st_file_browser("example_artifacts", artifacts_site="http://localhost:1024", show_choose_file=True, show_download_file=False, glob_patterns=('molecule/*',), key='C')
# st.write(event)