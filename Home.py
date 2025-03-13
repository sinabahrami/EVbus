import streamlit as st

st.title("Welcome to My Web App")
st.write("This app helps with [describe functionality].")

if st.button("Take me to the app"):
    st.switch_page("app.py")  
