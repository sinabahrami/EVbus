import streamlit as st

st.title("Welcome!")
st.markdown("""
<style>
.big-font {
    font-size: 20px !important;
    line-height: 2; /* Adjusts line spacing */
    }
</style>
<p class="big-font">This app has been developed by a team of researchers from the University of Michigan, including <a href="https://cee.engin.umich.edu/people/yin-yafeng/"> Prof. Yafeng Yin</a>, <a href="https://sinabahrami.github.io/"> Dr. Sina Bahrami</a>, and Dr. Manzi Li, as part of the deliverables for the MDOT (Michigan Department of Transportation) research project <strong>OR24-003</strong>: \"Inductive Vehicle Charging - Identify Best Practices/Applications and Optimum Locations for Light to Heavy Duty Vehicles\".</p>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.big-font {
    font-size: 20px !important;
    line-height: 2; /* Adjusts line spacing */
    }
</style>
<p class="big-font">This web app is designed to assist transit agencies and policymakers in evaluating the transition to an electric fleet. It utilizes <a href="https://gtfs.org/getting-started/what-is-GTFS/> GTFS (General Transit Feed Specification)</a> data along with user inputs, including:\".</p>
""", unsafe_allow_html=True)


if st.button("Take me to the app"):
    st.switch_page("pages/app.py")  
