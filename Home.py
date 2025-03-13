import streamlit as st

st.set_page_config(layout="wide")

st.title("Welcome!")
st.markdown("""
<style>
.big-font {
    font-size: 20px !important;
    line-height: 2; /* Adjusts line spacing */
    }
</style>
<p class="big-font">This app has been developed by a team of researchers from the University of Michigan, including <a href="https://cee.engin.umich.edu/people/yin-yafeng/"> Prof. Yafeng Yin</a>, <a href="https://cee.engin.umich.edu/people/bahrami-sina/"> Dr. Sina Bahrami</a>, and <a href="https://cee.engin.umich.edu/people/li-manzi/">Dr. Manzi Li</a>, as part of the deliverables for the MDOT (Michigan Department of Transportation) research project <strong>OR24-003</strong>: \"Inductive Vehicle Charging - Identify Best Practices/Applications and Optimum Locations for Light to Heavy Duty Vehicles\".</p>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.med-font {
    font-size: 18px !important;
    line-height: 2; /* Adjusts line spacing */
}
ul.med-font {
    padding-left: 30px;  /* Adjust the indentation */
}
# ul.med-font li {
#     margin-bottom: 10px;  /* Adjust spacing between list items */
# }
</style>

<p class="big-font">
    This web app is designed to assist transit agencies and policymakers in evaluating the transition to an electric fleet. It utilizes 
    <a href="https://gtfs.org/getting-started/what-is-GTFS/">GTFS (General Transit Feed Specification)</a> data along with user inputs, including:
    <ul class="med-font">
        <li>Electric bus range (the distance an electric bus can travel on a fully charged battery),</li>
        <li>Bus energy usage,</li>
        <li>Stationary and dynamic charging power, and </li>
        <li>Stationary charging setup time (inductive/wireless chargers can start charging immediately, whereas plug-in chargers require a few minutes to connect and begin charging).</li>
    </ul>
</p>
""", unsafe_allow_html=True)

st.markdown("""
<p class="big-font">The agency app assesses the agency's full service and determines the number of blocks that can be electrified based on user-provided inputs. It also provides the number of necessary stationary chargers, the required length of dynamic chargers, and maps their locations. </p>
""", unsafe_allow_html=True)

if st.button("Take me to the agency app"):
    st.switch_page("pages/Agency.py")  

st.markdown("&nbsp;" * 50, unsafe_allow_html=True) 

st.markdown("""
<style>
.small-font {
    font-size: 10px !important;
    line-height: 1; /* Adjusts line spacing */
    }
</style>
<p class="small-font">This web app provides general information and is not intended to serve as professional advice. The information and content are provided "as is" without warranties of any kind, either express or implied, including, but not limited to, implied warranties of merchantability, fitness for a particular purpose, or non-infringement. We assume no liability for any errors or omissions in the information presented or for any damages or losses that may arise from reliance on or use of this tool. Use of this web app is solely at your own risk." </p>
""", unsafe_allow_html=True)
