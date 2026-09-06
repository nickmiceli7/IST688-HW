import streamlit as st

st.title("HW Manager")
hw1 = st.Page('HW/HW1.py', title='HW1')
hw2 = st.Page('HW/HW2.py', title='HW2', default=True)

pg = st.navigation([hw1, hw2])
pg.run()