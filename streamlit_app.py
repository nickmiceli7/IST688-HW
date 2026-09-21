import streamlit as st

st.title("HW Manager")
hw1 = st.Page('HW/HW1.py', title='HW1')
hw2 = st.Page('HW/HW2.py', title='HW2')
hw3 = st.Page('HW/HW3.py', title='HW3')
hw4 = st.Page('HW/HW4.py', title='HW4', default=True)

pg = st.navigation([hw1, hw2, hw3, hw4])
pg.run()