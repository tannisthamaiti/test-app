import base64
import copy
import os
import random
import sqlite3
import string
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from dlisio import dlis
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from os.path import join as pjoin
import io

import PyPDF2
from utils_vug import *



            

def button_clicked(fmi_array, tdep_array, start, gt,  pred, end):
    
    zoneStart, zoneEnd = 2640, 2645
    fontSize = 15
    tadpoleLength = 600
    
    
    idx_start = find_nearest(tdep_array, start)
    idx_end = find_nearest(tdep_array, end)
    gtZone = gt[(gt.Depth>=start) & (gt.Depth<=end)]
    predZone = pred[(pred.Depth>=start) & (pred.Depth<=end)]
    fmiZone = fmi_array[idx_start:idx_end]
    tdepZone = tdep_array[idx_start:idx_end]
    windowSize = fmiZone.shape[0]
    fmiLength = fmiZone.shape[0]
    fmiRatio = fmiLength/tadpoleLength
    tadpoleScaler = MinMaxScaler((0, fmiZone.shape[0]))
    tadpoleScaler.fit([[start], [end]])
    st.markdown("Code working")
    st.write("idx_end",tdepZone[0],tdepZone[-1])

    comparison_plot(fmiZone, tdepZone, start, gtZone,  predZone, end, tadpoleScaler, fmiRatio, fontSize, 3, None, 300, (15, 30), save = True, split = True)

    


def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def merge_pdfs(pdf_paths):
    merged_pdf = PyPDF2.PdfMerger()
    
    for pdf_path in pdf_paths:
        merged_pdf.append(pjoin('whole', pdf_path))
    
    return merged_pdf


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # st.set_page_config(page_title="Vug Detection", page_icon="🤖", layout="wide", )  
    st.header("Automatic Fracture analysis from FMI logs")
    df1 = []
    
    conn = sqlite3.connect('your_database_name.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ranges (
            start REAL,
            end REAL,
            status TEXT
        )
    ''')
    def add_to_database(button_name):
        cursor.execute("INSERT INTO ranges (status) VALUES (?)", (button_name,))
        conn.commit()

    def clear_database():
        cursor.execute("DELETE FROM ranges")
        conn.commit()

    def clear_database_periodically():
        while True:
            time.sleep(600)  # Sleep for 10 minutes
            clear_database()

    # Start the background thread to clear the database
    clear_thread = threading.Thread(target=clear_database_periodically)
    clear_thread.daemon = True
    clear_thread.start()
    
    uploaded_file = st.file_uploader("Upload DLIS File", type=["dlis"])
    
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False
    
    if st.button("Import preloaded DLIS"):
        st.session_state.button_clicked = True
    # if uploaded_file is not None:
    
    if st.session_state.button_clicked:

        st.success("File uploaded successfully")
        fmi_df = pd.read_csv("fmi_array_dyn.csv")
        tdep_df = pd.read_csv("tdep_array.csv")
        gt = pd.read_csv("GTReservoirZone.csv").dropna()[1:]
        pred = pd.read_csv("PredReservoirZone.csv").dropna()[1:]
        well_radius = pd.read_csv("well_radius.csv")

        
        
        fmi_array = fmi_df.to_numpy()
        tdep_array = tdep_df.to_numpy()
        well_radius = well_radius.to_numpy()
        tdep_array = tdep_array.reshape(-1) 
        well_radius = well_radius.reshape(-1)
        fmi_array[fmi_array == -9999.0] = np.nan
        fmi_array = MinMaxScalerCustom(fmi_array, min = 0, max = 255)

        

        col1,col2,col3 = st.columns(3)
        with col1:
            
            #################### IMAGE DISPLAY BUTTON HERE ########################
            # existing pdf file path
            file_path = "SH R-1H1_Merged_diff.pdf"
            with open(file_path, "rb") as file:
                file_data = file.read()
            # Generate a downloadable link for the file
            st.download_button(label="Download File", data=file_data, file_name="SH R-1H1_Merged_diff.pdf", key="download_existing_file")

       
        
        data = pd.DataFrame(columns=['start', 'end', 'status'])
        if 'data' not in st.session_state:
            st.session_state.data = pd.DataFrame(columns=['start', 'end', 'status'])


        st.divider()
        # subheader to let the user know to end start and end to be of like 6m diff for least latency
        # st.write(':red[*] For this MVP, the application can plot a minimum input section of 1m and a maximum input section of 6m starting from the `Min Depth`')
        st.subheader('For this MVP, the application can plot a minimum input section of 1m and a maximum input section of 6m starting from the `Min Depth`')
        st.caption(':red[Entering a Max Depth <= Min Depth will result in an error]')
        col1,col2 = st.columns(2)

        default_start = 2633.35
        default_end = 2659.28 
        with col1:
            start = st.number_input("Min Depth ", value=default_start)
        with col2:
            end = st.number_input("Max Depth", value=default_end)

        button_clicked(fmi_array, tdep_array, start, gt,  pred, end)
        cursor.execute('''
                UPDATE ranges
                SET status = ?
                WHERE start = ? AND end = ?
            ''', ('evaluated', start, end))
        conn.commit()

        if not data.empty:
            st.markdown("Download Accepted and Flagged Ranges")

            cursor.execute('SELECT * FROM ranges')
            rows = cursor.fetchall()
            csv_data = pd.DataFrame(rows, columns=['start', 'end', 'status']).to_csv(index=False)
            st.download_button(
                label="Click here",
                data=csv_data,
                file_name="accepted_flagged_report.csv",
                key="download_ranges_button"
            )
        conn.close()
if __name__ == "__main__":
    main()
