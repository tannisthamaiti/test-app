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
import shutil
from dlisio import dlis
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from os.path import join as pjoin
import io

import PyPDF2
from utils_vug import *



            

def button_clicked(fmi_array, tdep_array, start, gt,  pred, end):
    
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
    

    comparison_plot(fmiZone, tdepZone, start, gtZone,  predZone, end, tadpoleScaler, fmiRatio, fontSize, 3, 'whole', 300, (15, 30), save = True, split = False)

    


def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def merge_pdfs(pdf_paths):
    merged_pdf = PyPDF2.PdfMerger()
    
    for pdf_path in pdf_paths:
        merged_pdf.append(pjoin('whole', pdf_path))
    
    return merged_pdf


def main():
    if os.path.exists('whole'):
        shutil.rmtree('whole')
    os.makedirs('whole', exist_ok=True)
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
        default_end = 2638.35 
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
        st.divider()    
        col1,col2 = st.columns(2)
        with col1:
             # reitrerate button for changed parameters 4 values
            st.markdown("Accept Original Interpretation")
            accepted_button = st.button("Accept")
            if accepted_button:
                st.success("Accepted Original Interpretation!!")
                status = "Accepted"
                zone_start = start
                zone_end = end
                # # accepted_range = {'start': [start], 'end': [end], 'status': ['Accepted']}
                # accepted_range = pd.DataFrame({'start': [zone_start], 'end': [zone_end], 'status': ['Accepted']})
                # # data = data.concat([data, accepted_range], ignore_index=True)
                # data = pd.concat([data, accepted_range], ignore_index=True)
                cursor.execute('''
                    INSERT INTO ranges (start, end, status)
                    VALUES (?, ?, ?)
                ''', (zone_start, zone_end, 'accepted'))
                conn.commit()
            if st.button("Generate Final Report"):
                pdf_paths = os.listdir('whole')
                pdf_paths = [pdf_path for pdf_path in pdf_paths if pdf_path.endswith('.pdf')]
                merged_pdf = merge_pdfs(pdf_paths)
                
                # Provide a way to download the merged PDF
                pdf_data = io.BytesIO()
                merged_pdf.write(pdf_data)
                pdf_data.seek(0)

                st.success("Report Generated successfully! Click below to download:")
                st.download_button(label="Download Report", data=pdf_data, file_name="merged.pdf", key="merged_pdf")
                shutil.rmtree('whole')
        with col2:
            st.markdown("Flag Original Interpretation")
            flag_button = st.button("Flag")
            if flag_button:
                st.success("Flagged original Interpretation")
                status = "Flagged"
                zone_start = start
                zone_end = end
                # flagged_range = pd.DataFrame({'start': [zone_start], 'end': [zone_end], 'status': ['Flagged']})
                # data = pd.concat([data, flagged_range], ignore_index=True)
                cursor.execute('''
                INSERT INTO ranges (start, end, status)
                    VALUES (?, ?, ?)
                ''', (zone_start, zone_end, 'flagged'))
                conn.commit()
        

        conn.close()
if __name__ == "__main__":
    main()
