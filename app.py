import base64
import copy
import io
import os
import random
import sqlite3
import string
import threading
import time
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2
import seaborn as sns
import streamlit as st
from dlisio import dlis
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from utils_vug import *

def st_display_pdf(pdf_file):
    '''
    Display pdf file embedded in streanlit application
    '''
    with open(pdf_file,"rb") as f:
      base64_pdf = base64.b64encode(f.read()).decode('utf-8')
      pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}"   width="400" height="1000" type="application/pdf" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>'
      st.markdown(pdf_display, unsafe_allow_html=True)

            

def button_clicked(start, end, tdep_array_doi, fmi_array_doi, well_radius_doi, gt, stride_mode, k, c_threshold, 
                   min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio, mean_diff_thresh, pred_df, combined_centroids, 
                   final_combined_contour, final_combined_vugs, height_idx, contour_x, contour_y, total_filtered_vugs, vicinity_threshold, num_rows, vugs_threshold):
    
    st.write("FMI-Contour-Predicted plot for selected values:")

    pred_df, contour_x, contour_y, total_filtered_vugs = detect_vugs(start, end, tdep_array_doi, fmi_array_doi, well_radius_doi, gt, stride_mode, k, c_threshold, 
                                                min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio, mean_diff_thresh, pred_df, 
                                                combined_centroids, final_combined_contour, final_combined_vugs, height_idx, contour_x, contour_y, total_filtered_vugs)

    pred_df = filter(pred_df, vicinity_threshold, num_rows, vugs_threshold)
    df1 = pred_df

    plot(fmi_array_doi, pred_df, start, end, contour_x, contour_y, gt, fontsize = 25)


def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # st.set_page_config(page_title="Vug Detection", page_icon="🤖", layout="wide", )  
    st.header("Automatic vug analysis from FMI logs")
    df1 = []
    
    conn = sqlite3.connect('your_database.db')
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
        fmi_df = pd.read_csv("fmi_array_stat.csv")
        tdep_df = pd.read_csv("tdep_array.csv")
        gt = pd.read_csv("Khufai_Vugs.csv").dropna()[1:].astype('float')
        well_radius = pd.read_csv("well_radius_array.csv")

        
        
        fmi_array = fmi_df.to_numpy()
        tdep_array = tdep_df.to_numpy()
        well_radius = well_radius.to_numpy()
        tdep_array = tdep_array.reshape(-1) 
        well_radius = well_radius.reshape(-1)

        ##########################################################
        # Display PDF here 
        ########################################################
        # fig, ax1 = plt.subplots(1, figsize=(10, 10), sharey=True)
        # #--------------------why changed to this?-------------
        # ax1.imshow(np.flip(fmi_array, 0), cmap="YlOrBr")
        # ax1.set_yticks(np.linspace(0, tdep_array.shape[0], 10), np.linspace(tdep_array[-1], tdep_array[0], 10).round(2))
        # ax1.invert_yaxis()
        # ax1.set_title("Original FMI")

        col1,col2,col3 = st.columns(3)
        with col1:
            #st.pyplot(fig)
            #################### IMAGE DISPLAY BUTTON HERE ########################
            # existing pdf file path
            file_path = "SH R-1H1_Merged_diff.pdf"
            with open(file_path, "rb") as file:
                file_data = file.read()
            # Generate a downloadable link for the file
            st.download_button(label="Download File", data=file_data, file_name="SH R-1H1_Merged_diff.pdf", key="download_existing_file")

            # st_display_pdf("SH R-1H1_Merged_diff.pdf")

        min_vug_area = 0.5 # Important param
        max_vug_area = 10.28 #not important params
        min_circ_ratio = 0.5 # Important param
        max_circ_ratio = 1  #not important params
        
        c_threshold = 'mean'
        mean_diff_thresh = 0.1
        stride_mode = 5
        k = 5
        height_idx = 0
        vugs_threshold = 1
        vicinity_threshold = 1
        num_rows = (vicinity_threshold*2)+1
        vugs_threshold = 1
        vicinity_threshold = 1
        combined_centroids, final_combined_contour, final_combined_vugs = [], [], []
        contour_x, contour_y = [], []
        total_filtered_vugs =[]
        pred_df = pd.DataFrame()

        
        data = pd.DataFrame(columns=['start', 'end', 'status'])
        if 'data' not in st.session_state:
            st.session_state.data = pd.DataFrame(columns=['start', 'end', 'status'])


        st.divider()
        # subheader to let the user know to end start and end to be of like 6m diff for least latency
        # st.write(':red[*] For this MVP, the application can plot a minimum input section of 1m and a maximum input section of 6m starting from the `Min Depth`')
        st.subheader('For this MVP, the application can plot a minimum input section of 1m and a maximum input section of 6m starting from the `Min Depth`')
        st.caption(':red[Entering a Max Depth <= Min Depth will result in an error]')
        col1,col2 = st.columns(2)

        default_start = 2739.02
        default_end = 2745.02
        with col1:
            start = st.number_input("Min Depth ", value=default_start)
        with col2:
            end = st.number_input("Max Depth", value=default_end)

        mask = (tdep_array>=start) & (tdep_array<=end)

        tdep_array_doi = tdep_array[mask]
        fmi_array_doi = fmi_array[mask]
        well_radius_doi = well_radius[mask]

        pred_df, contour_x, contour_y, total_filtered_vugs = detect_vugs(start, end, tdep_array_doi, fmi_array_doi, well_radius_doi, gt, stride_mode, k, c_threshold, 
                                                    min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio, mean_diff_thresh, pred_df, 
                                                    combined_centroids, final_combined_contour, final_combined_vugs, height_idx, contour_x, contour_y, total_filtered_vugs)

        pred_df = filter(pred_df, vicinity_threshold, num_rows, vugs_threshold)
        df1 = pred_df

        plot(fmi_array_doi, pred_df, start, end, contour_x, contour_y, gt, fontsize = 25)

        if st.button('Show Statistical Analysis'):
            filtered_vugs = [i['area'] for filtered_vugs_ in total_filtered_vugs for i in filtered_vugs_]

            fig1, ax = plt.subplots()
            sns.histplot(filtered_vugs, ax=ax)
            st.pyplot(fig1)

        
        st.divider()    
        col1,col2 = st.columns(2)
        
        with col1:
            # values after the user changes parameters
            new_default_min_vug_area = 0.5
            new_default_max_vug_area = 10.28
            new_default_min_circ_ratio = 0.5
            new_default_max_circ_ratio = 1.0

            min_vug_area = st.number_input("Min Vug Area (default: 0.5)", value=new_default_min_vug_area)
            max_vug_area = st.number_input("Max Vug Area (default: 10.28)", value=new_default_max_vug_area)
            min_circ_ratio = st.number_input("Min Circ Ratio (default: 0.5)", value=new_default_min_circ_ratio)
            max_circ_ratio = st.number_input("Max Circ Ratio (default: 1.0)", value=new_default_max_circ_ratio)

            # reset_button = st.button("Reset Values")

            # if reset_button:
            #     # min_vug_area = new_default_min_vug_area
            #     min_vug_area = 0.5
            #     max_vug_area = new_default_max_vug_area
            #     min_circ_ratio = new_default_min_circ_ratio
            #     max_circ_ratio = new_default_max_circ_ratio

            
        with col2:
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
               
            st.markdown("Change parameters for selected depth")
            reiterate_button = st.button("Reiterate")
            
            st.markdown("Click here to download report")
            cursor.execute('SELECT * FROM ranges')
            rows = cursor.fetchall()
            csv_data = pd.DataFrame(rows, columns=['start', 'end', 'status']).to_csv(index=False)
            st.download_button(
                label="Download Report",
                data=csv_data,
                file_name="accepted_flagged_ranges.csv",
                key="download_ranges_button"
            )

            # Convert the dataframe to CSV format
            # status = "Test"
            # df1 = [{key: round(value, 4) for key, value in inner_dict.items()} for inner_dict in df1]
            df1 = pd.DataFrame(df1)
            # df1['Status'] = status
            csv_data = df1.to_csv(index=False)

            # Create a download button for the CSV file
            st.download_button(
                label="Click here to download report for depth along with vug predicted",
                data=csv_data,
                file_name="depth_vs_vug.csv",
                key="download_vug_prediction_button"
            )

            if st.button("Generate Report"):
                pdf_paths = os.listdir('whole')
                pdf_paths = [pdf_path for pdf_path in pdf_paths if pdf_path.endswith('.pdf')]
                merged_pdf = merge_pdfs(pdf_paths)
                
                # Provide a way to download the merged PDF
                pdf_data = io.BytesIO()
                merged_pdf.write(pdf_data)
                pdf_data.seek(0)

                st.success("PDFs merged successfully! Click below to download:")
                st.download_button(label="Download Report", data=pdf_data, file_name="merged.pdf", key="merged_pdf")

                
        st.divider()   
        if reiterate_button:
            combined_centroids, final_combined_contour, final_combined_vugs = [], [], []
            contour_x, contour_y = [], []
            total_filtered_vugs =[]
            pred_df = pd.DataFrame()
            # button_clicked(reiterate_button, tdep_array, fmi_array, well_radius_doi, gt,start,end,min_vug_area,max_vug_area, min_circ_ratio, max_circ_ratio)
            mask = (tdep_array>=start) & (tdep_array<=end)
            tdep_array_doi = tdep_array[mask]
            fmi_array_doi = fmi_array[mask]
            well_radius_doi = well_radius[mask]
            button_clicked(start, end, tdep_array_doi, fmi_array_doi, well_radius_doi, gt, stride_mode, k, c_threshold, 
                   min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio, mean_diff_thresh, pred_df, combined_centroids, 
                   final_combined_contour, final_combined_vugs, height_idx, contour_x, contour_y, total_filtered_vugs, vicinity_threshold, num_rows, vugs_threshold)
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
