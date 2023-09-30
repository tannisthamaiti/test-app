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

from utils_vug import *

# def inital_plot(tdep_array, fmi_array, well_radius_doi, gt,start,end,min_vug_area,max_vug_area,min_circ_ratio,max_circ_ratio):
#         # c_threshold = -1
#         c_threshold = 'mean'
#         stride_mode = 5
#         k = 5
#         threshold_plotting = False
#         distribution_plot = False
#         area_histogram_plot = False
#         original_contour_plot = True
#         save_plot = True
#         greater_than_mode_percentage_plot = False
#         thrsholding_type = 'adaptive'
#         nearest_point_5 = True

#         if thrsholding_type == 'normal':
#             std_no = 1

#         mean_diff_thresh = 0.1

#         plotting_start, plotting_end = 2637.02,2887.76
#         #end of variables
        
#         well_radius_doi= well_radius_doi.reshape(-1)

    
#         stds_pos, vars_pos, skews_pos, kurts_pos = [], [], [], []
#         stds_neg, vars_neg, skews_neg, kurts_neg = [], [], [], []
#         combined_centroids, final_combined_contour, final_combined_vugs = [], [], []
#         pred_df = pd.DataFrame()
#         c = 0
#         height_idx = 0


        
        
#         contour_x, contour_y = [], []
#         total_filtered_vugs =[]
#         for one_meter_zone_start in tqdm(np.arange(start,end, 1)):
#             one_meter_zone_end = one_meter_zone_start + 1
#             output = get_one_meter_fmi_and_GT(one_meter_zone_start, one_meter_zone_end, 
#                                             tdep_array, fmi_array, well_radius_doi, gt)
#             fmi_array_one_meter_zone, tdep_array_one_meter_zone, well_radius_one_meter_zone, gtZone = output
            
#             height = fmi_array_one_meter_zone.shape[0]

#             different_thresholds = get_mode_of_interest_from_image(fmi_array_one_meter_zone, stride_mode, k)
#             for i, diff_thresh in enumerate(different_thresholds):
#                 thresold_img = apply_adaptive_thresholding(fmi_array_one_meter_zone, diff_thresh, block_size = 21, c = c_threshold)
#                 holeR, pixLen = well_radius_one_meter_zone.mean()*100, (np.diff(tdep_array_one_meter_zone)*100).mean()
                    
                
#                 contours, centroids, vugs = get_contours(thresold_img, depth_to=one_meter_zone_start, 
#                                                             depth_from=one_meter_zone_end, radius = holeR, pix_len = pixLen, 
#                                                             min_vug_area = min_vug_area, max_vug_area = max_vug_area, 
#                                                             min_circ_ratio=min_circ_ratio, max_circ_ratio=max_circ_ratio) #values changed here
#                 output = get_combined_contours_and_centroids(contours, centroids, vugs,combined_centroids, 
#                                                                 final_combined_contour, final_combined_vugs,i, threshold = 5)
#                 combined_centroids, final_combined_contour, final_combined_vugs = output

#                 filtered_contour, filtered_vugs = filter_contours_based_on_original_image(final_combined_contour, final_combined_vugs, 
#                                                                                         fmi_array_one_meter_zone, 0.2)
#                 filtered_contour_ = copy.deepcopy(filtered_contour)
#                 filtered_vugs_ = copy.deepcopy(filtered_vugs)

#                 # filter the contours based on the mean pixel in and around the original contour
                
#                 filtered_contour_, filtered_vugs_ = filter_contour_based_on_mean_pixel_in_and_around_original_contour(fmi_array_one_meter_zone, 
#                                                                                                                         filtered_contour_, 
#                                                                                                                         filtered_vugs_, 
#                                                                                                                         threshold = mean_diff_thresh)
                
#                 # total vugs count                                                                                                      
#                 total_filtered_vugs.append(filtered_vugs_)
#                 for pts in filtered_contour_:
#                     x = pts[:, 0, 0]
#                     y = pts[:, 0, 1]
#                     x = np.append(x, x[0])
#                     y = np.append(y, y[0])

#                     y+=height_idx

#                     contour_x.append(x)
#                     contour_y.append(y)
#                 detected_vugs_percentage = detected_percent_vugs(filtered_contour_, fmi_array_one_meter_zone, tdep_array_one_meter_zone, 
#                                                                 one_meter_zone_start, one_meter_zone_end)
#                 pred_df = pd.concat([pred_df, detected_vugs_percentage], axis=0)
#                 print(pred_df)
#             height_idx+=height

        
               
#         vugs_threshold = 1.5
#         vicinity_threshold = 1
#         per_page_depth_zone = abs(end-start) # if selected 6m then change to 6 if 10 then change to 10 1457.8 1498.2 (1498.8)
#         num_rows = (vicinity_threshold*2)+1
#         pred_df_copy = pred_df.reset_index(drop=True)

#         for i in range(len(pred_df_copy)-vicinity_threshold):
#             pred_df_zone = pred_df_copy.iloc[i:i+num_rows]
#             idx_voi = list(pred_df_zone.index)[1]
#             if pred_df_copy.iloc[idx_voi].Vugs<=vugs_threshold:
#                 if pred_df_zone.Vugs.max()<=vugs_threshold:
#                     pred_df_copy.loc[idx_voi, 'Vugs'] = 0

#         fmi_array_doi = fmi_array
#         tdep_array_doi = tdep_array

        
#         img_idx = 0
#         for zone_start in tqdm(np.arange(start, end, per_page_depth_zone)):
#             zone_end = zone_start + per_page_depth_zone
#             temp_mask = (tdep_array_doi>=zone_start) & (tdep_array_doi<=zone_end)
#             fmi_zone = fmi_array_doi[temp_mask]
#             pred_df_zone = pred_df_copy[(pred_df_copy.Depth>=zone_start) & (pred_df_copy.Depth<=zone_end)]
#             gt_zone = gt[(gt.Depth>=zone_start) & (gt.Depth<=zone_end)]


#             img_height = fmi_zone.shape[0]
#             coord = [[k, j] for i, (k, j) in enumerate(zip(contour_x, contour_y)) if (j.min()>=img_idx) & (j.max()<=(img_idx+img_height))]

#             _, ax = plt.subplots(1, 4, figsize=(20, 30), gridspec_kw = {'width_ratios': [3,3,1, 1], 'height_ratios': [1]})
#             ax[0].imshow(fmi_zone, cmap='YlOrBr')
#             ax[1].imshow(fmi_zone, cmap='YlOrBr')
#             for x_, y_ in coord:
#                 centroid_y = get_centeroid(np.concatenate([x_.reshape(-1, 1), (y_-img_idx).reshape(-1, 1)], axis=1))[1]
#                 scaler = MinMaxScaler((zone_start, zone_end))
#                 scaler.fit([[0], [img_height]])
#                 centroid_depth = scaler.transform([[centroid_y]])[0][0]

#                 depth_values, target_value = pred_df_zone.Depth.values, centroid_depth

#                 # Find the index where the target_value should be inserted
#                 insert_index = np.searchsorted(depth_values, target_value, side='right') - 1

#                 # Check if the target_value is greater than the last depth value, in that case, it will be inserted at the end
#                 if insert_index == len(depth_values) - 1 and target_value > depth_values[-1]:
#                     insert_index = len(depth_values) - 1
#                 if pred_df_zone.iloc[insert_index].Vugs != 0:
#                     ax[1].plot(x_, y_-img_idx, color='black', linewidth=2)

#             ax[1].set_xticks([])
#             ax[1].set_yticks([])
#             ax[0].set_xticks([])
#             ax[0].set_yticks([])
                
#             # Ensure consistent range and scaling for both pred and GT plots
#             depth_values = pred_df_zone.Depth.values  # or gt_zone.Depth.values
#             # bar_heights_pred = pred_df_zone['Vugs'].values
#             bar_heights_gt = gt_zone['Vugs'].values
#             bar_heights_pred = abs(end-start)

#             plot_barh(ax[2], depth_values, bar_heights_pred, zone_start, zone_end-0.1, "Pred\n0-25%", max_scale=25, fontsize=12)
#             plot_barh(ax[3], depth_values, bar_heights_gt, zone_start, zone_end-0.1, "GT\n0-25%", max_scale=25, fontsize=12)

            
#             ax[0].set_title("Original FMI", fontsize=12)
#             ax[1].set_title("Contours", fontsize=12)

#             plt.tight_layout()
#             st.pyplot(plt)
#             # plt.savefig(f"whole/{zone_start}.png", dpi=400, bbox_inches='tight')
#         #     plt.close()
#             # break
#             img_idx+=img_height
#             # if img_idx>=5000:
#             #     break
#         if st.button('Show Statistical Analysis'):
#             filtered_vugs = [i['area'] for filtered_vugs_ in total_filtered_vugs for i in filtered_vugs_]

#             fig1, ax = plt.subplots()
#             sns.histplot(filtered_vugs, ax=ax)
#             st.pyplot(fig1)

def st_display_pdf(pdf_file):
    '''
    Display pdf file embedded in streanlit application
    '''
    with open(pdf_file,"rb") as f:
      base64_pdf = base64.b64encode(f.read()).decode('utf-8')
      pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}"   width="400" height="1000" type="application/pdf" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>'
      st.markdown(pdf_display, unsafe_allow_html=True)

            

def button_clicked(reiterate_button, tdep_array, fmi_array, well_radius_doi, gt,start,end,min_vug_area,max_vug_area,min_circ_ratio,max_circ_ratio):
    
            st.write("FMI-Contour-Predicted plot for selected values:")

            stride_mode = 5
            k = 5
            c_threshold = 'mean'
            mean_diff_thresh = 0.1
            
            well_radius_doi= well_radius_doi.reshape(-1)

        
            combined_centroids, final_combined_contour, final_combined_vugs = [], [], []
            pred_df = pd.DataFrame()
            c = 0
            height_idx = 0

            contour_x, contour_y = [], []
            total_filtered_vugsa =[]
            for one_meter_zone_start in tqdm(np.arange(start,end, 1)):
                one_meter_zone_end = one_meter_zone_start + 1
                output = get_one_meter_fmi_and_GT(one_meter_zone_start, one_meter_zone_end, 
                                                tdep_array, fmi_array, well_radius_doi, gt)
                fmi_array_one_meter_zone, tdep_array_one_meter_zone, well_radius_one_meter_zone, gtZone = output
                
                height = fmi_array_one_meter_zone.shape[0]

                different_thresholds = get_mode_of_interest_from_image(fmi_array_one_meter_zone, stride_mode, k)
                for i, diff_thresh in enumerate(different_thresholds):
                    thresold_img = apply_adaptive_thresholding(fmi_array_one_meter_zone, diff_thresh, block_size = 21, c = c_threshold)
                    holeR, pixLen = well_radius_one_meter_zone.mean()*100, (np.diff(tdep_array_one_meter_zone)*100).mean()
                        
                    
                    contours, centroids, vugs = get_contours(thresold_img, depth_to=one_meter_zone_start, 
                                                                depth_from=one_meter_zone_end, radius = holeR, pix_len = pixLen, 
                                                                min_vug_area = min_vug_area, max_vug_area = max_vug_area, 
                                                                min_circ_ratio=min_circ_ratio, max_circ_ratio=max_circ_ratio) #values changed here
                    output = get_combined_contours_and_centroids(contours, centroids, vugs,combined_centroids, 
                                                                    final_combined_contour, final_combined_vugs,i, threshold = 5)
                    combined_centroids, final_combined_contour, final_combined_vugs = output

                    filtered_contour, filtered_vugs = filter_contours_based_on_original_image(final_combined_contour, final_combined_vugs, 
                                                                                            fmi_array_one_meter_zone, 0.2)
                    filtered_contour_ = copy.deepcopy(filtered_contour)
                    filtered_vugs_ = copy.deepcopy(filtered_vugs)

                    # filter the contours based on the mean pixel in and around the original contour
                    
                    filtered_contour_, filtered_vugs_ = filter_contour_based_on_mean_pixel_in_and_around_original_contour(fmi_array_one_meter_zone, 
                                                                                                                            filtered_contour_, 
                                                                                                                            filtered_vugs_, 
                                                                                                                            threshold = mean_diff_thresh)
                    
                    # total vugs count                                                                                                      
                    total_filtered_vugsa.append(filtered_vugs_)
                    for pts in filtered_contour_:
                        x = pts[:, 0, 0]
                        y = pts[:, 0, 1]
                        x = np.append(x, x[0])
                        y = np.append(y, y[0])

                        y+=height_idx

                        contour_x.append(x)
                        contour_y.append(y)
                    detected_vugs_percentage = detected_percent_vugs(filtered_contour_, fmi_array_one_meter_zone, tdep_array_one_meter_zone, 
                                                                    one_meter_zone_start, one_meter_zone_end)
                    pred_df = pd.concat([pred_df, detected_vugs_percentage], axis=0)
                height_idx+=height

            
                
            vugs_threshold = 1.5
            vicinity_threshold = 1
            per_page_depth_zone = abs(end-start) # if selected 6m then change to 6 if 10 then change to 10 1457.8 1498.2 (1498.8)
            num_rows = (vicinity_threshold*2)+1
            pred_df_copy = pred_df.reset_index(drop=True)

            for i in range(len(pred_df_copy)-vicinity_threshold):
                pred_df_zone = pred_df_copy.iloc[i:i+num_rows]
                idx_voi = list(pred_df_zone.index)[1]
                if pred_df_copy.iloc[idx_voi].Vugs<=vugs_threshold:
                    if pred_df_zone.Vugs.max()<=vugs_threshold:
                        pred_df_copy.loc[idx_voi, 'Vugs'] = 0

            fmi_array_doi = fmi_array
            tdep_array_doi = tdep_array

            img_idx = 0
            for zone_start in tqdm(np.arange(start, end, per_page_depth_zone)):
                zone_end = zone_start + per_page_depth_zone
                temp_mask = (tdep_array_doi>=zone_start) & (tdep_array_doi<=zone_end)
                fmi_zone = fmi_array_doi[temp_mask]
                pred_df_zone = pred_df_copy[(pred_df_copy.Depth>=zone_start) & (pred_df_copy.Depth<=zone_end)]
                gt_zone = gt[(gt.Depth>=zone_start) & (gt.Depth<=zone_end)]


                img_height = fmi_zone.shape[0]
                coord = [[k, j] for i, (k, j) in enumerate(zip(contour_x, contour_y)) if (j.min()>=img_idx) & (j.max()<=(img_idx+img_height))]

                _, ax = plt.subplots(1, 4, figsize=(20, 30), gridspec_kw = {'width_ratios': [3,3,1, 1], 'height_ratios': [1]})
                ax[0].imshow(fmi_zone, cmap='YlOrBr')
                ax[1].imshow(fmi_zone, cmap='YlOrBr')
                for x_, y_ in coord:
                    centroid_y = get_centeroid(np.concatenate([x_.reshape(-1, 1), (y_-img_idx).reshape(-1, 1)], axis=1))[1]
                    scaler = MinMaxScaler((zone_start, zone_end))
                    scaler.fit([[0], [img_height]])
                    centroid_depth = scaler.transform([[centroid_y]])[0][0]

                    depth_values, target_value = pred_df_zone.Depth.values, centroid_depth

                    # Find the index where the target_value should be inserted
                    insert_index = np.searchsorted(depth_values, target_value, side='right') - 1

                    # Check if the target_value is greater than the last depth value, in that case, it will be inserted at the end
                    if insert_index == len(depth_values) - 1 and target_value > depth_values[-1]:
                        insert_index = len(depth_values) - 1
                    if pred_df_zone.iloc[insert_index].Vugs != 0:
                        ax[1].plot(x_, y_-img_idx, color='black', linewidth=2)

                ax[1].set_xticks([])
                ax[1].set_yticks([])
                ax[0].set_xticks([])
                ax[0].set_yticks([])

                zone_start = 1
                zone_end = 1

                plot_barh(ax[2], pred_df_zone.Depth.values, pred_df_zone['Vugs'].values, 
                            zone_start, zone_end-0.1, "Pred\n0-25%", max_scale=25, fontsize = 14)  # change this for font increase
                    
                plot_barh(ax[3], gt_zone.Depth.values, gt_zone['Vugs'].values, 
                            zone_start, zone_end-0.1, "GT\n0-25%", max_scale=25, fontsize=14)  # change this for font increase

                
                ax[0].set_title("Original FMI", fontsize=14)  # change this for font increase
                ax[1].set_title("Contours", fontsize=14)  # change this for font increase

                plt.tight_layout()
                plt.savefig(f"whole/{zone_start}.png", dpi=400, bbox_inches='tight')
                st.pyplot(plt)
                # plt.savefig(f"whole/{zone_start}.png", dpi=400, bbox_inches='tight')
            #     plt.close()
                img_idx+=img_height
                # if st.button('Show Statistical Analysis for reiterated one'):
                #     filtered_vugs = [i['area'] for filtered_vugs_ in total_filtered_vugs for i in filtered_vugs_]

                #     fig1, ax = plt.subplots()
                #     sns.histplot(filtered_vugs, ax=ax)
                #     st.pyplot(fig1)

def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

def detect_vugs(start, end, tdep_array_doi, fmi_array_doi, well_radius_doi, gt, stride_mode, k, c_threshold, 
                min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio, mean_diff_thresh, pred_df, combined_centroids, 
                final_combined_contour, final_combined_vugs, height_idx, contour_x, contour_y, total_filtered_vugs):
    for one_meter_zone_start in tqdm(np.arange(start, end, 1)):
        one_meter_zone_end = one_meter_zone_start + 1
        
        # get fmi, depth, well radius and ground truth for one meter zone
        output = get_one_meter_fmi_and_GT(one_meter_zone_start, one_meter_zone_end, 
                                        tdep_array_doi, fmi_array_doi, well_radius_doi, gt)
        fmi_array_one_meter_zone, tdep_array_one_meter_zone, well_radius_one_meter_zone, gtZone = output

        height = fmi_array_one_meter_zone.shape[0]

        # get top k thresholds based on std from the derived one meter zone
        different_thresholds = get_mode_of_interest_from_image(fmi_array_one_meter_zone, stride_mode, k)
            
        # get all the contours from different thresholds for the derived one meter zone
        for i, diff_thresh in enumerate(different_thresholds):
            
            # apply thresholding on the derived one meter zone
            thresold_img = apply_adaptive_thresholding(fmi_array_one_meter_zone, diff_thresh, block_size = 21, c = c_threshold) #values changed here
            
            # get well parameters
            holeR, pixLen = well_radius_one_meter_zone.mean()*100, (np.diff(tdep_array_one_meter_zone)*100).mean()
            
            # get contours and centroids from the thresholded image of the derived one meter zone
            contours, centroids, vugs = get_contours(thresold_img, depth_to=one_meter_zone_start, 
                                                        depth_from=one_meter_zone_end, radius = holeR, pix_len = pixLen, 
                                                        min_vug_area = min_vug_area, max_vug_area = max_vug_area, 
                                                        min_circ_ratio=min_circ_ratio, max_circ_ratio=max_circ_ratio) #values changed here
            output = get_combined_contours_and_centroids(contours, centroids, vugs,combined_centroids, 
                                                            final_combined_contour, final_combined_vugs,i, threshold = 5)
            combined_centroids, final_combined_contour, final_combined_vugs = output

        # filter the contours based on the contrast of each contour with the original image
        filtered_contour, filtered_vugs = filter_contours_based_on_original_image(final_combined_contour, final_combined_vugs, 
                                                                                fmi_array_one_meter_zone, 0.2)

        # saving original filtered contour and vugs in new variable for further use
        filtered_contour_ = copy.deepcopy(filtered_contour)
        filtered_vugs_ = copy.deepcopy(filtered_vugs)

        # filter the contours based on the mean pixel in and around the original contour
        filtered_contour_, filtered_vugs_ = filter_contour_based_on_mean_pixel_in_and_around_original_contour(fmi_array_one_meter_zone, 
                                                                                                                filtered_contour_, 
                                                                                                                filtered_vugs_, 
                                                                                                                threshold = mean_diff_thresh)


        # get the contours and centroids from the filtered contours and save them in a list for further use
        # these saved contours are not relative to 1m zone, but to the whole image
        total_filtered_vugs.append(filtered_vugs_)
        for pts in filtered_contour_:
            x = pts[:, 0, 0]
            y = pts[:, 0, 1]
            x = np.append(x, x[0])
            y = np.append(y, y[0])

            y+=height_idx

            contour_x.append(x)
            contour_y.append(y)

        detected_vugs_percentage = detected_percent_vugs(filtered_contour_, fmi_array_one_meter_zone, tdep_array_one_meter_zone, 
                                                        one_meter_zone_start, one_meter_zone_end)
        pred_df = pd.concat([pred_df, detected_vugs_percentage], axis=0)

        height_idx+=height
    return pred_df, contour_x, contour_y, total_filtered_vugs

def filter(pred_df, vicinity_threshold, num_rows, vugs_threshold):
    pred_df_copy = pred_df.reset_index(drop=True)
    for i in range(len(pred_df_copy)-vicinity_threshold):
        pred_df_zone = pred_df_copy.iloc[i:i+num_rows]
        idx_voi = list(pred_df_zone.index)[1]
        if pred_df_copy.iloc[idx_voi].Vugs<=vugs_threshold:
            if pred_df_zone.Vugs.max()<=vugs_threshold:
                pred_df_copy.loc[idx_voi, 'Vugs'] = 0
    return pred_df_copy

def plot(fmi_zone, pred_df_zone, start, end, contour_x, contour_y, gt):
    gt_zone = gt[(gt.Depth>=start) & (gt.Depth<=end)]

    coord = [[k, j] for i, (k, j) in enumerate(zip(contour_x, contour_y))]

    _, ax = plt.subplots(1, 4, figsize=(20, 30), gridspec_kw = {'width_ratios': [3,3,1, 1], 'height_ratios': [1]})
    ax[0].imshow(fmi_zone, cmap='YlOrBr')
    ax[1].imshow(fmi_zone, cmap='YlOrBr')
    for x_, y_ in coord:
        centroid_y = get_centeroid(np.concatenate([x_.reshape(-1, 1), (y_).reshape(-1, 1)], axis=1))[1]
        scaler = MinMaxScaler((start, end))
        scaler.fit([[0], [fmi_zone.shape[0]]])
        centroid_depth = scaler.transform([[centroid_y]])[0][0]

        depth_values, target_value = pred_df_zone.Depth.values, centroid_depth

        # Find the index where the target_value should be inserted
        insert_index = np.searchsorted(depth_values, target_value, side='right') - 1

        # Check if the target_value is greater than the last depth value, in that case, it will be inserted at the end
        if insert_index == len(depth_values) - 1 and target_value > depth_values[-1]:
            insert_index = len(depth_values) - 1
        if pred_df_zone.iloc[insert_index].Vugs != 0:
            ax[1].plot(x_, y_, color='black', linewidth=2)

    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    plot_barh(ax[2], pred_df_zone.Depth.values, pred_df_zone['Vugs'].values, 
                    start, end-0.1, "Pred\n0-25%", max_scale=25, fontsize = 12)
        
    plot_barh(ax[3], gt_zone.Depth.values, gt_zone['Vugs'].values, 
                    start, end-0.1, "GT\n0-25%", max_scale=25, fontsize = 12)

    ax[0].set_title("Original FMI", fontsize=12)
    ax[1].set_title("Contours", fontsize=12)

    plt.tight_layout()
    st.pyplot(plt)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # st.set_page_config(page_title="Vug Detection", page_icon="🤖", layout="wide", )  
    st.header("Automatic vug analysis from FMI logs")
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
        fmi_df = pd.read_csv("fmi_array.csv")
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
            # pdf_path = "SH R-1H1_Merged_diff.pdf"

            # Path to your existing file
            file_path = "SH R-1H1_Merged_diff.pdf"

            # Create a download button for the existing file
            
            with open(file_path, "rb") as file:
                file_data = file.read()
            # Generate a downloadable link for the file
            st.download_button(label="Download File", data=file_data, file_name="SH R-1H1_Merged_diff.pdf", key="download_existing_file")

            # st.file_download(pdf_path, label="Download File", key="download_file")
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

        plot(fmi_array_doi, pred_df, start, end, contour_x, contour_y, gt)

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
            print(df1.head())
            # df1['Status'] = status
            csv_data = df1.to_csv(index=False)

            # Create a download button for the CSV file
            st.download_button(
                label="Click here to download report for depth along with vug predicted",
                data=csv_data,
                file_name="depth_vs_vug.csv",
                key="download_vug_prediction_button"
            )
            
                
        st.divider()   
        if reiterate_button:
            button_clicked(reiterate_button, tdep_array, fmi_array, well_radius_doi, gt,start,end,min_vug_area,max_vug_area, min_circ_ratio, max_circ_ratio)
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
