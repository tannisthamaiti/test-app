import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle   
from io import BytesIO
import base64
import csv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
# import time



os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import cv2


def get_binary_file_downloader_html(buffer, filename, link_text):
    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


# for data loading -> progress bar
# import time

# progress_text = "Operation in progress. Please wait."
# my_bar = st.progress(0, text=progress_text)

# for percent_complete in range(100):
#     time.sleep(0.1)
#     my_bar.progress(percent_complete + 1, text=progress_text)
# Function to display the uploaded image with zoom in/out

def display_image(image, zoom_factor):
    width, height = image.size
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    resized_image = image.resize((new_width, new_height))
    st.image(resized_image)
    
def display_image1(image, zoom_factor):
    image = Image.open('contours/image.png')
    width, height = image.size
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    resized_image = image.resize((new_width, new_height))
    st.image(resized_image, width=new_width)

# Function to estimate vug% for the entire well (dummy data)
def estimate_vug_percent(image):
    # Dummy vug% estimation, replace with actual algorithm
    return np.random.randint(0, 100)

# Function to display the graphs (dummy data)
# ... Rest of your code ...

def display_graphs():
    # Generate dummy data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(x, y1, label="Graph 1 (Otsu)")
    ax.plot(x, y2, label="Graph 2 (Adaptive Filtering)")
    ax.legend()

    return fig



def create_database():
    # Replace 'your_database.db' with the path to your SQLite3 database file
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS vug_reports
                      (image_id TEXT PRIMARY KEY, vug_percent REAL)''')
    conn.close()

# Function to store the reported vug% in the SQLite3 database
def save_vug_percent_to_db(image_id, reported_vug_percent):
    # Replace 'your_database.db' with the path to your SQLite3 database file
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT OR REPLACE INTO vug_reports (image_id, vug_percent) 
                      VALUES (?, ?)''', (image_id, reported_vug_percent))
    conn.commit()
    conn.close()

def modify_image(image, color, contrast):
    # Convert the image to numpy array
    np_image = np.array(image)

    # Normalize the pixel values to [0, 1]
    np_image = np_image / 255.0

    # Apply color adjustment
    if color == "Red":
        np_image[:, :, 0] = np_image[:, :, 0] * 1.5  # Increase red channel
    elif color == "Green":
        np_image[:, :, 1] = np_image[:, :, 1] * 1.5  # Increase green channel
    elif color == "Blue":
        np_image[:, :, 2] = np_image[:, :, 2] * 1.5  # Increase blue channel

    # Apply contrast adjustment
    if contrast == "Low":
        np_image = np_image * 0.8
    elif contrast == "High":
        np_image = np_image * 1.2

    # Clip the pixel values to [0, 1]
    np_image = np.clip(np_image, 0, 1)

    # Convert the numpy array back to PIL Image
    modified_image = Image.fromarray((np_image * 255).astype(np.uint8))

    return modified_image

# this generates the report
def generate_report(image, vug_percent, corrected_vug, graphs):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Add header
    header_text = "Geologist Vug% Report"
    header = Paragraph(header_text, styles['h1'])
    header.alignment = 1  # Center alignment
    doc.build([header])

    # Add uploaded image
    doc.build([Spacer(1, 20)])  # Some spacing between header and image
    doc.build([RLImage(image, width=400, height=300, hAlign='CENTER')])

    # Add vug% information
    doc.build([Spacer(1, 20)])  # Some spacing between image and vug% info
    vug_info_text = f"Estimated Vug% for the entire well: {vug_percent}%"
    vug_info = Paragraph(vug_info_text, styles['Normal'])
    doc.build([vug_info])

    # Add graphs
    doc.build([Spacer(1, 20)])  # Some spacing between vug% info and graphs
    for _ in range(1):
        fig = display_graphs()
        st.pyplot(fig)
        # Add the figure object to 'graphs' list
        graphs.append(fig)

    # Add corrected vug% if applicable
    if corrected_vug is not None:
        doc.build([Spacer(1, 20)])  # Some spacing between graphs and corrected vug% info
        corrected_vug_text = f"Corrected Vug% submitted: {corrected_vug}%"
        corrected_vug_info = Paragraph(corrected_vug_text, styles['Normal'])
        doc.build([corrected_vug_info])

    # Add footer
    doc.build([Spacer(1, 40)])  # Some spacing between content and footer
    footer_text = "Generated by Geologist Vug% Report App"
    footer = Paragraph(footer_text, styles['Normal'])
    footer.alignment = 1  # Center alignment
    doc.build([footer])

    # Close the PDF document
    doc.build([])

    # Save the PDF content to a file
    buffer.seek(0)
    return buffer

def generate_dummy_data(size=1000):
    return np.random.randn(size)

def load_image_from_depth(selected_depth):
    image_path = os.path.join("data", f"{selected_depth}.png")
    return Image.open(image_path)


# Main Streamlit app
def main():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    st.set_page_config(page_title="Geologist Vug% Report", layout="wide")
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #004466 !important;
            color: #FFFFFF !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: bold !important;
        }

        .stTextInput>div>div>input {
            background-color: #edf6f9 !important;
            color: black !important;
            font-family: 'Inter', sans-serif !important;
        }

        .stTextArea>div>textarea {
            background-color: #edf6f9 !important;
            color: black !important;
            font-family: 'Inter', sans-serif !important;
        }

        .css-17eq0hr {
            background-color: #edf6f9 !important;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
            color: #FFFFFF;
            font-weight: bold;
        }

        .st-bb {
            background-color: #d9f1f8 !important;
            padding: 5px 15px;
            border-radius: 5px;
        }

        .st-bc {
            background-color: #edf6f9 !important;
            color: #FFFFFF !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: bold !important;
            padding: 5px 15px;
            border-radius: 5px;
        }

        .st-dz {
            color: #000000 !important;
            font-family: 'Inter', sans-serif !important;
        }

        .st-de {
            color: #FFFFFF !important;
            font-family: 'Inter', sans-serif !important;
        }

        .st-cu {
            background-color: #edf6f9 !important;
            border-radius: 5px;
            padding: 10px;
            color: black;
            font-family: 'Inter', sans-serif;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
    container_style = "padding: 15px; border-radius: 5px; margin-bottom: 10px; color: #FFFFFF; font-weight: bold;"
 
    st.header("Automatic vug detection from FMI logs")
    uploaded_file = st.file_uploader("Choose a dlis file....", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.subheader("Uploaded Image")
        zoom_factor = st.slider("Zoom Factor:", 0.1, 3.0, 0.16)
        st.subheader("Zoomed-in Image:")
        display_image(image, zoom_factor)

        vug_percent = estimate_vug_percent(image)
        st.markdown(
        f'<div style="background-color: #edf6f9; {container_style} font-family: Inter; color: black;">'
        f'Estimated Vug% for the entire well: {vug_percent}%<br>'
        '</div>',
        unsafe_allow_html=True,
    )
    
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_depth = st.selectbox("Select Depth:", ["2635-2640m", "2640-2645m", "2645-2650m","2650-2655m","2655-2660m"])

        with col2:
            color_options = ["Red", "Green", "Blue"]
            selected_color = st.selectbox("Select Color Filter:", color_options)

        with col3:
            contrast_options = ["Low", "Medium", "High"]
            selected_contrast = st.selectbox("Select Contrast:", contrast_options)

        modified_image = modify_image(image, selected_color, selected_contrast)

        st.subheader("Modified Image:")
        display_image(modified_image, zoom_factor)

        
        img = np.array(image)
        import cv2 as cv
        from matplotlib import pyplot as plt
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        img_gray = cv.medianBlur(img_gray, 5)
        th2 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                   cv.THRESH_BINARY, 11, 2)

        titles = ['Original Image',
                  'Adaptive Mean Thresholding', "Otsu's Thresholding"]
        images = [img_gray, th2]

        # Define the target size for the resized images
        target_width = 200
        target_height = 300

        # Create a button to show the thresholding results
        if st.button("Show Adaptive Filtering", key="show_results_button"):
            st.write("Thresholding Results:")
            for i in range(2):
                # Resize the image to the target size
                resized_image = cv2.resize(images[i], (target_width, target_height))

                # Create a smaller plot by setting the figsize parameter
                plt.figure(figsize=(8, 6))
                plt.imshow(resized_image, 'gray')
                plt.title(titles[i])
                plt.xticks([])
                plt.yticks([])

                # Save the plot to a BytesIO buffer
                buffer = BytesIO()
                plt.savefig(buffer, format="png")
                buffer.seek(0)

                # Encode the image to base64
                image_base64 = base64.b64encode(buffer.read()).decode()

                # Create an HTML link that opens the image in a new browser tab
                image_link = f'<a href="data:image/png;base64,{image_base64}" target="_blank"><img src="data:image/png;base64,{image_base64}" /></a>'
                st.write(image_link, unsafe_allow_html=True)
        
        img = np.array(image)
        import cv2 as cv
        from matplotlib import pyplot as plt
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        img_gray = cv.medianBlur(img_gray, 5)
        _, th_otsu = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        titles = ['Original Image', "Otsu's Thresholding"]
        images = [img_gray, th_otsu]

        # Define the target size for the resized images
        target_width = 200
        target_height = 300

        # Create a button to show the thresholding results
        if st.button("Show Otsu Filtering", key="show_results_buttons"):
            st.write("Thresholding Results:")
            for i in range(2):
                # Resize the image to the target size
                resized_image = cv2.resize(images[i], (target_width, target_height))

                # Create a smaller plot by setting the figsize parameter
                plt.figure(figsize=(8, 6))
                plt.imshow(resized_image, 'gray')
                plt.title(titles[i])
                plt.xticks([])
                plt.yticks([])

                # Save the plot to a BytesIO buffer
                buffer = BytesIO()
                plt.savefig(buffer, format="png")
                buffer.seek(0)

                # Encode the image to base64
                image_base64 = base64.b64encode(buffer.read()).decode()

                # Create an HTML link that opens the image in a new browser tab
                image_link = f'<a href="data:image/png;base64,{image_base64}" target="_blank"><img src="data:image/png;base64,{image_base64}" /></a>'
                st.write(image_link, unsafe_allow_html=True)
        
        # chart_data = pd.DataFrame(
        #     np.random.randn(20, 3),
        #     columns=['a', 'b', 'c'])

        # st.area_chart(chart_data)
        # area_chart
        # line_chart

        # st.header("Predicted Vug for")
        # st.markdown(f'{depth}')
        # # display image.png
        # st.image('image.png', width=50)
        # st.header("Predicted Vug for : ")
        st.header(
            f"Predicted Vug% for the entire well: {selected_depth}"
            )
        st.image(load_image_from_depth(selected_depth),width=50)
        
        # Add two buttons A and B
        if st.button("Histogram"):
            # with st.spinner("Gathering information..."):
            #     # Simulate some data processing
            #     time.sleep(1)
            # with st.spinner("Processing data..."):
            #     # Simulate some data processing
            #     time.sleep(1)

            st.success("Done!")

            # Show the actual content
            st.subheader("Histogram")

            # Generate dummy data
            data = generate_dummy_data()

                    # Reduce the size of the plot
            plt.figure(figsize=(4, 2))  # Specify the size of the figure here (width, height)

            # Create the histogram-like bar plot with smaller size
            counts, bins, patches = plt.hist(data, bins=10, alpha=0.7, color='b', edgecolor='black')

            # Customize the appearance
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.title("Histogram Visualization")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Show the plot using Streamlit
            st.pyplot(plt)

            # Optionally, you can also display the raw counts as a table
            # st.subheader("Raw Counts")
            # counts_table = np.vstack((bins[:-1], bins[1:], counts)).T
            # st.table(counts_table)
                    

        if st.button("FMI contours"):
            # with st.spinner("Gathering information..."):
            #     # Simulate some data processing
            #     time.sleep(1)
            # with st.spinner("Processing data..."):
            #     # Simulate some data processing
            #     time.sleep(1)
            #load image.png from contours
            # st.image('contours/image.png', width=100)
            zoom_factors = st.slider("Zoom:", 0.1, 3.0, 0.16)
            display_image1('contours/image.png', zoom_factors)
            # st.success("Done!")

        
        
        # Buttons to accept, reject, and flag
        st.header("Actions")

        # Side-by-side buttons
        col1, col2, col3 = st.columns(3)

        if col1.button("Accept"):
            # Handle Accept action here
            st.success("Vug% Accepted!")

        create_database()

        if col2.button("Reject"):
            # Handle Reject action here
            st.warning("Vug% Rejected!")
            st.subheader("Enter Estimated Vug%:")
            corrected_vug = st.number_input("Vug%", value=0, min_value=0, max_value=100)
            submit_reject = st.button("Submit Rejection", key="submit_reject")  # Unique key for the button
            if submit_reject:
                # Save the estimated vug% to the SQLite3 database
                image_id = uploaded_file.name  # Make sure 'uploaded_file' is accessible here
                save_vug_percent_to_db(image_id, corrected_vug)
                st.info(f"Data saved in the database: Image ID: {image_id}, Corrected Vug%: {corrected_vug}")

        if col3.button("Flag"):
            # Handle Flag action here
            st.info("Image, Well Name, and Reported Vug% flagged for verification.")
            
        

        # Generate report
        st.header("Generate Report")
        if st.button("Generate Report"):
            # Get the corrected vug% if applicable
            corrected_vug = None
            if col2.button("Reject", key="reject_button"):
                # Handle Reject action here
                st.warning("Vug% Rejected!")
                st.subheader("Enter Estimated Vug%:")
                corrected_vug = st.number_input("Vug%", value=0, min_value=0, max_value=100)
                submit_reject = st.button("Submit Rejection", key="submit_reject")  # Unique key for the button
                if submit_reject:
                    # Save the estimated vug% to the SQLite3 database
                    image_id = uploaded_file.name  # Make sure 'uploaded_file' is accessible here
                    save_vug_percent_to_db(image_id, corrected_vug)
                    st.info(f"Data saved in the database: Image ID: {image_id}, Corrected Vug%: {corrected_vug}")


            # Generate graphs
            graphs = [display_graphs() for _ in range(3)]  # Modify the range based on the number of graphs you want

            # Generate the PDF report
            pdf_buffer = generate_report(uploaded_file, vug_percent, corrected_vug, graphs)

            # Provide the link to download the PDF
            download_link = get_binary_file_downloader_html(pdf_buffer, "Generated_Report.pdf", "Click here to download the report!")
            st.markdown(download_link, unsafe_allow_html=True)
        
        
        # CREDS_FILE = 'sturdy-tuner-393016-49517332d8cc.json'
        # SPREADSHEET_ID = '1HOY6C_agR5GtlGARll6MSN-e3_7thqs3eN638Hh_GIU'

        # def add_email_to_csv(email):
        #     file_exists = os.path.isfile("emails.csv")

        #     with open("emails.csv", "a", newline="") as file:
        #         writer = csv.writer(file)
        #         if not file_exists:
        #             writer.writerow(["Email"])
        #         writer.writerow([email])

        # def add_feedback_to_sheet(feedback):
        #     creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE)
        #     client = gspread.authorize(creds)
        #     sheet = client.open_by_key(SPREADSHEET_ID).sheet1
        #     sheet.append_row(feedback)

        # st.header("Feedback")
        
        # email = st.text_input("Enter your email for us to reach you out")
        # question1 = st.selectbox("How much will you rate the overall experience ? ", ["0 - 3", "4 - 6", "7 - 10"])
        # question2 = st.selectbox("Was all your needs satisfied ", ["Yes, totally.", "Could have been better.", "Not really"])
        # # ask any feedbacks on what can be improved text format
        # question3 = st.text_input("Any feedbacks on what can be improved ?")

        # if st.button("Submit Feedback"):
        #     feedback = [email, question1, question2, question3]
        #     add_feedback_to_sheet(feedback)
        #     st.success("Feedback has been stored!")

if __name__ == "__main__":
    main()
