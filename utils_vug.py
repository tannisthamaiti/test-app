import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import skew, kurtosis
import seaborn as sns
from os.path import join as pjoin
#from dlisio import dlis


def get_logical_file(dlis_file_name, data_path, dyn):
    """
    This function is used to get the logical file from the DLIS file and filter it based on the type of data (static or dynamic) provided by the user.

    Parameters
    ----------

    dlis_file_name : list
        List of DLIS file names
    data_path : str
        Path of the data folder
    dyn : bool
        True if dynamic data is required, False if static data is required

    Returns
    -------
    logical_file : dlisio.logicalfile
        Logical file containing the data of the type specified by the user

    """
    #get correct file name as per user input
    if len(dlis_file_name)>1:
        if dyn:
            fmi_file_name = [i for i in dlis_file_name if 'dyn' in i.lower()]
            print('dynamic dlis file name retrieved')
        else:
            fmi_file_name = [i for i in dlis_file_name if 'stat' in i.lower()]
            print('static dlis file name retrieved')
    elif len(dlis_file_name)==1:
        fmi_file_name = dlis_file_name
        print('common dlis file name retrieved')
    else:
        print('no dlis file found for this well')
        return

    dlis_path = pjoin(data_path, fmi_file_name[0])
    print("path for dlis file: {}".format(dlis_path))

    dlis_file = dlis.load(dlis_path)

    #get correct logical file as per user input
    if len(dlis_file)>1:
        if dyn:
            logical_file = [i for i in dlis_file if 'dyn' in str(i).lower()][0]
            print('dynamic logical file retrieved')
        else:
            logical_file = [i for i in dlis_file if 'stat' in str(i).lower()][0]
            print('static logical file retrieved')
    else:
        logical_file = dlis_file[0]
        print('logical file retrieved')

    return logical_file

def get_fmi_with_depth_and_radius(logical_file, dyn, reverse = True):
    """
    This function is used to get the FMI and CMI data from the logical file and filter it based on the type of data (static or dynamic) provided by the user.

    Parameters
    ----------
    logical_file : dlisio.logicalfile
        Logical file containing the data of the type specified by the user
    dyn : bool
        True if dynamic data is required, False if static data is required
    reverse : bool, optional
        True if the data is to be reversed (Deeper to shallower). The default is True.

    Returns
    -------
    fmi_array : numpy array
        Array containing the FMI and CMI data of the type specified by the user
    tdep_array : numpy array
        Array containing the depth data
    well_radius : numpy array
        Array containing the well radius data

    """
    for channel in logical_file.channels:
        if dyn:
            if channel.name in ['FMI_DYN', 'CMI_DYN', 'TB_IMAGE_DYN_DMS_IMG']:
                fmi_array = channel.curves()
                print('{} loaded'.format(channel.name))
                print("shape of {} array: {}".format(channel.name, fmi_array.shape))
        else:
            if channel.name in ['FMI_STAT', 'CMI_STAT', 'TB_IMAGE_STA_DMS_IMG', 'FMI_-STAT']:
                fmi_array = channel.curves()
                print('{} loaded'.format(channel.name))
                print("shape of {} array: {}".format(channel.name, fmi_array.shape))

        if channel.name in ['TDEP', 'MD', 'DEPTH']:
            tdep_array = channel.curves()
            print('{} loaded'.format(channel.name))
            print("shape of {} array: {}".format(channel.name, tdep_array.shape))
            print("original depth array: {}".format(tdep_array))
            print("unit of the given depth is {}".format(channel.units))

            if channel.units.endswith('in'):
                tdep_array = inch_to_meter(tdep_array)
                print("depth array after conversion: {}".format(tdep_array))
            else:
                if tdep_array[0]>5000:
                    tdep_array = inch_to_meter(tdep_array)
                    print("depth array after conversion: {}".format(tdep_array))

    channels = logical_file.channels
    channle_names = [i.name for i in channels]

    if 'ASSOC_CAL' in channle_names:
        print('getting radius from assoc_cal channel')
        radius_channel = channels[channle_names.index('ASSOC_CAL')]
        well_diameter = radius_channel.curves()
        well_diameter[well_diameter == -9999.] = np.nan
        if 'in' in radius_channel.units:
            well_diameter = inch_to_meter(well_diameter, radius = True)
        well_radius = well_diameter/2
        print(f"well radius: {well_radius}")

    else:
        caliper_log = ['C1_13', 'C1_24', 'C2_13', 'C2_24', 'C3_13', 'C3_24']
        well_diameter = np.asarray([channels[channle_names.index(i)].curves() for i in caliper_log])
        well_diameter[well_diameter == -9999.] = np.nan
        well_diameter = well_diameter.mean(axis=0)
        if 'in' in channels[channle_names.index(caliper_log[0])].units:
            well_diameter = inch_to_meter(well_diameter, radius = True)
        well_radius = well_diameter/2
        print(f"well radius: {well_radius}")

    if reverse:
        if tdep_array[0]<tdep_array[-1]:
            print("data retrieved from dlis file is in reverse format that what's expected, reversing the"\
                " tdep_array & fmi_array to match what's expected by the code")
            fmi_array = fmi_array[::-1]
            tdep_array = tdep_array[::-1]
            well_radius = well_radius[::-1]
            print("depth array after temporarily reversing the array: {}".format(tdep_array))
    else:
        if tdep_array[0]>tdep_array[-1]:
            fmi_array = fmi_array[::-1]
            tdep_array = tdep_array[::-1]
            well_radius = well_radius[::-1]
            print("depth array after temporarily reversing the array: {}".format(tdep_array))
    

    return fmi_array, tdep_array, well_radius

def inch_to_meter(tdep_array, radius = False):
    """
    converts the depth values from inches to meters.

    parameters
    ----------
    tdep_array : numpy array
        the depth values in inches.
    radius : bool, optional
        if True, the depth values are in meters. the default is False.

    returns
    -------
    tdep_array : numpy array
        the depth values in meters.
    """

    print('converting inch to meters')
    if not radius:
        tdep_array = tdep_array/10
    depth_ft = tdep_array*0.0833333
    tdep_array = depth_ft*0.3048
    return tdep_array

def get_centeroid(cnt):
    """Get centroid from a given contour"""
    length = len(cnt)
    sum_x = np.sum(cnt[..., 0])
    sum_y = np.sum(cnt[..., 1])
    return int(sum_x / length), int(sum_y / length)

def MinMaxScalerCustom(X, min = 0, max = 1):
    """
    Scales the input data between the specified min and max values.

    Parameters
    ----------
    X : numpy array
        The input data.
    min : int, optional
        The minimum value. The default is 0.
    max : int, optional
        The maximum value. The default is 1.

    Returns
    -------
    X_scaled : numpy array
        The scaled data.
    """
    X_std = (X - np.nanmin(X)) / (np.nanmax(X) - np.nanmin(X))
    X_scaled = X_std * (max - min) + min
    return X_scaled

def get_depth_from_pixel(val, scaled_max, scaled_min, raw_max, raw_min=0):
    """
    Converts the pixel value to depth value.

    Parameters
    ----------
    val : int
        The pixel value.
    scaled_max : int
        The maximum depth value.
    scaled_min : int
        The minimum depth value.
    raw_max : int
        The maximum pixel value.
    raw_min : int, optional
        The minimum pixel value. The default is 0.

    Returns
    -------
    int
        The depth value.
    """
    return ((scaled_max-scaled_min)/raw_max) * (val-raw_min) + scaled_min

def plot_contours(circles, ax, linewidth=2):
    """Plot contours on ax
    
    Parameters
    ----------
    circles : list
        List of contours
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes to plot on

    Returns
    -------
    None
    """
    for pts in circles:
        x = pts[:, 0, 0]
        y = pts[:, 0, 1]
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        ax.plot(x, y, color='black', linewidth=linewidth)

def combine_two_contours(c1, c2):
    """combines two contours/centroids
    """
    combined_contour = []
    for c in c1:
        combined_contour.append(c)
    for c in c2:
        combined_contour.append(c)
    return combined_contour

def get_every_nth_element(array, n):
    """Get every nth element from array

    Parameters
    ----------
    array : list
        List of elements
    n : int
        nth element

    Returns
    -------
    list
        List of every nth element
    """
    return array[::n]

def combine_two_centroids(centroids1, centroids2):
    """Combine two centroids and return combined centroids
    takes
    centroids1: list of centroid1
        len(centroid1) = n
    centroids2: list of centroid2
        len(centroid2) = m

    return:
        combined_centroids: list of combined centroids
    """
    combined_centroids = []
    for centroid1 in centroids1:
        combined_centroids.append(list(centroid1))
    for centroid2 in centroids2:
        combined_centroids.append(list(centroid2))
    return combined_centroids

def filter_contours_based_on_original_image(contours, vugs, original_image, thresold):
    """"Filters the contours based on the original image
        For every contour, it checks the contrast of the original image, if the contrast is high, it is a valid contour
        else it is a false positive and removed from the list of contours
        and icrease the bounding box by 10% in all direction

    Parameters
    ----------
    contours : list
        list of contours, each element of contours is a numpy array of shape (N, 1, 2) which represents a cotour of polygon, 
            it is not a rectangle
    vugs : list
        list of vugs, each element of vugs is a dictionary with keys 'id', 'area', 'depth', 'centroid_x', 'centroid_y'
    original_image : numpy array
        original image
    thresold : int
        thresold value for the contrast

    Returns
    -------
    filtered_contours : list
        list of contours, each element of contours is a numpy array of shape (N, 1, 2) which represents a cotour of polygon
    filtered_vugs : list
        list of vugs, each element of vugs is a dictionary with keys 'id', 'area', 'depth', 'centroid_x', 'centroid_y'
    """
    filtered_contours = []
    filtered_vugs = []

    # for every contour, get the bounding box and check the contrast of the bounding box
    for contour, vug in zip(contours, vugs):

        # get the bounding box for each contour
        x, y, w, h = cv.boundingRect(contour)

        # increase the bounding box by 10% in all direction
        x -= int(0.1*w)
        y -= int(0.1*h)
        w += int(0.2*w)
        h += int(0.2*h)
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x+w > original_image.shape[1]:
            w = original_image.shape[1] - x
        if y+h > original_image.shape[0]:
            h = original_image.shape[0] - y
        roi = original_image[y:y+h, x:x+w]
        roi = np.float32(roi)
        
        contrast = cv.Laplacian(roi, cv.CV_32F, ).var()
        if contrast > thresold:
            filtered_contours.append(contour)
            filtered_vugs.append(vug)
    return filtered_contours, filtered_vugs

def compare_two_centroids(reference_centroid, main_centroid, threshold):
    """Compare two centroids and return index of those centroid which are beyond threshold distance
    takes
    reference_centroid: list of refrence centroid from which comparison is to be fone
        len(centroid1) = n
    main_centroid: list of main centroid in which nearby or duplicate centroid is to be found and removed
        len(centroid2) = m

    return:
        idx: index of main_centroid which are beyond threshold distance from reference_centroid
    """
    idx = []
    if len(reference_centroid) != 0:
        for i, centroid in enumerate(main_centroid):
            distance = [np.linalg.norm(np.asarray(centroid) - np.asarray(ref_centroid)) for ref_centroid in reference_centroid]
            if min(distance) > threshold:
                idx.append(i)
    else:
        idx = list(range(len(main_centroid)))
    return idx

def calculate_statistical_features(img):
    """calculate the statistical features of the image."""

    img = img.reshape(-1)
    img = img[~np.isnan(img)]
    mean = np.mean(img)
    std = np.std(img)
    var = np.var(img)
    skewness = skew(img.reshape(-1))
    kurtosis_val = kurtosis(img.reshape(-1))
    return img.min(), img.max(), np.float16(mean), np.float16(std), np.float16(var), np.float16(skewness), np.float16(kurtosis_val)

def filter_contours_based_on_statistics_of_image(
        contours, 
        vugs, 
        std, 
        var, 
        skew, 
        kurt, 
        mins,
        maxs,
        means,
        std_thes=3.0, 
        var_thes=5.0, 
        skew_thes=8.0, 
        kurt_thes=70.0, 
        skew_thes_low=0.0, 
        kurt_thes_low=0.0,
        min_high = 49, 
        max_low = 75, 
        means_low = 60,):
    """Filter contours based on statistics of image, if the statistics of image doesn't satisfy the codition then contour is removed
    if std is greater than 3, then contour is removed
    if var is greater than 5, then contour is removed
    if skew is negative or greater than 8, then contour is removed
    if kurt is greater than 70, then contour is removed

    Parameters
    ----------
    contours: list of contours
    vugs: list of vugs
    std: standard deviation of image
    var: variance of image
    skew: skewness of image
    kurt: kurtosis of image

    Returns
    -------
        filtered_contours: list of filtered contours
        filtered_vugs: list of filtered vugs
    """
    if std >= std_thes:
        return [], []
    if var >= var_thes:
        return [], []
    if mins<=min_high:
        return [], []
    if maxs>=max_low:
        return [], []
    if means>=means_low:
        return [], []
    if skew <= skew_thes_low or skew >= skew_thes:
        return [], []
    if kurt <= kurt_thes_low or kurt >= kurt_thes:
        return [], []
    return contours, vugs

def turn_off_particular_subplot(ax, which='all'):
    """Turn off particular subplot

    Parameters
    ----------
    ax: matplotlib axes
    which: str
        'all': turn off all
        'x': turn off x axis
        'y': turn off y axis
    """
    if which == 'all':
        ax.set_axis_off()
    elif which == 'x':
        ax.xaxis.set_visible(False)
    elif which == 'y':
        ax.yaxis.set_visible(False)
    else:
        raise ValueError("which should be 'all', 'x' or 'y'")

def get_one_meter_fmi_and_GT(one_meter_zone_start, one_meter_zone_end, tdep_array_doi, fmi_array_doi, well_radius, gt):
    """
    Returns the fmi and GT for a one meter zone
    
    Parameters
    ----------
    
    one_meter_zone_start : int
        Start of the one meter zone
    one_meter_zone_end : int
        End of the one meter zone
    tdep_array_doi : numpy array
        Time dependent array of depth of interest
    fmi_array_doi : numpy array
        FMI array of depth of interest
    well_radius : numpy array
        Radius of the well
    gt : pandas dataframe
        Ground truth dataframe
        
    Returns
    -------
        
    fmi_array_one_meter_zone : numpy array
        FMI array of one meter zone
    tdep_array_one_meter_zone : numpy array
        Time dependent array of one meter zone
    gtZone : pandas dataframe
        Ground truth dataframe of one meter zone
        
    """
    mask_one_meter_zone = (tdep_array_doi>=one_meter_zone_start) & (tdep_array_doi<one_meter_zone_end)
    
    fmi_array_one_meter_zone = fmi_array_doi[mask_one_meter_zone]
    tdep_array_one_meter_zone = tdep_array_doi[mask_one_meter_zone]
    well_radius_one_meter_zone = well_radius[mask_one_meter_zone]
    gtZone = gt[(gt.Depth>=one_meter_zone_start) & (gt.Depth<one_meter_zone_end)]
    return fmi_array_one_meter_zone, tdep_array_one_meter_zone, well_radius_one_meter_zone, gtZone

def get_mode_of_interest_from_image(fmi_array_one_meter_zone, stride_mode, k):
    """
    Returns the mode of interest from the image
    
    Parameters
    ----------
    
    fmi_array_one_meter_zone : numpy array
        FMI array of one meter zone
    stride_mode : int
        Stride for the mode of interest
    k : int
        Number of modes to return
        
    Returns
    -------
        
    mode_of_interest : list
        List of modes of interest
        
    """
    value, counts = np.unique(fmi_array_one_meter_zone[~np.isnan(fmi_array_one_meter_zone)
                                                    ].astype(np.float16), return_counts=True)
    idx = np.argsort(counts)[::-1]
    mode_of_interest = get_every_nth_element(value[idx], stride_mode)[:k]
    return mode_of_interest

def apply_adaptive_thresholding(fmi_array_one_meter_zone, moi, block_size = 21, c = 'mean'):
    """
    Returns the thresholded image

    Parameters
    ----------

    fmi_array_one_meter_zone : numpy array
        FMI array of one meter zone
    moi : int
        Mode of interest
    block_size : int
        Block size for adaptive thresholding
    c : int or str
        Constant for adaptive thresholding. If 'mean', then the mean of the image is used
    
    Returns
    -------

    thresold_img : numpy array
        Thresholded image

    """

    fmi_array_one_meter_mode_subtracted = fmi_array_one_meter_zone-moi
    fmi_array_one_meter_mode_subtracted = cv.convertScaleAbs(fmi_array_one_meter_mode_subtracted)
    if isinstance(c, str):
        C = fmi_array_one_meter_mode_subtracted.mean()
    else:
        C = c
    thresold_img = cv.adaptiveThreshold(fmi_array_one_meter_mode_subtracted, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv.THRESH_BINARY, block_size, C)
    return thresold_img

def plot_distribution_of_fmi(fmi_array_one_meter_zone, 
                             fmi_array_doi, 
                             one_meter_zone_start, 
                             mean, 
                             std, 
                             var, 
                             skewness, 
                             kurtosis_val, 
                             fig_size=(15, 2),
                             save = True):
    """
    Plots the distribution of FMI

    Parameters
    ----------

    fmi_array_one_meter_zone : numpy array
        FMI array of one meter zone
    fmi_array_doi : numpy array
        FMI array of depth of interest
    one_meter_zone_start : int
        Start of the one meter zone
    mean : float
        Mean of the FMI array
    std : float
        Standard deviation of the FMI array
    var : float
        Variance of the FMI array
    skewness : float
        Skewness of the FMI array
    kurtosis_val : float
        Kurtosis of the FMI array
    fig_size : tuple
        Figure size
        
    Returns
    -------

    None

    """
    plt.figure(figsize=fig_size)
    sns.histplot(fmi_array_one_meter_zone.reshape(-1))
    plt.title(f"Mean: {mean}, Std: {std}, Var: {var}, Skewness: {skewness}, Kurtosis: {kurtosis_val}")
    plt.xlim(np.nanmin(fmi_array_doi), np.nanmax(fmi_array_doi))
    plt.tight_layout()
    if save:
        plt.savefig(f"hist/{one_meter_zone_start}.png", dpi=400)
        plt.close()
    else:
        plt.show()

def convert_an_array_to_nearest_point_5_or_int(array, nearest_point_5=True):
    """
    Converts an array to nearest point 5 or int

    Parameters
    ----------

    array : numpy array
        Array to be converted
    nearest_point_5 : bool
        If True, then the array is converted to nearest point 5. If False, then the array is converted to nearest int

    Returns
    -------

    array : numpy array
        Converted array

    """
    if nearest_point_5:
        return np.round(array*2)/2
    else:
        return np.round(array)

def get_outliers_from_image(fmi_array_one_meter_zone, std_no, k, dtype=np.uint8, nearest_point_5=True):
    """
    Returns the outliers from the image

    Parameters
    ----------

    fmi_array_one_meter_zone : numpy array
        FMI array of one meter zone
    std_no : int
        Number of standard deviations to consider as outliers
    k : int
        Number of outliers to return
    dtype : numpy dtype
        Data type of the returned array

    Returns
    -------

    outlier_sorted : numpy array
        Array of outliers

    """
    mean_image = np.nanmean(fmi_array_one_meter_zone)
    std_image = np.nanstd(fmi_array_one_meter_zone)

    z_score = (fmi_array_one_meter_zone - mean_image) / std_image
    outliers = fmi_array_one_meter_zone[np.abs(z_score) > std_no]
    outliers = outliers[outliers>mean_image].astype(dtype)

    outliers = convert_an_array_to_nearest_point_5_or_int(outliers, nearest_point_5=nearest_point_5)

    val_outlier, count_outlier = np.unique(outliers, return_counts=True)
    idx_outlier = np.argsort(count_outlier)[::-1]
    outlier_sorted = val_outlier[idx_outlier][:k]
    return outlier_sorted

def get_combined_contours_and_centroids(contours, centroids, vugs, combined_centroids, final_combined_contour, final_combined_vugs, i, threshold = 5):
    """
    Returns the combined contours and centroids

    Parameters
    ----------

    contours : list
        List of contours
    centroids : list
        List of centroids
    vugs : list
        List of vugs
    combined_centroids : list
        List of combined centroids
    final_combined_contour : list
        List of combined contours
    final_combined_vugs : list
        List of combined vugs
    i : int
        Index of the contours and centroids
    threshold : int
        Threshold for comparing the centroids
        
    Returns
    -------

    combined_centroids : list
        List of combined centroids
    final_combined_contour : list
        List of combined contours

    """
    if i == 0:
        combined_centroids = centroids
        final_combined_contour = contours
        final_combined_vugs = vugs
    else:
        idx = compare_two_centroids(combined_centroids, centroids, threshold)
        combined_centroids = combine_two_centroids(combined_centroids, centroids)
        contours = [contours[i] for i in idx]
        vugs = [vugs[i] for i in idx]
        final_combined_contour = combine_two_contours(final_combined_contour, contours)
        final_combined_vugs = combine_two_contours(final_combined_vugs, vugs)
    return combined_centroids, final_combined_contour, final_combined_vugs

def plot_threshold_axis(thresold_img, ax, title, fontsize = 8):
    """Plot the threshold image"""
    ax.imshow(thresold_img, cmap='gray')
    ax.set_title(title, fontsize = fontsize)

def plot_original_image_axis(image, start, end, ax, title, fontsize = 8):
    """Plot the original image"""
    ax.imshow(image, cmap='YlOrBr')
    ax.set_title(title, fontsize = fontsize)
    ax.set_yticks(np.linspace(0, image.shape[0], 10).round(2), np.linspace(start, end, 10).round(2))

def plot_contour_axis(image, contours, title, ax, fontsize = 8, linewidth = 2):
    """
    Plot the contours on the image
    """
    ax.imshow(image, cmap='YlOrBr')
    ax.set_title(title, fontsize = fontsize)
    plot_contours(contours, ax, linewidth = linewidth)

def turn_off_all_the_ticks(ax, except_ax):
    """Turn off all the ticks of the axis"""
    
    for i in range(len(ax)):
        if ax[i]!=except_ax:
            ax[i].set_xticks([])
            if i!=0:
                ax[i].set_yticks([])

def plot_barh(ax, y, x, one_meter_zone_start, one_meter_zone_end, title, max_scale=25, fontsize = 8, colors='k', linestyles='dashed', linewidth=1, vlines=5):
    """
    Plots the barh plot of the detected and GT vugs

    Parameters
    ----------

    ax: matplotlib axis
        axis on which the plot is to be plotted
    y: numpy array
        y axis of the plot
    x: numpy array
        x axis of the plot
    one_meter_zone_start: int
        start of the one meter zone
    one_meter_zone_end: int
        end of the one meter zone
    title: str
        title of the plot
    max_scale: int
        max scale of the plot
    fontsize: int
        fontsize of the plot
    colors: str
        color of the plot
    linestyles: str
        linestyles of the plot
    linewidth: int
        linewidth of the plot

    Returns
    -------
    None
    """
    ax.barh(y, x, align='center', height=0.08)
    for i in range(vlines, 100, vlines):
        ax.vlines(i, one_meter_zone_start, one_meter_zone_end, colors=colors, linestyles=linestyles, linewidth=linewidth)
    ax.invert_yaxis()
    ax.set_xlim(0, max_scale)
    ax.set_ylim(one_meter_zone_end, one_meter_zone_start)
    ax.set_title(title, fontsize = fontsize)

def convert_vugs_to_df(filtered_vugs):
    """converts the vugs to dataframe
    dataframe should consist two columns, depth and area

    Parameters
    ----------
    filtered_vugs : List
        Filtered vugs is a list of vugs. Each element of list is a dictionary of vugs with keys being 'id', 'area', 'depth', 'centroid_x', 'centroid_y'

    Returns
    -------
    df : DataFrame
        dataframe of vugs which contain two columns, depth and area
    """
    df = pd.DataFrame({'depth': [v['depth'] for v in filtered_vugs], 'area': [v['area'] for v in filtered_vugs]})
    df = df.sort_values(by=['depth'])
    return df

def convert_df_to_zone(df, start, end, zone_len):
    """converts the dataframe to zone. 
    


    Parameters
    ----------
    df : DataFrame
        df contains two columns, depth and area.
        depth is in meters and it's value can range anywhere between start and end.
    start : int
        actual start depth of the df, which is not necessarily present
        df can start from 2652.2 and end at 2652.4, but start can be 2652.0
    end : int
        actual end depth of the df, which is not necessarily present
        df can start from 2652.2 and end at 2652.4, but end can be 2653.0
    zone_len : int
        length of the zone, this is the zone in which df needs to be recreated
        if zone_len is 0.1 then df will be recreated in 0.1 meter depth intervals
        and area will be summed up for each 0.1 meter depth interval

    Returns
    -------
    zone : DataFrame
        dataframe of vugs which contain two columns, depth and area
    """
    zone_depth = np.arange(start, end, zone_len)
    zone = pd.DataFrame()
    for i in zone_depth:
        start, end = i, i+0.1
        area = df[(df.depth>=start) & (df.depth<end)]['area'].sum()
        zone = pd.concat([zone, pd.DataFrame({'depth': [start], 'area': [area]})], ignore_index=True)
    return zone

def detected_percent_vugs(filtered_contour, fmi_array_one_meter_zone, tdep_array_one_meter_zone, one_meter_zone_start, one_meter_zone_end):
    """calculates the percentage of vugs detected in the one meter zone by the contours

    Parameters
    ----------

    filtered_contour : List
        Filtered contour is a list of contours. Each element of list is a dictionary of contours with keys being 'id', 'area', 'depth', 'centroid_x', 'centroid_y'
    fmi_array_one_meter_zone : numpy array
        fmi array of one meter zone
    tdep_array_one_meter_zone : numpy array
        tdep array of one meter zone
    one_meter_zone_start : int
        start of the one meter zone
    one_meter_zone_end : int
        end of the one meter zone

    Returns
    -------
    df : DataFrame
        dataframe of vugs which contain two columns, depth and area
    """
    height, width = fmi_array_one_meter_zone.shape
    
    blank_im = np.zeros((height, width))
    blank_im[blank_im==0]=255

    cv.drawContours(blank_im, filtered_contour, -1, (0,255,52), -1)
    df = pd.DataFrame()
    for start in np.arange(one_meter_zone_start, one_meter_zone_end, 0.1):
        end = start + 0.1
        mask = (tdep_array_one_meter_zone>=start) & (tdep_array_one_meter_zone<end)
        
        blank_im_point_one_meter_zone = blank_im[mask]
        
        white_px = cv.countNonZero(blank_im_point_one_meter_zone)
        total_px = blank_im_point_one_meter_zone.shape[0] * width
        df = pd.concat([df, pd.DataFrame({'Depth': [start], 'Vugs': [(1 - white_px/total_px)*100]})], ignore_index=True)
    return df

def check(value, blind_depth_range):
    """
    checks if the value is within the blind depth range.
    
    parameters
    ----------
    value : float
        the value to be checked.
    blind_depth_range : tuple
        the blind depth range.

    returns
    -------
    bool
        True if the value is within the blind depth range, False otherwise.
    """
    
    if blind_depth_range.start <= value <= blind_depth_range.stop:
        return True
    return False

def get_contours(thresold_img, depth_to, depth_from, radius, pix_len, 
                 min_vug_area = 1.2, max_vug_area = 10.28, min_circ_ratio=0.5, max_circ_ratio=1):
    """Get contours from thresholded image
    
    Parameters
    ----------
    thresold_img : np.array
        Thresholded image
    depth_to : int
        Depth to
    depth_from : int
        Depth from
    radius : int
        Radius
    pix_len : int
        1px == how much cm
    min_vug_area : float, optional
        Minimum vug area, by default 1.2
    max_vug_area : float, optional
        Maximum vug area, by default 10.28
    min_circ_ratio : float, optional
        Minimum circularity ratio, by default 0.5
    max_circ_ratio : float, optional
        Maximum circularity ratio, by default 1

    Returns
    -------
    contours : list
        List of contours
    centroids : list
        List of centroids
    vugg : list
        List of vuggs
    """
    height, width = thresold_img.shape
    contours, _ = cv.findContours(thresold_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE, )
    circles = []
    centroids = []
    ix = 1          
    total_vugg = []
    HOLE_R = radius
    PIXEL_LEN = pix_len
    PIXLE_SCALE = np.radians(1) * HOLE_R * PIXEL_LEN 
    for _, contour in enumerate(contours):
        epsilon = 0.01 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        area = cv.contourArea(contour) * PIXLE_SCALE
        (x,y), radius = cv.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = int(radius)
        area_enclosing = np.pi * radius**2 * PIXLE_SCALE
        if len(approx) >= 3 and \
                min_vug_area < area <= max_vug_area and \
                min_circ_ratio <= (area/area_enclosing) <= max_circ_ratio:
            circles.append(contour)
            centroids.append(get_centeroid(contour))
            y = get_centeroid(contour)[1]
            x = get_centeroid(contour)[0]
            ix +=1
            vugg = {'id':ix, 
                    'area': area, 
                    'depth': get_depth_from_pixel(val=height-y, 
                                                  scaled_max=depth_to, 
                                                  scaled_min=depth_from, 
                                                  raw_max=height), 
                    'centroid_x':x, 
                    'centroid_y': y}
            total_vugg.append(vugg)
    return circles, centroids, total_vugg

def remove_contour_from_bedding_planes(fmi, 
                                       tdep, 
                                       zone_start, 
                                       zone_end, 
                                       thresh, 
                                       filtered_contour, 
                                       filtered_vugs, 
                                       zoi = 0.05, 
                                       thresh_percent = 15):
    """
    Remove Vugs from bedding planes

    Parameters
    ----------
    fmi : np.ndarray
        FMI data
    tdep : np.ndarray
        TDEP data
    zone_start : float
        Starting depth of the zone
    zone_end : float
        Ending depth of the zone
    thresh : float
        Threshold value that is selected based on first mode
    filtered_contour : list
        List of contours
    filtered_vugs : list
        List of vugs
    zoi : float, optional
        Zone of interest length, by default 0.05

    Returns
    -------
    None
    """
    total_starts = []
    start_ = 0
    end_ = 0
    total_percent = []
    drop_idx = []
    for i, zoi_start in enumerate(np.arange(zone_start, zone_end, zoi)):
        zoi_end = zoi_start + zoi
        mask_zoi = (tdep>=zoi_start) & (tdep<zoi_end)
        fmi_zoi = fmi[mask_zoi]
        tdep_zoi = tdep[mask_zoi]
        mins, maxs, mean, std, var, skewness, kurtosis_val = calculate_statistical_features(fmi_zoi)
        # print(f"{std} | {var} | {skewness} | {kurtosis_val}: {zoi_start}-{zoi_end}")
        start_=end_
        end_+=tdep_zoi.shape[0]
        # numerator_mask = np.logical_and(fmi_zoi>=thresh.min(), fmi_zoi<=thresh.max())
        numerator_mask = np.logical_and(fmi_zoi>=thresh[0]-0.1, fmi_zoi<=thresh[0]+0.1)
        percent = (numerator_mask.sum()/(fmi_zoi.reshape(-1).shape[0]))*100
        total_percent.append(percent)
        total_starts.append(zoi_start)

        filtering_required = True if percent<=thresh_percent else False
        if filtering_required:
            for i, (cnt, vgs) in enumerate(zip(filtered_contour, filtered_vugs)):
                cnt_y = cnt[:, 0, :][:, 1]
                cnt_y_range = range(cnt_y.min(), cnt_y.max())
                if sum([check(i, cnt_y_range) for i in range(start_-1, end_+1)]) != 0:
                    second_condition = filter_contour_based_on_bedding_plane(cnt, tdep, fmi, k = 2, zoi = 0.1, low = -1, high = 1)
                    if second_condition:
                        drop_idx.append(i)

    filtered_contour = [j for i, j in enumerate(filtered_contour) if i not in drop_idx]
    filtered_vugs = [j for i, j in enumerate(filtered_vugs) if i not in drop_idx]

    return total_percent, total_starts, filtered_contour, filtered_vugs

def filter_contour_based_on_bedding_plane(cnt, tdep_array_one_meter_zone, fmi_array_one_meter_zone, k = 2, zoi = 0.1, low = -0.1, high = 0.1):
    filtering_required = False
    cnt_x = cnt[:, 0, :][:, 0]
    cnt_y = cnt[:, 0, :][:, 1]
    cnt_y_min, cnt_y_max = cnt_y.min(), cnt_y.max()
    cnt_x_min, cnt_x_max = cnt_x.min(), cnt_x.max()
    cnt_dept = tdep_array_one_meter_zone[cnt_y_min:cnt_y_max]
    window_size = int(np.ceil(zoi/(cnt_dept[1:] - cnt_dept[:-1]).mean()))
    cnt_heigt = cnt_y_max - cnt_y_min
    cnt_width = cnt_x_max - cnt_x_min

    padding_required_height = window_size - cnt_heigt
    padding_required_width = window_size - cnt_width
    pad_top_bottom = int(np.ceil(padding_required_height/2))
    pad_left_right = int(np.ceil(padding_required_width/2))
    cnt_y_min = cnt_y_min - pad_top_bottom if cnt_y_min - pad_top_bottom > 0 else 0
    cnt_y_max = cnt_y_max + pad_top_bottom if cnt_y_max + pad_top_bottom < tdep_array_one_meter_zone.shape[0] else tdep_array_one_meter_zone.shape[0]
    cnt_x_min = cnt_x_min - pad_left_right if cnt_x_min - pad_left_right > 0 else 0
    cnt_x_max = cnt_x_max + pad_left_right if cnt_x_max + pad_left_right < 360 else 360
    fmi_window = fmi_array_one_meter_zone[cnt_y_min:cnt_y_max, :]
    fmi_contour = fmi_window[:, cnt_x_min:cnt_x_max]

    cnt_x_left_max = cnt_x_min
    cnt_x_left_min = cnt_x_min - fmi_contour.shape[1]*k if cnt_x_min - fmi_contour.shape[1]*k > 0 else 0

    cnt_x_right_min = cnt_x_max
    cnt_x_right_max = cnt_x_max + fmi_contour.shape[1]*k if cnt_x_max + fmi_contour.shape[1]*k < 360 else 360

    left, right = cnt_x_left_max - cnt_x_left_min, cnt_x_right_max - cnt_x_right_min
    optmial_pad = min(left, right)

    if cnt_x_min > 20 and cnt_x_max < 340:
        if left>right:
            fmi_contour_right = fmi_window[:, cnt_x_right_min:cnt_x_right_max]
            fmi_contour_left = fmi_window[:, cnt_x_left_max-optmial_pad:cnt_x_left_max]
        else:
            fmi_contour_left = fmi_window[:, cnt_x_left_min:cnt_x_left_max]
            fmi_contour_right = fmi_window[:, cnt_x_right_min:cnt_x_right_min+optmial_pad]
        col_wise_diff_mean = np.nanmean(np.mean(fmi_contour_left, axis = 0) - np.mean(fmi_contour_right, axis = 0))
        if low < col_wise_diff_mean < high:
            filtering_required = True
    return filtering_required

def get_elements_from_circular_roi_based_on_center_and_radius(fmi, center, radius):
    """
    Get elements of fmi image from circular ROI based on center and radius without nans

    Parameters
    ----------
    fmi : np.ndarray
        FMI data
    center : tuple
        Center of the circle
    radius : int
        Radius of the circle

    Returns
    -------
    roi : np.ndarray
        Elements from circular ROI
    """
    
    h, w = fmi.shape

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Calculate the Euclidean distance
    distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Create a mask
    mask = distances <= radius

    # Apply the mask to the image
    roi = np.zeros_like(fmi)
    roi[mask] = fmi[mask]

    # drop 0s from the numpy array
    roi = roi[roi!=0]

    # drop nans from the numpy array
    roi = roi[~np.isnan(roi)]

    return roi

def filter_contour_based_on_mean_pixel_in_and_around_original_contour(fmi, contours, vugs, threshold = 1):
    """
    Filter contours based on the mean pixel in and around the original contour

    Parameters
    ----------
    fmi : np.ndarray
        FMI data
    contours : list
        List of contours
    vugs : list
        List of vugs
    threshold : int, optional
        Threshold value, by default 1

    Returns
    -------
    filtered_contour : list
        List of filtered contours
    filtered_vugs : list
        List of filtered vugs
    """
    
    filtered_contour = []
    filtered_vugs = []

    # iterate over each contour
    for contour_test, vugs_test in zip(contours, vugs):

        # get the center and radius of the contour
        (x,y),radius = cv.minEnclosingCircle(contour_test)
        center = (int(x),int(y))
        radius = int(radius)

        # Create a binary mask based on the contour coordinates
        mask = np.zeros_like(fmi, dtype=np.uint8)
        contour_points = np.array(contour_test)

        # fill the contour with white color
        mask = cv.fillPoly(mask, [contour_points], 255)

        # get the pixels from the contour
        contour_pixels = fmi[mask > 0]

        # drop nans from the numpy array
        contour_pixels = contour_pixels[~np.isnan(contour_pixels)]

        # get the pixels from the bounding padded circle
        bounding_padded_circle = get_elements_from_circular_roi_based_on_center_and_radius(fmi, center, radius*2)

        # get the mean of the pixels from the contour and bounding padded circle
        contour_mean = contour_pixels.mean()
        bounding_padded_circle_mean = bounding_padded_circle.mean()

        # if the difference between the mean of the pixels from the contour and bounding padded circle is greater than 1 then keep the contour
        if abs(contour_mean - bounding_padded_circle_mean)>=threshold:
            filtered_contour.append(contour_test)
            filtered_vugs.append(vugs_test)
            # print(contour_mean - bounding_padded_circle_mean)
        else:
            # print('Oh Shit!!!')
            pass
        
    return filtered_contour, filtered_vugs

def plot_sinusoids(DOI, dep, ax, linewidth, bbColor, Fcolor):

    for row in DOI.iterrows():
        row = row[1]
        amp, phase = calculateAmpPhase(row["Dip"],row["Azimuth"],0.108)
        x_line, y_line = createSinusoid(amp,phase)
        y_line = y_line + row.Depth
        new_y = []
        for i in y_line:
            new_y.append(find_nearest(dep, i))
        if row.Type == 'Bb':
            ax.plot(new_y, bbColor, linewidth = linewidth)
        else:
            ax.plot(new_y, Fcolor, linewidth = linewidth)

def tadpolePlotGTComparison(df, ax, sinTypeStart, color, scaler):

    for rows in df.iterrows():
        if rows[1].Type.lower().startswith(sinTypeStart.lower()):

            ax.plot(rows[1].Dip, int(scaler.transform([[rows[1].Depth]])), marker = '.', markersize = 10, 
                    color = color)
            ax.plot(rows[1].Dip, int(scaler.transform([[rows[1].Depth]])), marker = (1, 2, -rows[1].Azimuth), 
                    markersize = 40, color = color)

def comparison_plot(fmiZone, tdepZone, zoneStart, gtZone,  predZone, zoneEnd, tadpoleScaler, fmiRatio, fontSize, linewidth, 
                    save_path, dpi = 50, figsize = (20, 25), save = True, split = False):
    _, ax = plt.subplots(1, 4, sharey = True, figsize = figsize,
                         gridspec_kw = {'width_ratios': [fmiRatio, fmiRatio, fmiRatio, 1]})
    
    ax[0].imshow(fmiZone, cmap = 'YlOrBr')
    ax[1].imshow(fmiZone, cmap = 'YlOrBr')
    ax[2].imshow(fmiZone, cmap = 'YlOrBr')
    ax[3].imshow(np.zeros((fmiZone.shape[0], 90)), cmap = 'gray', vmin = -10, vmax = 0)
    
    plt.yticks(np.linspace(0, tdepZone.shape[0], 10), np.linspace(zoneStart, zoneEnd, 10).round(2))
    ax[0].tick_params(axis = 'y', labelsize = fontSize + 5)

    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks([])
    ax[3].set_xticks([0, 30, 60, 90])
    ax[3].tick_params(axis = 'x', labelsize = fontSize - 5)
    print ("App working1")
    vlines = list(range(0, 100, 10))
    ax[3].vlines(vlines, ymin = 0, ymax = tdepZone.shape[0] - 1, linestyles = 'dotted')
    ax[3].set_xlim(-10, 100)
    
    plot_sinusoids(gtZone, tdepZone, ax[1], linewidth, 'red', 'black')
    plot_sinusoids(predZone, tdepZone, ax[2], linewidth, 'green', 'blue')
    tadpolePlotGTComparison(gtZone, ax[3], 'F', 'black', tadpoleScaler)
    tadpolePlotGTComparison(predZone, ax[3], 'F', 'blue', tadpoleScaler)

    print ("App working2")
    ax[0].set_title('Input Image', fontsize = fontSize)
    ax[1].set_title('Ground Truth\nBb: Green\nFrac: Blue', fontsize = fontSize)
    ax[2].set_title('Prediction\nBb: Green\nFrac: Blue', fontsize = fontSize)
    ax[3].set_title('Fracture\nGT: Black\nPred: Blue', fontsize = fontSize)
    print ("App working3")
    
    plt.tight_layout()
    if save:
        if split:
            fname = save_path
        else:
            fname = pjoin(save_path, '{}m - {}m.pdf'.format(round(tdepZone[0], 2), round(tdepZone[-1], 2)))
        plt.savefig(fname, format = 'pdf', dpi = dpi, bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()
def inchToMeter(tdep_array):
    print('Converting inch to meters')
    depth_in = tdep_array/10
    depth_ft = depth_in*0.0833333
    tdep_array = depth_ft*0.3048
    return tdep_array

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def calculateAmpPhase(dip_deg,azi_deg,rwell):
    dip_rad = dip_deg*(np.pi/180)
    amp = np.tan(dip_rad)*rwell
    azi_rad = azi_deg*(np.pi/180)
    phase =  np.pi/2 - azi_rad
    return amp, phase  

def createSinusoid(amp,phase):
    w=0.0175
    fitfunc = lambda t:  amp*np.sin(w*t + phase)
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.arange(0, 360,1) 
    # calculate the output for the range
    y_line = fitfunc(x_line)
    return x_line, y_line 