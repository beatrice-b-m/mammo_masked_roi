# img masking pipeline

import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import queue
import threading
from tqdm import tqdm
from numba import jit
import json


class DataFrameMasker:
    def __init__(self, pos_df, neg_df):
        self.pos_df = pos_df
        self.pos_out_df = None
        self.neg_df = neg_df
        self.neg_out_df = None
        self.n_images = len(pos_df)
        self.save_dir = None
        self.image_queue = None
        
        # initialize cols/rows arrays to record tissue distributions
        # make them larger than we need, then crop to max img cols/rows after opening all imgs
        self.cols_array = np.zeros((self.n_images, 6000), np.int16)
        self.rows_array = np.zeros((self.n_images, 6000), np.int16)
        self.crop_cols_array = None
        self.crop_rows_array = None
        
        # initialize roi dictionary to record ROIs indexed against the cols/rows arrays
        self.roi_dict = {}
        
        # initalize max n rows/cols vars to track largest seen images for cropping
        self.max_n_rows = 0
        self.max_n_cols = 0
        
    def start(self, mode, mask_factor_list, target_size: tuple or None = None, 
              save_dir: str or None = None, plot_out: bool = False, load_existing: bool = False):
        # assign save_dir if it exists
        self.save_dir = save_dir
        
        # if there's a location to save the imgs to, start the img saver thread
        if self.save_dir is not None:
            print(f'saving images to {self.save_dir}')
            
            # start a thread-safe queue
            print('starting image queue...')
            self.image_queue = queue.Queue()
            
            # start the consumer thread
            consumer = threading.Thread(target=self.image_saver)
            print('starting image saver thread...')
            consumer.start()            
            
            save_out = True
        else:
            print('not saving images...')
            save_out = False
        
        # start the producer thread
        producer = self.start_producer(
            mode, 
            mask_factor_list, 
            target_size, 
            save_out, 
            plot_out, 
            load_existing
        )
        
        # join threads
        producer.join()
        if save_out:
            consumer.join()
            
    def start_producer(self, mode, mask_factor_list, target_size, 
                       save_out, plot_out, load_existing):
        # if mode is pos, start pos image generator thread
        if mode == 'pos':
            # start pos producer thread
            producer = threading.Thread(
                target=self.pos_image_generator, 
                args=(mask_factor_list, target_size, save_out, plot_out)
            )
        
        # else if mode is neg, start neg image generator thread
        elif mode == 'neg':
            # start neg producer thread
            producer = threading.Thread(
                
                target=self.neg_image_generator, 
                args=(mask_factor_list, target_size, save_out, plot_out, load_existing)
            )
            
        else:
            raise ValueError(f"unrecognized mode '{mode}', please choose 'pos' or 'neg'...")
            
        print(f'starting {mode} image generator thread...')
        producer.start()
        
        return producer
        
    def image_saver(self):
        while True:
            item = self.image_queue.get()
            
            # if next item in queue is None, break
            if item is None:
                print('queue received stop signal, ending...')
                break
            
            # get img and save_path from item
            img, save_path = item
            
            # get directory from filename and make it if it doesn't already exist
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            
            # save image
            cv.imwrite(save_path, img)
#             print(f'image saved to {save_path} ...')
        
    def pos_image_generator(self, mask_factor_list, target_size, save_out: bool, plot_out: bool):
        # get output dataframe
        self.pos_out_df = self.pos_df.copy()
        
        # iterate over dataframe
        for i, data in tqdm(self.pos_df.iterrows(), total=self.n_images):
            # load img
            img = cv.imread(data.png_path)

            # get img view type
            view_type = data.ViewType
            
            # load roi
            roi_list = extract_roi(data.ROI_coords)

            # check img laterality
            laterality = check_side(img)

            # if laterality is 'right' flip img and roi coords
            if laterality == 'R':
                img = np.fliplr(img)
                roi_list = switch_coord_side(roi_list, img.shape)
            
            # record roi and tissue distribution
            self.record_roi_tissue_dist(img, roi_list, i)

            # initalize masked_img_list
            masked_img_list = []
        
            # initalize masked_img_paths string
            masked_img_paths = ''
            
            # iterate over mask factors
            for mask_factor in mask_factor_list:
                # generate masked images
                masked_img = self.mask_image(img, i, roi_list, mask_factor, 'pos')
                
                # append masked images to list
                masked_img_list.append(masked_img)
                
                if save_out:
                    self.save_image(masked_img, target_size, data.png_filename)
                    masked_img_paths += (masked_img.save_path + '$')
            
            self.pos_out_df.at[i, 'masked_factors'] = str(mask_factor_list)
            self.pos_out_df.at[i, 'masked_png_paths'] = masked_img_paths
                        
            if plot_out:
                plot_images(img, masked_img_list, mask_factor_list)
        
        # send stop signal to the queue if it exists
        self.send_stop_signal()

        # crop rows/cols arrays to max recorded img size
        self.crop_out_arrays()
        
        if save_out:
            self.save_attrs()
            
    def send_stop_signal(self):
        # send stop signal to queue if it exists
        if self.image_queue is not None:
            self.image_queue.put((None))
                
    def save_attrs(self):
        # save crop cols/rows arrays as .npy files
        np.save(
            os.path.join(self.save_dir, 'cols_arr.npy'), 
            self.crop_cols_array
        )
        np.save(
            os.path.join(self.save_dir, 'rows_arr.npy'), 
            self.crop_rows_array
        )

        # save roi dict to .json file
        with open(os.path.join(self.save_dir, 'roi_dict.json'), 'w') as json_dict:
            json.dump(self.roi_dict, json_dict)
                
    def load_attrs(self):
        # load crop cols/rows arrays from .npy files
        self.crop_cols_array = np.load(os.path.join(self.save_dir, 'cols_arr.npy'))
        self.crop_rows_array = np.load(os.path.join(self.save_dir, 'rows_arr.npy'))
        
        # load roi dict from .json file
        roi_dict_path = os.path.join(self.save_dir, 'roi_dict.json')
        with open(roi_dict_path, 'r') as json_dict:
            self.roi_dict = json.load(json_dict)
        
    def neg_image_generator(self, mask_factor_list, target_size, save_out: bool, 
                            plot_out: bool, load_existing: bool):
        # get output dataframe
        self.neg_out_df = self.neg_df.copy()
        
        self.neg_out_df['matched_pos_roi'] = np.nan
        self.neg_out_df['matched_pos_roi'] = self.neg_out_df['matched_pos_roi'].astype('object')
        
        # if load existing, load existing tissue dist arrays and roi dict
        if load_existing:
            self.load_attrs()
            # check that crop_cols/rows_array and roi_dict were loaded successfully
            if (self.crop_cols_array is not None) &\
            (self.crop_rows_array is not None) &\
            (self.roi_dict is not None):
                print('roi matching attributes loaded successfully...')
            else:
                print('error while loading roi matching attributes...')
        
        # iterate over dataframe
        for i, data in tqdm(self.neg_df.iterrows(), total=len(self.neg_df)):
            # load img
            img = cv.imread(data.png_path)

            # check img laterality
            laterality = check_side(img)

            # if laterality is 'right' flip img and roi coords
            if laterality == 'R':
                img = np.fliplr(img)
            
            # get tissue stats and find idx with lowest mse
            min_idx = self.find_min_tissue_dist_mse(img)
            
            # look up roi
            new_roi_list = self.roi_dict.get(str(min_idx))
            
            # initalize vars to track masked imgs and their save paths
            masked_img_list = []
            masked_img_paths = ''
            
            # generate masked images from roi
            # iterate over mask factors
            for mask_factor in mask_factor_list:
                # generate masked images
                masked_img = self.mask_image(img, i, new_roi_list, mask_factor, 'neg')
                
                # append masked images to list
                masked_img_list.append(masked_img)
                
                # if save out, save image
                if save_out:
                    self.save_image(masked_img, target_size, data.png_filename)
#                     masked_img.get_save_path(self.save_dir, data.png_filename)
                    masked_img_paths += (masked_img.save_path + '$')
            
            # record new roi to dataframe
            self.neg_out_df.at[i, 'matched_pos_idx'] = min_idx
            self.neg_out_df.at[i, 'matched_pos_roi'] = new_roi_list
            self.neg_out_df.at[i, 'masked_factors'] = str(mask_factor_list)
            self.neg_out_df.at[i, 'masked_png_paths'] = masked_img_paths
            
            if plot_out:
                plot_images(img, masked_img_list, mask_factor_list)
                
        # send stop signal to the queue if it exists
        self.send_stop_signal()
            
            
    def find_min_tissue_dist_mse(self, img):
        # record tissue tissue shape to cols/rows_array --------------------------
        # get a low percentile value of the image to deal with diff normalization
        bg_threshold = np.percentile(img, 2)

        # count the number of non-bg pixels in each of the cols/rows
        px_count_cols = (img[:, :, 0] > bg_threshold).sum(axis=0)
        px_count_rows = (img[:, :, 0] > bg_threshold).sum(axis=1)
        
        # map the px_count_cols/rows to padded arrays matching the size of crops cols/rows arrays
        px_count_cols_pad = np.zeros((self.crop_cols_array.shape[-1],), dtype=np.int32)
        px_count_cols_pad[:px_count_cols.shape[0]] = px_count_cols
        
        px_count_rows_pad = np.zeros((self.crop_rows_array.shape[-1],), dtype=np.int32)
        px_count_rows_pad[:px_count_rows.shape[0]] = px_count_rows
        
        # assume cols/rows arrays exist in object for now
        # add functionality to load them from .npy!!!
        cols_mse_array = compute_mse(px_count_cols_pad, self.crop_cols_array)
        rows_mse_array = compute_mse(px_count_rows_pad, self.crop_rows_array)
        
        # get array of the average cols/rows mse
        avg_mse_array = (cols_mse_array + rows_mse_array) / 2
        
        # find the minimum idx in the avg_mse array
        min_idx = np.argmin(avg_mse_array)
        
        return min_idx
            
    
    def record_roi_tissue_dist(self, img, roi_list, i):
        img_rows, img_cols, _ = img.shape

        # if img shape is bigger than max recorded, record it
        # WOULD THE MAX() FUNCTION BE FASTER HERE <-------------------------?
        if img_rows > self.max_n_rows:
            self.max_n_rows = img_rows
        if img_cols > self.max_n_cols:
            self.max_n_cols = img_cols
        
        # record tissue tissue shape to cols/rows_array --------------------------
        # get a low percentile value of the image to deal with diff normalization
        bg_threshold = np.percentile(img, 2)

        # count the number of non-bg pixels in each of the cols/rows
        px_count_cols = (img[:, :, 0] > bg_threshold).sum(axis=0)
        px_count_rows = (img[:, :, 0] > bg_threshold).sum(axis=1)

        # record current tissue distribution to main arrays
        self.cols_array[i, :img_cols] = px_count_cols
        self.rows_array[i, :img_rows] = px_count_rows

        # record current roi_list to roi_dict
        self.roi_dict[i] = roi_list
        
    def mask_image(self, img, i, roi_list, mask_factor, label):
        """
        function to mask rois on image according to mask factor
        """
        # if mask_factor is not 1.0, resize the mask accordingly
        if mask_factor != 1.0:
            roi_list = self.resize_roi(img, roi_list, mask_factor)

        # copy image and mask regions of interest
        masked_img_arr = img.copy()
        for roi in roi_list:
            masked_img_arr[roi[0]:roi[2], roi[1]:roi[3], :] = img.min()
            
        # initialize MaskedImage object
        masked_img = MaskedImage(masked_img_arr, i, roi_list, mask_factor, label)

        return masked_img

    def resize_roi(self, img, roi_list, mask_factor):
        """
        function to resize roi according to mask_factor
        """
        new_roi_list = []

        # get max image bounds
        img_max_y, img_max_x, _ = img.shape

        for roi in roi_list:
            # get height (ymax - ymin), and width (xmax - xmin)
            roi_height = roi[2] - roi[0]
            roi_width = roi[3] - roi[1]

            # get half of the adjusted height/width
            new_roi_height = roi_height * mask_factor
            new_roi_width = roi_width * mask_factor
            
            # get adjustments necessary to resize roi
            roi_y_adjust = int((new_roi_height - roi_height) // 2)
            roi_x_adjust = int((new_roi_width - roi_width) // 2)

            # apply adjustments to new_roi and clip between min and max coords
            new_roi = [
                clip_value(roi[0] - roi_y_adjust, 0, img_max_y), 
                clip_value(roi[1] - roi_x_adjust, 0, img_max_x), 
                clip_value(roi[2] + roi_y_adjust, 0, img_max_y), 
                clip_value(roi[3] + roi_x_adjust, 0, img_max_x)
            ]

            # append new_roi to new_roi_list
            new_roi_list.append(new_roi)

        return new_roi_list
    
    def crop_out_arrays(self):
        self.crop_cols_array = self.cols_array[:, :self.max_n_cols]
        self.crop_rows_array = self.rows_array[:, :self.max_n_rows]
        
    def save_image(self, masked_img, target_size, png_filename):
        # if img needs to be resized, resize it
        if target_size is not None:
            img = self.resize_image(masked_img.img, target_size)
        else:
            img = masked_img.img

        # get the path to save the image to
        masked_img.get_save_path(self.save_dir, png_filename)

        # add the image and 
        self.image_queue.put((img, masked_img.save_path))

    def resize_image(self, img, target_size):
        # create a black base image
        out_img = np.zeros(target_size, dtype=np.uint8)

        # find which dimension needs to be resized most to fit
        y_scale_factor = img.shape[0] / target_size[0]
        x_scale_factor = img.shape[1] / target_size[1]
        
        # get the actual image shape to resize to
        img_resize_shape = (int(img.shape[0] / y_scale_factor), int(img.shape[1] / x_scale_factor), 3)

        # resize the image
        resized_img = np.resize(img, img_resize_shape, )
        
class MaskedImage:
    def __init__(self, img, idx: int, roi_list: list, mask_factor: float, label: str):
        self.img = img
        self.idx = idx
        self.roi_list = roi_list
        self.mask_factor = mask_factor
        self.label = label
        self.save_path = ''
            
    def get_save_path(self, base_dir, png_filename):
        # get new filename
        filename_stem = png_filename.split('.')[0]
        
        # convert mask_factor to string for filename (e.g. 1.0 > '1_0')
        mask_factor_str = str(self.mask_factor).replace('.', '_')
        masked_filename = f"{filename_stem}_mask_{mask_factor_str}x.png"

        # get file subdirectory name
        sub_directory = f"masked_{mask_factor_str}x"
        
        # get label directory name
        label_directory = str(self.label)
        
        # update self.save_path
        self.save_path = os.path.join(
            base_dir, 
            sub_directory, 
            label_directory, 
            masked_filename
        )

        
def check_side(image, slice_width: int = 20, percentile: int = 2):
    """
    function to check laterality of png/dcm
    """    
    # take slices of the image on the left and right sides of the img
    slice_L = image[:, :slice_width]
    slice_R = image[:, -slice_width:]
    
    # get the background threshold by finding a percentile of the img
    bg_threshold = np.percentile(image, percentile)
    
    # count number of background pixels on each side
    num_bg_L = (np.array(slice_L) <= bg_threshold).sum()
    num_bg_R = (np.array(slice_R) <= bg_threshold).sum()
    
    # if there are fewer background pixels on the left than the right
    # laterality is left
    if num_bg_L < num_bg_R:
        return 'L'
    
    # if there are fewer background pixels on the right than the left
    # laterality is right
    elif num_bg_R < num_bg_L:
        return 'R'
    
    # if num background pixels is equal, laterality can't be determined
    else:
        print('Error: Laterality could not be determined!')
        return 'E'
    
def switch_coord_side(roi_list, full_shape):
    """
    function to switch the side of an roi
    """
    roi_list_out = []
    
    for roi in roi_list:
        minX = roi[1]
        maxX = roi[3]
        roi[1] = full_shape[1] - maxX
        roi[3] = full_shape[1] - minX
        roi_list_out.append(roi)
        
    return roi_list_out
        
def plot_images(img, masked_img_list, mask_factors):
    """
    function to plot original and masked images
    """
    # Create a new figure
    plt.figure(figsize=(10, 5))  # Adjust the size to your needs
    
    # get number of plots
    num_plots = 1 + len(masked_img_list)
    
    # Plot original img
    plt.subplot(1, num_plots, 1)  # 1 row, 2 columns, 1st subplot
    plt.imshow(img)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.title('Original')
        
    for i, masked_img in enumerate(masked_img_list):
        # Plot masked images
        plt.subplot(1, num_plots, 2 + i)  # 1 row, 2 columns, 2nd subplot
        plt.imshow(masked_img.img)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.title(f'{masked_img.mask_factor}x Masked')

    # Show the plot
    plt.tight_layout()  # Adjust the spacing between plots
    plt.show()
        
    
def extract_roi(roi: str):
    """
    convert ROI string to list
    """
    roi = roi.translate({ord(c): None for c in "][)(,"})
    roi = list(map(int, roi.split()))
    roi_list = []
    for i in range(len(roi) // 4):
        roi_list.append(roi[4*i:4*i+4])
    return roi_list

def clip_value(val, min_val, max_val):
    """
    function to clip value to range (like np.clip)
    """
    return max(min_val, min(val, max_val))

@jit
def compute_mse(array_a, array_b):
    n = array_b.shape[0]
    mse_values = np.empty(n)
    for i in range(n):
        mse_values[i] = np.mean((array_b[i] - array_a) ** 2)
    return mse_values
