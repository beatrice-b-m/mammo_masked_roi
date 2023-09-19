import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv


class BaseGenerator:
    def __init__(self, df, image_queue=None, save_dir=None, mask_factor_list=None, target_size=None, plot_out=False):
        """
        base generator class which pos/neg inherit from
        """
        # save dataframes and get the number of images in it
        self.df = df
        self.out_df = None
        self.n_images = len(df)

        # save image queue and the save dir
        self.image_queue = image_queue
        self.save_dir = save_dir

        # initialize cols/rows arrays to record tissue distributions
        # make them larger than we need, then crop to max img cols/rows after opening all imgs
        self.cols_array = np.zeros((2, self.n_images, 6000), np.int16)
        self.rows_array = np.zeros((2, self.n_images, 6000), np.int16)
        self.crop_cols_array = None
        self.crop_rows_array = None

        # save image generation params
        self.mask_factor_list = mask_factor_list
        self.target_size = target_size
        self.plot_out = plot_out

        # load roi dicts
        self.cc_roi_dict = {}
        self.mlo_roi_dict = {}

        print('finish this bit! :)')


    def get_view_axis_idx(self, data):
        if data.ViewType == 'CC':
            view_axis_idx = 0
        elif data.ViewType == 'MLO':
            view_axis_idx = 1
        else:
            raise ValueError(f"view type '{data.ViewType}' is not CC or MLO, please filter the dataframe...")

        return view_axis_idx


    def save_image(self, masked_img, png_filename):
        """
        sends an image to the queue
        """
        # if img needs to be resized, resize it
        if self.target_size is not None:
            img = self.resize_image(masked_img.img, self.target_size)
        else:
            img = masked_img.img

        # get the path to save the image to
        masked_img.get_save_path(self.save_dir, png_filename)

        # add the image and 
        self.image_queue.put((img, masked_img.save_path))


    def send_stop_signal(self):
        """
        Sends a stop signal to the queue
        """
        # send stop signal to queue if it exists
        if self.image_queue is not None:
            self.image_queue.put((None))


    def check_side(self, image, slice_width: int = 20, percentile: int = 2):
        """
        static but kept here for inheritance
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
        called by both generators
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
                self.clip_value(roi[0] - roi_y_adjust, 0, img_max_y), 
                self.clip_value(roi[1] - roi_x_adjust, 0, img_max_x), 
                self.clip_value(roi[2] + roi_y_adjust, 0, img_max_y), 
                self.clip_value(roi[3] + roi_x_adjust, 0, img_max_x)
            ]

            # append new_roi to new_roi_list
            new_roi_list.append(new_roi)

        return new_roi_list


    def clip_value(self, val, min_val, max_val):
        """
        this function is static but it's being included here so
        both pos and neg generators inherit it
        function to clip value to range (like np.clip)
        """
        return max(min_val, min(val, max_val))


    def resize_image(self, img):
        """
        called by both generators
        might be worth implementing a cropping function in this?
        """
        # create a black base image
        out_img = np.zeros(self.target_size + (3,), dtype=np.uint8)

        # find which dimension needs to be resized most to fit
        y_scale_factor = img.shape[0] / self.target_size[0]
        x_scale_factor = img.shape[1] / self.target_size[1]

        # find which scale factor is the largest
        max_scale_factor = max(y_scale_factor, x_scale_factor)
        
        # get the actual image shape to resize to
        img_resize_shape = (int(img.shape[0] / max_scale_factor), int(img.shape[1] / max_scale_factor))

        # resize the image
        resized_img = cv.resize(img, dsize=img_resize_shape, interpolation=cv.INTER_AREA)

        # apply resized image to black base
        out_img[:img_resize_shape[0], :img_resize_shape[1], :] = resized_img
        
        return out_img


    def plot_images(img, masked_img_list):
        """
        also a static function included here for inheritance
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



class MaskedImage:
    def __init__(self, img, idx: int, roi_list: list, mask_factor: float, label: str):
        """
        class to store all relevant info for each series of masked images.
        """
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