import tqdm
import cv2 as cv
import numpy as np
import os
import json
from numba import jit
from generator.BaseGenerator import BaseGenerator, MaskedImage


class NegGenerator(BaseGenerator):
    def __init__(self, df, image_queue, save_dir, mask_factor_list, target_size, plot_out, load_existing):
        super().__init__(df, image_queue, save_dir, mask_factor_list, target_size, plot_out)

        # if existing attributes should be loaded, load them
        if load_existing:
            self.load_attributes()

        # start generator loop
        self.generator_loop(load_existing)

    def load_attributes(self):
        """
        function to load existing attributes for negative image generation
        """
        # load crop cols/rows arrays from .npy files
        self.crop_cols_array = np.load(os.path.join(self.save_dir, 'cols_arr.npy'))
        self.crop_rows_array = np.load(os.path.join(self.save_dir, 'rows_arr.npy'))
        
        # load roi dict from .json file
        cc_roi_dict_path = os.path.join(self.save_dir, 'cc_roi_dict.json')
        with open(cc_roi_dict_path, 'r') as json_dict:
            self.cc_roi_dict = json.load(json_dict)

        mlo_roi_dict_path = os.path.join(self.save_dir, 'mlo_roi_dict.json')
        with open(mlo_roi_dict_path, 'r') as json_dict:
            self.mlo_roi_dict = json.load(json_dict)

        # check that crop_cols/rows_array and roi_dict were loaded successfully
        if all([self.crop_cols_array, self.crop_rows_array, self.cc_roi_dict, self.mlo_roi_dict]):
            print('roi matching attributes loaded successfully...')
        else:
            print('error while loading roi matching attributes...')


    def generator_loop(self, load_existing):
        # get output dataframe
        self.out_df = self.df.copy()
        
        self.out_df['matched_pos_roi'] = np.nan
        self.out_df['matched_pos_roi'] = self.out_df['matched_pos_roi'].astype('object')
        
        # if load existing, load existing tissue dist arrays and roi dict
        if load_existing:
            self.load_attrs()
        
        # iterate over dataframe
        for i, data in tqdm(self.df.iterrows(), total=len(self.df)):
            # load img
            img = cv.imread(data.png_path)

            # get img view type
            view_axis_idx = self.get_view_axis_idx(data)

            # check img laterality
            laterality = self.check_side(img)

            # if laterality is 'right' flip img and roi coords
            if laterality == 'R':
                img = np.fliplr(img)
            
            # get tissue stats and find idx with lowest mse
            min_idx = self.find_min_tissue_dist_mse(img, view_axis_idx)
            
            # look up roi in corresponding dict
            if view_axis_idx == 0:
                new_roi_list = self.cc_roi_dict.get(str(min_idx))
            else:
                new_roi_list = self.mlo_roi_dict.get(str(min_idx))
            
            # initalize vars to track masked imgs and their save paths
            masked_img_list = []
            masked_img_paths = ''
            
            # generate masked images from roi
            # iterate over mask factors
            for mask_factor in self.mask_factor_list:
                # generate masked images
                masked_img = self.mask_image(img, i, new_roi_list, mask_factor, 'neg')
                
                # append masked images to list
                masked_img_list.append(masked_img)
                
                # if save out, save image
                if self.save_dir is not None:
                    self.save_image(masked_img, self.target_size, data.png_filename)
                    masked_img_paths += (masked_img.save_path + '$')
            
            # record new roi to dataframe
            self.out_df.at[i, 'matched_pos_idx'] = min_idx
            self.out_df.at[i, 'matched_pos_roi'] = new_roi_list
            self.out_df.at[i, 'masked_factors'] = str(self.mask_factor_list)
            self.out_df.at[i, 'masked_png_paths'] = masked_img_paths
            
            if self.plot_out:
                self.plot_images(img, masked_img_list)
                
        # send stop signal to the queue if it exists
        self.send_stop_signal()


    def find_min_tissue_dist_mse(self, img, view_axis_idx):
        """
        Move to NegGenerator.py
        """
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
        
        # calculate mse for each col/row distribution array
        cols_mse_array = compute_mse(px_count_cols_pad, self.crop_cols_array[view_axis_idx, :, :])
        rows_mse_array = compute_mse(px_count_rows_pad, self.crop_rows_array[view_axis_idx, :, :])
        
        # get array of the average cols/rows mse
        avg_mse_array = (cols_mse_array + rows_mse_array) / 2
        
        # find the minimum idx in the avg_mse array
        min_idx = np.argmin(avg_mse_array)
        
        return min_idx

@jit
def compute_mse(array_a, array_b):
    """
    Move to NegGenerator.py
    """
    n = array_b.shape[0]
    mse_values = np.empty(n)
    for i in range(n):
        # if any values in array b are non-zero calculate mse
        if array_b[i].any():
            mse_values[i] = np.mean((array_b[i] - array_a) ** 2)
            
        # otherwise mse is positive infinity
        else:
            mse_values[i] = np.inf
        
    return mse_values