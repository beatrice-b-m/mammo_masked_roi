from tqdm import tqdm
import cv2 as cv
import numpy as np
import os
import json
from ImageMasker.generator.BaseGenerator import BaseGenerator, MaskedImage


class PosGenerator(BaseGenerator):
    def __init__(self, df, image_queue, save_dir, mask_factor_list, target_size, plot_out):
        super().__init__(df, image_queue, save_dir, mask_factor_list, target_size, plot_out)

        # initalize max n rows/cols vars to track largest seen images for cropping
        self.max_n_rows = 0
        self.max_n_cols = 0

        # start generator loop
        self.generator_loop()


    def generator_loop(self): # mask_factor_list, target_size, save_out: bool, plot_out: bool
        # get output dataframe
        self.out_df = self.df.copy()
        
        # iterate over dataframe
        for i, data in tqdm(self.df.iterrows(), total=self.n_images):
            # load img
            img = cv.imread(data.png_path)
            print(f'\nimg {i} loaded')

            # get img view type
            view_axis_idx = self.get_view_axis_idx(data)
            
            # load roi
            roi_list = extract_roi(data.ROI_coords)

            # check img laterality
            laterality = self.check_side(img)

            # if laterality is 'right' flip img and roi coords
            if laterality == 'R':
                img = np.fliplr(img)
                roi_list = switch_coord_side(roi_list, img.shape)
                
            print('laterality checked')
            
            # record roi and tissue distribution
            self.record_roi_tissue_dist(img, view_axis_idx, roi_list, i)
            print('tissue dist recorded')

            # initalize masked_img_list
            masked_img_list = []
        
            # initalize masked_img_paths string
            masked_img_paths = ''
            
            # iterate over mask factors
            for mask_factor in self.mask_factor_list:
                # generate masked images
                masked_img = self.mask_image(img, i, roi_list, mask_factor, 'pos')
                
                # append masked images to list
                masked_img_list.append(masked_img)
                
                if self.save_dir is not None:
                    self.save_image(masked_img, data.png_filename)
                    masked_img_paths += (masked_img.save_path + '$')
                    
            print('images generated')
            
            self.out_df.at[i, 'masked_factors'] = str(self.mask_factor_list)
            self.out_df.at[i, 'masked_png_paths'] = masked_img_paths
            
            print('dataframe updated')
                        
            if self.plot_out:
                self.plot_images(img, masked_img_list)

        # send stop signal to the queue if it exists
        self.send_stop_signal()

        # crop rows/cols arrays to max recorded img size
        self.crop_out_arrays()
        
        # if saving, save pos image attributes
        if self.save_dir is not None:
            self.save_attrs()


    def record_roi_tissue_dist(self, img, view_axis_idx, roi_list, i):
        """
        Move to PosGenerator.py
        """
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
        self.cols_array[view_axis_idx, i, :img_cols] = px_count_cols
        self.rows_array[view_axis_idx, i, :img_rows] = px_count_rows

        # record current roi_list to roi_dict
        if view_axis_idx == 0:
            self.cc_roi_dict[i] = roi_list
        else:
            self.mlo_roi_dict[i] = roi_list


    def crop_out_arrays(self):
        """
        crop cols/rows arrays to the max recorded size
        """
        self.crop_cols_array = self.cols_array[:, :, :self.max_n_cols]
        self.crop_rows_array = self.rows_array[:, :, :self.max_n_rows]

    def save_attrs(self):
        """
        save cols/rows arrays and roi dict to file
        """
        # save crop cols/rows arrays as .npy files
        np.save(
            os.path.join(self.save_dir, 'cols_arr.npy'), 
            self.crop_cols_array
        )
        np.save(
            os.path.join(self.save_dir, 'rows_arr.npy'), 
            self.crop_rows_array
        )

        # save roi dicts to .json files
        with open(os.path.join(self.save_dir, 'cc_roi_dict.json'), 'w') as json_dict:
            json.dump(self.cc_roi_dict, json_dict)
        with open(os.path.join(self.save_dir, 'mlo_roi_dict.json'), 'w') as json_dict:
            json.dump(self.mlo_roi_dict, json_dict)


def switch_coord_side(roi_list, full_shape):
    """
    static function to switch the side of an roi
    if the tissue is right-aligned
    """
    roi_list_out = []
    
    for roi in roi_list:
        minX = roi[1]
        maxX = roi[3]
        roi[1] = full_shape[1] - maxX
        roi[3] = full_shape[1] - minX
        roi_list_out.append(roi)
        
    return roi_list_out


def extract_roi(roi: str):
    """
    static function to convert ROI string from
    dataframe to nested list
    """
    roi = roi.translate({ord(c): None for c in "][)(,"})
    roi = list(map(int, roi.split()))
    roi_list = []
    for i in range(len(roi) // 4):
        roi_list.append(roi[4*i:4*i+4])
    return roi_list

