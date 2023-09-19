import os
import cv2 as cv


class ImageSaver:
    def __init__(self, image_queue):
        self.image_queue = image_queue
        self._main_loop()

    def _main_loop(self):
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
