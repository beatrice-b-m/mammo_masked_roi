import queue
import threading
from ImageMasker.generator.PosGenerator import PosGenerator
from ImageMasker.generator.NegGenerator import NegGenerator
from ImageMasker.saver.ImageSaver import ImageSaver


class DataFrameMasker:
    def __init__(self, pos_df, neg_df, mask_factor_list, target_size: tuple or None = None):
        self.pos_df = pos_df
        self.neg_df = neg_df

        self.mask_factor_list = mask_factor_list
        self.target_size = target_size
        
        # initialize roi dictionary to record ROIs indexed against the cols/rows arrays
        self.roi_dict = {}


    def start(self, mode, save_dir: str or None = None, plot_out: bool = False, load_existing: bool = False):
        # assign save_dir if it exists
        self.save_dir = save_dir
        
        # if there's a location to save the imgs to, start the img saver thread
        if self.save_dir is not None:
            print(f"saving images to '{self.save_dir}'...")
            saver, image_queue = self.start_saver()

        else:
            print('not saving images...')
            image_queue = None

        # start the generator thread
        generator = self.start_generator(
            mode,
            image_queue,
            plot_out, 
            load_existing
        )

        # join threads
        generator.join()
        if self.save_dir is not None:
            saver.join()

    def start_saver(self):    
        # start a thread-safe queue
        print('starting image queue...')
        image_queue = queue.Queue()
        
        # start the consumer thread
        saver = threading.Thread(
            target=ImageSaver, 
            args=(image_queue)
        )
        print('starting image saver thread...')
        saver.start()

        return saver, image_queue
            
    def start_generator(self, mode, image_queue, plot_out, load_existing):
        # if mode is pos, start pos image generator thread
        if mode == 'pos':
            # start pos generator thread
            generator = threading.Thread(
                # PosGenerator(self, )
                target=PosGenerator, 
                args=(
                    self.pos_df, 
                    image_queue, 
                    self.save_dir, 
                    self.mask_factor_list, 
                    self.target_size, 
                    plot_out
                )
            )
        
        # else if mode is neg, start neg image generator thread
        elif mode == 'neg':
            # start neg generator thread
            generator = threading.Thread(
                # start neg generator thread
                target=NegGenerator, 
                args=(
                    self.neg_df, 
                    image_queue, 
                    self.save_dir, 
                    self.mask_factor_list, 
                    self.target_size, 
                    plot_out, 
                    load_existing
                )
            )
            
        else:
            raise ValueError(f"unrecognized mode '{mode}', please choose 'pos' or 'neg'...")
            
        print(f'starting {mode} image generator thread...')
        generator.start()
        
        return generator
