import h5py
import numpy as np
import random
from PIL import Image
import os

dirpath = os.pardir
import sys

sys.path.append(dirpath)
from common.utils import unfold_label, shuffle_data, my_fft, my_fft_trans


class BatchImageGenerator: #datasets
    def __init__(self, flags, stage, file_path, b_unfold_label):

        if stage not in ['train', 'val', 'test']:
            assert ValueError('invalid stage!')

        self.configuration(flags, stage, file_path)
        self.load_data(b_unfold_label)

    def configuration(self, flags, stage, file_path):
        self.batch_size = flags.batch_size 
        self.file_path = file_path
        self.stage = stage 
        self.flags = flags 
        self.nums_d = len(self.file_path)
        self.current_indexs = [-1 for _ in range(self.nums_d)]


    def normalize(self, inputs):

        # the mean and std used for the normalization of
        # the inputs for the pytorch pretrained model
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # norm to [0, 1]
        inputs = inputs / 255.0

        inputs_norm = [] #注释了
        for item in inputs:
            item = np.transpose(item, (2, 0, 1)) #from hwc to chw
            item_norm = []
            for c, m, s in zip(item, mean, std):
                c = np.subtract(c, m)
                c = np.divide(c, s)
                item_norm.append(c)
            item_norm = np.stack(item_norm)
            inputs_norm.append(item_norm)

        return np.stack(inputs_norm)

    def load_data(self, b_unfold_label):
        # resize the image to 224 for the pretrained model
        def resize(x):
            x = x[:, :,
                [2, 1, 0]]  # we use the pre-read hdf5 data file from the download page and need to change BGR to RGB
            x = x.astype(np.uint8)
            img = np.array(Image.fromarray(x).resize((224, 224)))

            return img
        
        self.images = [[] for _ in range(self.nums_d)]
        self.labels = [[] for _ in range(self.nums_d)]
        
        for d_index in range(self.nums_d):
            f = h5py.File(self.file_path[d_index], "r")
            print(len(f['images']))
            images = np.array(list(map(resize, np.array(f['images']))))            
            self.images[d_index] = self.normalize(images)
            self.labels[d_index] = np.array(f['labels'])
            f.close()

        self.file_num_trains = [[] for _ in range(self.nums_d)]
        for d_index in range(self.nums_d): 
            assert np.max(self.images[d_index]) <= 5.0 and np.min(self.images[d_index]) >= -5.0
            assert len(self.images[d_index]) == len(self.labels[d_index])
            # shift the labels to start from 0
            self.labels[d_index] -= np.min(self.labels[d_index])
            self.file_num_trains[d_index] = len(self.labels[d_index])
            
        if self.stage is 'train':
            for d_index in range(self.nums_d):
                self.images[d_index], self.labels[d_index] = \
                                shuffle_data(samples=self.images[d_index], labels=self.labels[d_index])

    def get_images_labels_batch(self, domain_index):

        images = []
        labels = []
        H = []
        L = []
        for index in range(self.batch_size):
            self.current_indexs[domain_index]+=1
            # void over flow
            if self.current_indexs[domain_index] > self.file_num_trains[domain_index] - 1:
                self.current_indexs[domain_index] %= self.file_num_trains[domain_index]
                self.images[domain_index], self.labels[domain_index] = \
                            shuffle_data(samples=self.images[domain_index], labels=self.labels[domain_index])
                
            image_src = self.images[domain_index][self.current_indexs[domain_index]]
            decoder_H, decoder_L, decoder_H_ag, decoder_L_ag, thresh = my_fft(image_src, self.flags.threshold)
            decoder_H = np.transpose(decoder_H, (2, 0, 1))
            decoder_L = np.transpose(decoder_L, (2, 0, 1))
            decoder_H_ag = np.transpose(decoder_H_ag, (2, 0, 1))
            decoder_L_ag = np.transpose(decoder_L_ag, (2, 0, 1))
            
            image_ag = my_fft_trans(image_src, thresh)
            image_ag = self.normalize(np.array([image_ag])).squeeze()
                        
            images.append(image_ag)
            images.append(self.images[domain_index][self.current_indexs[domain_index]])
            
            labels.extend([self.labels[domain_index][self.current_indexs[domain_index]]]*2)
            H.extend([decoder_H_ag/255, decoder_H/255])
            L.extend([decoder_L_ag/255, decoder_L/255])
                                                          
        images = np.stack(images)
        labels = np.stack(labels)
        H = np.stack(H)
        L = np.stack(L)     
        return images, labels, H, L

