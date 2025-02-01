import numpy as np
import random
import os
import logging

logger = logging.getLogger(__name__)

class InputHandle:
    def __init__(self, data, indices, input_param):
        self.paths = input_param['paths']
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.channel = input_param['channel']
        self.data = data
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.total_length = input_param['seq_length']
        self.input_length = input_param['input_length']

    def total(self):
        return self.indices.shape[0]

    def begin(self, do_shuffle = True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        return self.current_position + self.minibatch_size > self.total()

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.total_length, self.image_width, self.image_width, self.channel)).astype(
            self.input_data_type)
        for i in range(self.current_batch_indices.shape[0]):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.total_length
            #if end > self.data.shape[0]: end = self.data.shape[0]
            data_slice = self.data[begin:end]
            #current_batch = np.concatenate([data_slice, np.zeros((self.total_length - self.input_length, self.image_width, self.image_width, self.channel))], axis=0)
            #print(begin, end, self.input_length, self.total_length, self.current_batch_indices.shape, data_slice.shape)
            
            input_batch[i, :self.total_length] = data_slice
            # logger.info('data_slice shape')
            # logger.info(data_slice.shape)
            # logger.info(input_batch.shape)
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

class DataProcess:
    def get_train_input_handle(self, input_param):
        files = os.listdir(input_param["paths"][0])

        data = []
        indices = [0]

        for i, file in enumerate(files):
            clip: np.ndarray = np.load(f'{input_param["paths"][0]}/{file}')['images'][:5]

            clip_2d = clip.reshape((clip.shape[0], 60, 60)) # Unflatten image array
            clip_2d = np.pad(clip_2d, ((0, 0), (0, 4), (0, 4))) # Make resolution power of 2
            data.append(clip_2d.reshape(*clip_2d.shape, 1)) # Add clip to data, reshaping to add channel

            if i > 0:
                indices.append(indices[-1] + clip.shape[0])
        
        data = np.concatenate(data, axis=0)
        indices = np.array(indices)
        print("TRAIN", data.shape, indices.shape)

        return InputHandle(data, indices, input_param)

    def get_test_input_handle(self, input_param):
        test_path = input_param["paths"][0]
        folders = os.listdir(test_path)

        data = []
        indices = [0]

        file_index = 0
        for folder in folders:
            folder_path = os.path.join(test_path, folder)
            files = os.listdir(folder_path)
            for file in files:
                clip: np.ndarray = np.load(os.path.join(folder_path, file))['images']
                
                clip_2d = clip.reshape((clip.shape[0], 60, 60)) # Unflatten image array
                clip_2d = np.pad(clip_2d, ((0, 0), (0, 4), (0, 4))) # Make resolution power of 2
                data.append(clip_2d.reshape(*clip_2d.shape, 1)) # Add clip to data, reshaping to add channel

                if file_index > 0:
                    indices.append(indices[-1] + clip.shape[0])
                
                file_index += 1
        
        data = np.concatenate(data, axis=0)
        indices = np.array(indices)
        print("TEST", data.shape, indices.shape)

        return InputHandle(data, indices, input_param)