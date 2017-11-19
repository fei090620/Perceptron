'''
read mnist data from mnist files
11/19/2017
fei090620
'''

import struct

import numpy as np

test_image_path = "./mnist_data/t10k-images.idx3-ubyte"
test_label_path = "./mnist_data/t10k-labels.idx1-ubyte"
train_imgae_path = "./mnist_data/train-images.idx3-ubyte"
train_label_path = "./mnist_data/train-labels.idx1-ubyte"


class mnist_data(object):
    def __init__(self):
        #big ender and 4 unsigned int
        self.image_header_tmf = '>IIII'
        #big ender and 2 unsigned int
        self.lable_header_tmf = '>II'
        #big ender and 1 byte
        self.lable_tmf = '>1B'

    def read_images(self, path):
        images_bytes = open(path, 'rb').read()
        magic_number, image_num, image_row, image_column = struct.unpack_from(self.image_header_tmf, images_bytes, 0)
        print magic_number, image_num, image_row, image_column
        images_offset = []
        image_size_tmf = '>{0}B'.format(image_row * image_column)
        index = struct.calcsize(self.image_header_tmf)
        for i in range(image_num):
            image_bytes = np.array(struct.unpack_from(image_size_tmf, images_bytes, index)).reshape(image_row, image_column)
            print image_bytes
            index += struct.calcsize(image_size_tmf)
            print image_bytes
            images_offset.append(image_bytes)
        return np.array(images_offset)


    def read_labels(self, path):
        labels_bytes = open(path, 'rb').read()
        magic_num, label_num = struct.unpack_from(self.lable_header_tmf, labels_bytes, 0)
        print magic_num, label_num
        labels_offset = []
        index = struct.calcsize(self.lable_tmf)
        for i in range(label_num):
            label_bytes = np.array(struct.unpack_from(self.lable_tmf, labels_bytes, index))
            print label_bytes
            index += struct.calcsize(self.lable_tmf)
            labels_offset.append(label_bytes)

        return np.array(labels_bytes)



if __name__ == '__main__':
    data_reader = mnist_data()
    t_images = data_reader.read_images(test_image_path)
    t_labels = data_reader.read_labels(test_label_path)
    tr_images = data_reader.read_images(train_imgae_path)
    tr_labels = data_reader.read_labels(train_label_path)

