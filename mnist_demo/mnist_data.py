'''
read mnist data from mnist files
11/19/2017
fei090620
'''

import struct
import os
import numpy as np


class mnist_data_reader(object):
    # big ender and 4 unsigned int
    image_header_tmf = '>IIII'
    # big ender and 2 unsigned int
    lable_header_tmf = '>II'
    # big ender and 1 byte
    lable_tmf = '>1B'

    model_path = os.path.dirname(__file__)
    test_image_path = model_path + "/mnist_data/t10k-images.idx3-ubyte"
    test_label_path = model_path + "/mnist_data/t10k-labels.idx1-ubyte"
    train_imgae_path = model_path + "/mnist_data/train-images.idx3-ubyte"
    train_label_path = model_path + "/mnist_data/train-labels.idx1-ubyte"

    @staticmethod
    def read_images(path):
        images_bytes = open(path, 'rb').read()
        magic_number, image_num, image_row, image_column = struct.unpack_from(mnist_data_reader.image_header_tmf,
                                                                              images_bytes, 0)
        print magic_number, image_num, image_row, image_column
        images_offset = []
        image_size_tmf = '>{0}B'.format(image_row * image_column)
        index = struct.calcsize(mnist_data_reader.image_header_tmf)
        for i in range(image_num):
            image_bytes = np.array(struct.unpack_from(image_size_tmf, images_bytes, index)).reshape(image_row,
                                                                                                    image_column)
            print image_bytes
            index += struct.calcsize(image_size_tmf)
            print image_bytes
            images_offset.append(image_bytes)
        return images_offset

    @staticmethod
    def read_labels(path):
        labels_bytes = open(path, 'rb').read()
        magic_num, label_num = struct.unpack_from(mnist_data_reader.lable_header_tmf, labels_bytes, 0)
        print magic_num, label_num
        labels_offset = []
        index = struct.calcsize(mnist_data_reader.lable_tmf)
        for i in range(label_num):
            label_bytes = np.array(struct.unpack_from(mnist_data_reader.lable_tmf, labels_bytes, index))
            print label_bytes
            index += struct.calcsize(mnist_data_reader.lable_tmf)
            labels_offset.append(label_bytes)

        return labels_bytes

    @staticmethod
    def read_train_test_datas():
        t_images = mnist_data_reader.read_images(mnist_data_reader.test_image_path)
        t_labels = mnist_data_reader.read_labels(mnist_data_reader.test_label_path)
        tr_images = mnist_data_reader.read_images(mnist_data_reader.train_imgae_path)
        tr_labels = mnist_data_reader.read_labels(mnist_data_reader.train_label_path)
        return t_images, t_labels, tr_images, tr_labels


if __name__ == '__main__':
    t_images, tr_labels, tr_images, tr_labels = mnist_data_reader.read_train_test_datas()
    print t_images, tr_labels, tr_images, tr_labels
