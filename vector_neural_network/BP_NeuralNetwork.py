import numpy as np
import sys

sys.path.append('..')

from mnist_demo.mnist_data import mnist_data_reader


class bp_neural_network(object):
    def __init__(self, layer_nodes):
        self.layer_num = len(layer_nodes)
        self.weights = []
        # self.deltas = []
        self.layer_nodes = layer_nodes
        self.init_weights(self.layer_nodes)
        # self.init_delta(self.layer_nodes)

    def one_step_train(self, input_X, labels, activator, speed):
        output, outputs = self.cal_output_Y(input_X, activator)
        deltas = self.cal_delta(labels, output, outputs)
        self.update_weights(deltas, speed)

    def train(self, input_Xs, label_Ts, activator, speed, iteration_num, t_input_Xs, t_label_Ts):
        if len(input_Xs) <> len(label_Ts):
            print 'Error: input_Xs is not same len with lable_Ts'
        for i in range(iteration_num):
            input_label = zip(input_Xs, label_Ts)
            for (input_X, label) in input_label:
                self.one_step_train(input_X, label, activator, speed)
            # cal error rate to charge when to stop train
            if self.cal_error_rate(t_input_Xs, t_label_Ts, activator) < 0.5:
                return

    def update_weights(self, deltas, speed):
        self.weights += speed * deltas
        print 'weights:{0} /n'.format(self.weights)

    def cal_delta(self, labels, output_Y, out_puts):
        if len(labels) != len(output_Y):
            print 'Error: albels is not same len with output_Y'

        output_delta = np.array(map(lambda x, y: x * (1 - x) * (y - x), zip(output_Y, labels)))
        deltas = [output_delta]
        i = len(self.weights) - 2
        while len(deltas) < len(self.layer_nodes) - 1 and i >= 0:
            weight_deltas = np.dot(self.weights[i], deltas[0])
            output_delta = np.array(map(lambda x, y: x * (1 - x) * y), zip(out_puts[i], weight_deltas))
            deltas.insert(0, output_delta)

        return deltas

    def cal_output_Y(self, input_X, activator):
        output = np.array(input_X).flatten()
        out_puts = []
        for weight in self.weights:
            output = activator(np.dot(weight, output))
            out_puts.append(output)
        print 'output: {0}/n'.format(output)
        return output, out_puts

    def init_weights(self, layer_nodes):
        for i in range(self.layer_num - 1):
            weight = np.random.rand(layer_nodes[i + 1], layer_nodes[i])
            print weight
            self.weights.append(weight)

    def cal_error_rate(self, t_input_Xs, t_label_Ts, activator):
        output_Ys = []
        for input_X in t_input_Xs:
            output_Ys.append(self.cal_output_Y(input_X, activator))

        var_error = np.array(output_Ys - t_label_Ts)
        var_error[var_error != 0] = 1
        rate = var_error.sum() / len(var_error)
        print rate
        return rate


# test for mnist

test_image_path = "../mnist_demo/mnist_data/t10k-images.idx3-ubyte"
test_label_path = "../mnist_demo/mnist_data/t10k-labels.idx1-ubyte"
train_imgae_path = "../mnist_demo/mnist_data/train-images.idx3-ubyte"
train_label_path = "../mnist_demo/mnist_data/train-labels.idx1-ubyte"

import math



if __name__ == '__main__':
    data_reader = mnist_data_reader()
    t_images = data_reader.read_images(test_image_path)
    t_labels = data_reader.read_labels(test_label_path)
    tr_images = data_reader.read_images(train_imgae_path)
    tr_labels = data_reader.read_labels(train_label_path)

    neural_network = bp_neural_network([28 * 28, 1000, 10])
    activator = np.vectorize(lambda a: 1 / (1 + math.exp(a)))
    input_Xs = tr_images
    label_Ts = tr_labels
    speed = 1
    iteration_num = 100
    t_input_Xs = t_images
    t_label_Ts = t_labels
    neural_network.train(input_Xs, label_Ts, activator, speed, iteration_num, t_input_Xs, t_label_Ts)
