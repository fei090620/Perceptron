import numpy as np


class cnn_neural_network(object):
    def __init__(self, input_layer, con_layers, oo_layers):
        self.oo_layers = oo_layers
        self.con_layers = con_layers
        self.input_layer = input_layer
        pass


if __name__ == '__main__':
    input_layer = np.array([280, 280])
    con_layers = [[10, 10, 0, 2], [10, 10, 0, 1], [5, 5, 0, 1]]
    oo_layers = []

