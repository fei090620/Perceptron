import numpy as np


class con_layer(object):
    def __init__(self, input_width, input_height, filter_width, filter_height,
                 filter_deep, foot_step, filter_num, filter_zero):
        self.filter_zero = filter_zero
        self.filers = []
        self.filter_num = filter_num
        self.filter_deep = filter_deep
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.input_height = input_height
        self.input_width = input_width
        self.feature_map_height = self.__cal_feature_height(input_height, filter_height, foot_step, filter_zero)
        self.feature_map_width = self.__cal_feature_width(input_width, filter_width, foot_step, filter_zero)
        self.foot_step = foot_step
        self.init_filters(filter_width, filter_height, filter_num, filter_deep)
        self.init_sensatives(input_width, input_height, filter_width, filter_height, filter_num, filter_deep)

    def init_filters(self, filter_width, filter_height, filter_num, filter_deep):
        for i in range(filter_num):
            current_filter = np.zeros([filter_height, filter_width, filter_deep])
            self.filers.append(current_filter)

    def cal_feature_maps(self, input_array):
        feature_maps = []
        for filterItem in self.filers:
            feature_maps.append(np.convolve(input_array, filterItem, 'valid'))

        return feature_maps

    def update_filters(self):
        pass

    def update_sensative(self):
        pass

    def __cal_feature_width(self, input_width, filter_width, foot_step, filter_zero):
        return (input_width - filter_width + 2 * filter_zero) / foot_step + 1

    def __cal_feature_height(self, input_height, filter_height, foot_step, filter_zero):
        return (input_height - filter_height + 2 * filter_zero) / foot_step + 1

    def init_sensatives(self, filter_num, filter_deep):
        self.sensatives = []
        for i in range(filter_num):
            temp_sensative = np.zeros([self.feature_map_height, self.feature_map_width ,filter_deep])
            self.sensatives.append(temp_sensative)




