from SingleNode import SingleNode

class Layer(object):
    def __init__(self, nodes):
        self.nodes = nodes

    def cal_nodes_outputs(self):
        for node in self.nodes:
            node.cal_output()

    def cal_deltas_for_output(self, labels):
        node_with_lables = zip(self.nodes, labels)
        for (node, lable) in node_with_lables:
            node.cal_delta_for_output(lable)

    def cal_deltas_for_hide(self):
        for node in self.nodes:
            node.cal_delta_for_hide()

    def set_inputs(self, inputs):
        node_with_inputs = zip(self.nodes, inputs)
        for (node, input_x) in node_with_inputs:
            node.set_output(input_x)

    def update_node_up_conn_weights(self, speed):
        for node in self.nodes:
            node.update_upstream_conn_weights(speed)

    def __str__(self):
        return "".join([node.__str__() for node in self.nodes])





