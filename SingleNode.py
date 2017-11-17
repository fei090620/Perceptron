
#neural network single node

class SingleNode(object):
    def __init__(self, layer_index, node_index, activator):
        self.activator = activator
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        self.output = output

    def append_downstream_connecttion(self, conns):
        self.downstream.extend(conns)

    def append_upstream_connecttion(self, conns):
        self.upstream.extend(conns)

    def cal_output(self):
        self.output = self.activator(reduce(lambda a, b: a + b, [up_conn.output() for up_conn in self.upstream]) + self.delta)

    def cal_delta_for_output(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def cal_delta_for_hide(self):
        self.delta = reduce(lambda x, y: x + y, [conn.delta() for conn in self.downstream]) * self.output * (1 - self.output)

    def update_upstream_conn_weights(self, speed):
        for conn in self.upstream:
            conn.update_weight(speed)

    def __str__(self):
        weights = " ".join([conn.__str__() + " " for conn in self.upstream])
        return "layer:{0} node:{1} output:{2} delta:{3} weights:{4} \n".format(self.layer_index, self.node_index, self.output, self.delta, weights)









