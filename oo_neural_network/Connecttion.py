
# define connection of neural network

class Connecttion(object):
    def __init__(self, upstream_node, downstream_node):
        self.downstream_node = downstream_node
        self.upstream_node = upstream_node
        self.weight = 0

    def output(self):
        return self.upstream_node.output * self.weight

    def delta(self):
        return self.downstream_node.delta * self.weight

    def update_weight(self, speed):
        out_put = self.upstream_node.output
        delta = self.upstream_node.delta
        self.weight += out_put * speed * delta

    def __str__(self):
        return "{0}_{1}weight:{2}".format(self.downstream_node.node_index, self.upstream_node.node_index, self.weight)



