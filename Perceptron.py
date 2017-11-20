#一个And感知器，当输入为[1,1]，输出为1;输入为[1,0]，输出为0;输入为[0,1]，输出为0;输入为[0,0]，输出为0;
class Perceptron(object):
    def __init__(self, input_xs, lables, train_speed, input_num, activator):
        self.input_xs = input_xs
        self.labels = lables
        self.train_speed = train_speed
        self.out_put_ys = []
        self.weights = [0 for _ in range(input_num)]
        self.b = 0
        self.activator = activator

    def get_output(self, input_x):
        return self.activator(
            reduce(lambda a, b: a+b,
                   map(lambda x, y: x*y,
                       input_x, self.weights)) + self.b)

    def one_step_train(self):
        samples = zip(self.input_xs, self.labels)
        print_weights = ['weights:%10.3f' % weight for weight in self.weights]
        print print_weights
        print_b = 'b:%10.3f' % self.b
        print print_b

        for (input_x, label) in samples:
            out_put = self.get_output(input_x)
            var_out_put = label - out_put
            self.update_weight(var_out_put, input_x)
            self.update_b(var_out_put)


    def update_weight(self, var_out_put, input_x):
        var_weights = [self.train_speed * var_out_put * x for x in input_x]
        self.weights = map(lambda a,b:a+b, self.weights, var_weights)

    def train(self, iteration):
        for i in range(iteration):
            self.one_step_train()

    def update_b(self, var_out_put):
        self.b = self.b + self.train_speed * var_out_put


def activator(out_put):
    return 0 if out_put <0 else 1


if __name__ == '__main__':
    input_xs = [[1,1],[1,0],[0,1],[0,0]]
    labels = [1,1,1,0]
    train_speed = 0.1
    input_num = 2

    and_perceptron = Perceptron(input_xs, labels, train_speed, input_num, activator)
    and_perceptron.train(10)


