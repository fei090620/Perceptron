from Perceptron import Perceptron

f = lambda x:x

class LinerUnite(Perceptron):
    def __init__(self, input_xs, lables, train_speed, input_num):
        Perceptron.__init__(self, input_xs, lables, train_speed, input_num, f)


def get_train_data():
    input_xs = [[5,1,2.1],[3,2,3.1],[8,0,1.1],[1.4,100,1],[10.1,1000,1000]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_xs,labels

if __name__ == '__main__':
    input_xs, lables = get_train_data()
    lineUnite = LinerUnite(input_xs, lables, 0.1, 3)
    lineUnite.train(100)