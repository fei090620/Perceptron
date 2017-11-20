
import math

from Connecttion import Connecttion
from SingleNode import SingleNode
from Layer import Layer


'''

define neural network and training progress

Data: 2017/11/16

'''

#activator
sigmode = lambda x: 1 / (1 + math.exp(x))


#define input layer nodes
input_node_1 = SingleNode(1, 1, sigmode)
input_node_2 = SingleNode(1, 2, sigmode)

#define hide 1 layer nodes
hide_2_node_3 = SingleNode(2, 3, sigmode)
hide_2_node_4 = SingleNode(2, 4, sigmode)
hide_2_node_5 = SingleNode(2, 5, sigmode)
hide_2_node_6 = SingleNode(2, 6, sigmode)

#define hide 2 layer nodes
hide_3_node_7 = SingleNode(3, 7, sigmode)
hide_3_node_8 = SingleNode(3, 8, sigmode)
hide_3_node_9 = SingleNode(3, 9, sigmode)
hide_3_node_10 = SingleNode(3, 10, sigmode)

#define output layer nodes
output_node_11 = SingleNode(4, 11, sigmode)
output_node_12 = SingleNode(4, 12, sigmode)

#define 1 and 2 layer's connections
conn_1_3 = Connecttion(input_node_1, hide_2_node_3)
conn_1_4 = Connecttion(input_node_1, hide_2_node_4)
conn_1_5 = Connecttion(input_node_1, hide_2_node_5)
conn_1_6 = Connecttion(input_node_1, hide_2_node_6)

input_node_1.append_downstream_connecttion([conn_1_3, conn_1_4, conn_1_5, conn_1_6])

conn_2_3 = Connecttion(input_node_2, hide_2_node_3)
conn_2_4 = Connecttion(input_node_2, hide_2_node_4)
conn_2_5 = Connecttion(input_node_2, hide_2_node_5)
conn_2_6 = Connecttion(input_node_2, hide_2_node_6)

input_node_2.append_downstream_connecttion([conn_2_3, conn_2_4, conn_2_5, conn_2_6])

#define the 2 and 3 layer's connections
conn_3_7 = Connecttion(hide_2_node_3, hide_3_node_7)
conn_3_8 = Connecttion(hide_2_node_3, hide_3_node_8)
conn_3_9 = Connecttion(hide_2_node_3, hide_3_node_9)
conn_3_10 = Connecttion(hide_2_node_3, hide_3_node_10)

hide_2_node_3.append_upstream_connecttion([conn_1_3, conn_2_3])
hide_2_node_3.append_downstream_connecttion([conn_3_7, conn_3_8, conn_3_9, conn_3_10])

conn_4_7 = Connecttion(hide_2_node_4, hide_3_node_7)
conn_4_8 = Connecttion(hide_2_node_4, hide_3_node_8)
conn_4_9 = Connecttion(hide_2_node_4, hide_3_node_9)
conn_4_10 = Connecttion(hide_2_node_4, hide_3_node_10)

hide_2_node_4.append_upstream_connecttion([conn_1_4, conn_2_4])
hide_2_node_4.append_downstream_connecttion([conn_4_7, conn_4_8, conn_4_9, conn_4_10])

conn_5_7 = Connecttion(hide_2_node_5, hide_3_node_7)
conn_5_8 = Connecttion(hide_2_node_5, hide_3_node_8)
conn_5_9 = Connecttion(hide_2_node_5, hide_3_node_9)
conn_5_10 = Connecttion(hide_2_node_5, hide_3_node_10)

hide_2_node_5.append_upstream_connecttion([conn_1_5, conn_2_5])
hide_2_node_5.append_downstream_connecttion([conn_5_7, conn_5_8, conn_5_9, conn_5_10])

conn_6_7 = Connecttion(hide_2_node_6, hide_3_node_7)
conn_6_8 = Connecttion(hide_2_node_6, hide_3_node_8)
conn_6_9 = Connecttion(hide_2_node_6, hide_3_node_9)
conn_6_10 = Connecttion(hide_2_node_6, hide_3_node_10)

hide_2_node_6.append_upstream_connecttion([conn_1_6, conn_2_6])
hide_2_node_6.append_downstream_connecttion([conn_6_7, conn_6_8, conn_6_9, conn_6_10])

#define 3 and 4 layer's connections
conn_7_11 = Connecttion(hide_3_node_7, output_node_11)
conn_7_12 = Connecttion(hide_3_node_7, output_node_12)

hide_3_node_7.append_upstream_connecttion([conn_3_7, conn_4_7, conn_5_7, conn_6_7])
hide_3_node_7.append_downstream_connecttion([conn_7_11, conn_7_12])

conn_8_11 = Connecttion(hide_3_node_8, output_node_11)
conn_8_12 = Connecttion(hide_3_node_8, output_node_12)

hide_3_node_8.append_upstream_connecttion([conn_3_8, conn_4_8, conn_5_8, conn_6_8])
hide_3_node_8.append_downstream_connecttion([conn_8_11, conn_8_12])

conn_9_11 = Connecttion(hide_3_node_9, output_node_11)
conn_9_12 = Connecttion(hide_3_node_9, output_node_12)

hide_3_node_9.append_upstream_connecttion([conn_3_9, conn_4_9, conn_5_9, conn_6_9])
hide_3_node_9.append_downstream_connecttion([conn_9_11, conn_9_12])

conn_10_11 = Connecttion(hide_3_node_10, output_node_11)
conn_10_12 = Connecttion(hide_3_node_10, output_node_12)

hide_3_node_10.append_upstream_connecttion([conn_3_10, conn_4_10, conn_5_10, conn_6_10])
hide_3_node_10.append_downstream_connecttion([conn_10_11, conn_10_12])

output_node_11.append_upstream_connecttion([conn_7_11, conn_8_11, conn_9_11, conn_10_11])
output_node_12.append_upstream_connecttion([conn_7_12, conn_8_12, conn_9_12, conn_10_12])

#define input 1 layer
input_Layer = Layer([input_node_1, input_node_2])
#define hide 2 layer
hide_2_layer = Layer([hide_2_node_3, hide_2_node_4, hide_2_node_5, hide_2_node_6])
#define hide 3 layer
hide_3_layer = Layer([hide_3_node_7, hide_3_node_8, hide_3_node_9, hide_3_node_10])
#define output layer
output_Layer = Layer([output_node_11, output_node_12])

def one_step_train(layers, input_X, labels, speed):
    input_layer = layers[0]
    input_layer.set_inputs(input_X)

    for node in input_layer.nodes:
        print node

    for layer in layers[1:]:
        layer.cal_nodes_outputs()

    output_layer = layers[-1]
    output_layer.cal_deltas_for_output(labels)


    for layer in layers[-2:0:-1]:
        layer.cal_deltas_for_hide()

    for layer in layers[1:]:
        layer.update_node_up_conn_weights(speed)
        print layer



def train(layers, input_Xs, lable_Ys, iterations, speed):
    input_lables = zip(input_Xs, label_Ys)
    for i in range(iterations):
        for (input_X, lable_Y) in input_lables:
            one_step_train(layers, input_X, lable_Y, speed)

def get_train_data():
    input_Xs = [[100,0],[101,1],[2,300],[0.11,22],[90,20],[10,20]]
    label_Ys = [[1,0],[1,0],[0,1],[0,1],[1,0],[0,0]]
    return input_Xs, label_Ys


if __name__ == '__main__':
    input_Xs, label_Ys = get_train_data()
    iterations = 100
    speed = 1
    train([input_Layer, hide_2_layer, hide_3_layer, output_Layer], input_Xs, label_Ys, iterations, speed)






