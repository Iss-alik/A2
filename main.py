import numpy
import scipy.special
import matplotlib.pyplot

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learninggrate):
        self.innodes = inputnodes
        self.hnodes = hiddennodes
        self.outnodes = outputnodes
        self.ln = learninggrate

        self.wih = (numpy.random.rand(self.hnodes,self.innodes) - 0.5)
        self.who = (numpy.random.rand(self.outnodes,self.hnodes) - 0.5)
    
        self.activation_function =lambda x: scipy.special.expit(x)

    def train(self, inputs_list, target_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin = 2).T

        f_output = self.query(inputs_list)
        h_output = self.h_output

        output_errors = targets - f_output 
        hiden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.ln * numpy.dot(output_errors  * f_output * (1-f_output),numpy.transpose(h_output))
        self.wih += self.ln * numpy.dot(hiden_errors  * h_output * (1-h_output),numpy.transpose(inputs))

    def query(self, inputs_list):
        self.inputs = numpy.array(inputs_list, ndmin =2).T

        h_input = numpy.dot(self.wih, self.inputs)
        self.h_output = self.activation_function(h_input)

        f_input = numpy.dot(self.who, self.h_output)
        f_output = self.activation_function(f_input)

        return f_output
    
sifr = NeuralNetwork(inputnodes=784, hiddennodes=100, outputnodes=10, learninggrate=0.3)

training_file = open("mnist_train.csv", 'r')
training_list = training_file.readlines()
training_file.close()

for record in training_list[1:]:
    all_values = record.split(',')

    inputs = (numpy.asfarray(all_values[1:]) / 255 * 0.99) +0.01

    targets = numpy.zeros(10) + 0.01
    targets[ int(all_values[0]) ] = 0.99
    sifr.train(inputs, targets)

scorecard = []

test_file = open("mnist_test.csv", 'r')
test_list = test_file.readlines()
test_file.close()

for record in test_list[1:]:
    all_values = record.split(',')

    correct_label = int(all_values[0])

    inputs = (numpy.asfarray(all_values[1:]) / 255 * 0.99) +0.01
    output = sifr.query(inputs)
    label = numpy.argmax(output)

    if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = numpy.asarray(scorecard)
print ("эффективность = ", scorecard_array.sum() / scorecard_array.size)