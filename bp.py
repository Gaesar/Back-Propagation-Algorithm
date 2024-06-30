import numpy as np
import math
import copy

def load_dataset(data_path):
    dataset =  np.load(data_path)
    return dataset[:130], dataset[:20], dataset[130:],

def print_net(network):
    for layer in network:
        print(layer)


def fun_z(weights, inputs):
    '''
    :param weights: [w1, w2, w3, ..., wn, b]
    :param inputs: [x1, x2, x3, ..., xn]
    :return: z = w1*x1 + w2*x2 + w3*x3 + ... + wn*xn + b
    '''
    # print(len(weights), " ", len(inputs))
    b = weights[-1]
    z = 0
    for i in range(len(inputs)):
        xi = inputs[i]
        wi = weights[i]
        z += xi*wi
    z += b
    return z


def sigmoid(z):
    '''
    :param z: z = w*x + b
    :return: f(z)
    '''
    if active == "sigmoid":
        return 1.0 / (1.0 + math.exp(-z))
    else:
        return (math.exp(z) - math.exp(-z)) / (math.exp(z) + math.exp(-z))

def sigmoid_derivative(output):
    """
    :param output of f(z)
    :return: f'(z) = (1 - f(z)) * f(z)
    """
    if active == "sigmoid":
        return (1.0 - output) * output
    else:
        return 1 - output**2


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    # 隐层层
    for i in range(len(n_hidden)):
        n_neuron = n_hidden[i]
        if i != 0:
            # 标准正态分布初始化,均值为0方差为1
            hidden_layer = [{'weights': [np.random.randn() for k in range(n_hidden[i - 1] + 1)]} for j in range(n_neuron)]
        else:
            # 第一层和w个数和输入个数有关
            hidden_layer = [{'weights': [np.random.randn() for k in range(n_inputs + 1)]} for j in range(n_neuron)]
        network.append(hidden_layer)
    # 输出层
    output_layer = [{'weights': [np.random.randn() for k in range(n_hidden[- 1] + 1)]} for j in range(n_outputs)]
    network.append(output_layer)
    return network


def forward_propagate(network, inputs):
    inputs_copy = copy.copy(inputs)
    for layer in network:
        next_input = []
        for neuron in layer:
            z = fun_z(neuron['weights'], inputs_copy)
            neuron['output'] = sigmoid(z)
            next_input.append(neuron['output'])
        inputs_copy = copy.copy(next_input)
    return inputs_copy


def backward_propagate_error(network, actual_value):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):  # 计算每一个神经元的误差
                neuron = layer[j]
                errors.append((actual_value[j] - neuron['output']))
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])


def update_parameters(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:  # 获取上一层网络的输出
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            # 更新偏置项
            neuron['weights'][-1] += l_rate * neuron['delta']


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs


def loss_calculation(actual_value, predicted_value):
    # return np.mean(np.square(np.array(actual_value) - np.array(predicted_value)))
    # 确保输入是NumPy数组
    actual_value = np.array(actual_value)
    predicted_value = np.array(predicted_value)

    # 计算均方误差
    mse = np.mean((actual_value - predicted_value) ** 2)

    return mse

def validation(network, val_data, n_inputs):
    predicted_value = []
    ret_predicted_value = []
    for row in val_data:
        prediction = predict(network, row[:-1])
        predicted_value.append(prediction)
        ret_predicted_value.append(prediction[0])
    actual_value = [row[n_inputs:] for row in val_data]
    re_actual_value = [row[-1] for row in val_data]
    loss = loss_calculation(actual_value, predicted_value)
    return loss, ret_predicted_value, re_actual_value


def test(network, val_data):
    normalize(val_data)
    predicted_value = []
    actual_value = []
    for row in val_data:
        prediction = predict(network, row[:-1])
        actual_value.append(row[-1])
        predicted_value.append(prediction[-1])
    denormalize(predicted_value, val_data)
    denormalize(actual_value, val_data)

    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.plot(predicted_value)
    # plt.plot(actual_value)
    # plt.show()
    return predicted_value, actual_value



def train(train_data, l_rate, epochs, n_hidden, val_data, test_data, mse):
    # 归一化
    normalize(train_data)
    normalize(val_data)
    normalize(test_data)
    n_inputs = len(train_data[0]) - 1
    n_outputs = 1
    # 初始化网络
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    val_losses = []
    test_losses = []
    predicted_values = []
    actual_value = []
    epoched = 0
    for epoch in range(epochs):  # 训练epochs个回合
        predicted_values = []
        actual_value = []
        for row in train_data:
            # 前馈计算
            forward_propagate(network, row[:-1])
            actual_value = [row[-1]]
            # 误差反向传播计算
            backward_propagate_error(network, actual_value)
            # 更新参数
            update_parameters(network, row, l_rate)
        # 保存当前epoch模型在验证集上的准确率
        val_loss, ret_predicted_value, re_actual_value= validation(network, val_data, n_inputs)
        test_loss, _, _ = validation(network, test_data, n_inputs)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        predicted_values = ret_predicted_value
        actual_value = re_actual_value
        epoched+=1
        if val_loss <= mse:
            break
    return network, val_losses, test_losses, predicted_values, actual_value, epoched
    # plt.xlabel('epochs')
    # plt.ylabel('val_loss')
    # plt.plot(val_loss)
    # plt.show()



def normalize(train_data):
    x = []
    y = []
    for row in train_data:
        x.append(row[:-1])
        y.append(row[-1])
    max_x = np.max(x)
    min_x = np.min(x)
    max_y = np.max(y)
    min_y = np.min(y)
    # x - min/max-min
    for row in train_data:
        for i in range(len(row) - 1):
            row[i] = (row[i] - min_x) / (max_x - min_x)
        row[-1] = (row[-1] - min_y) / (max_y - min_y)
    return train_data


def denormalize(y_, val_data):
    y = []
    for row in val_data:
        y.append(row[-1])
    max_y = np.max(y)
    min_y = np.min(y)
    for i in range(len(y_)):
        y_[i] = y_[i]*(max_y - min_y) + min_y

def bp(*args):
    # [0.01, 500.0, 0.05, [3, 3], 'D:/Desktop/BP/BP-Algorithm/iris.csv', 'D:/Desktop/BP/BP-Algorithm/iris.csv', 'sigmoid']
    global active
    l_rate = args[0]
    epochs = int(args[1])
    mse = args[2]
    n_hidden = args[3]
    data_path = args[4]
    active = args[5]
    # l_rate = 0.3  # 学习率
    # epochs = 1000  # 迭代训练的次数
    # n_hidden = [3, 3]  # 隐藏层神经元个数
    # n_train_data = 130  # 训练集的大小（总共150条数据，训练集130条，验证集20条）
    train_data, val_data, test_data = load_dataset(data_path)
    network, losses, losses2, re_predicted_values, re_actual_values, epoched = train(train_data, l_rate, epochs, n_hidden, val_data, test_data, mse)
    predicted_values, actual_values = test(network, test_data)
    return losses, losses2, re_predicted_values, re_actual_values, predicted_values, actual_values, epoched

