import numpy as np
import matplotlib.pylab as plt

# 阶跃函数
def step_function_v1(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function_v2(x):
    y = x > 0
    return y.astype(np.int)

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def step_pic():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_pic():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

def relu(x):
    return np.maximum(0, x)

def relu_pic():
    x = np.arange(-5.0, 6.0, 1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 5.1)
    plt.show()

def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1,0.3], [0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def test_1():
    x = 3.0
    print(step_function_v1(x))
    x = np.array([-1.0, 1.0, 2.0])
    print(x > 0)
    print(step_function(x))

def test_2():
    x = np.array([-1, 1.0, 2.0])
    print(sigmoid(x))

def test_3():
    step_pic()

def test_4():
    sigmoid_pic()

def test_5():
    relu_pic()

def test_6():
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
    B = np.array([0.1, 0.2, 0.3])
    A1 = np.dot(X, W1) + B
    Z1 = sigmoid(A1)
    print(A1)
    print(Z1)
    W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    B2 = np.array([0.1, 0.2])
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    W3 = np.array([[0.1,0.3], [0.2,0.4]])
    B3 = np.array([0.1,0.2])
    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3)
    print(Y)

def test_7():
    network = init_network()
    x = np.array([[1.0, 0.5]])
    y = forward(network, x)
    print(y)


if __name__ == '__main__':
    #test_1()
    #test_2()
    #test_3()
    #test_4()
    #test_5()
    test_6()
    test_7()