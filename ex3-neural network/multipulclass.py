import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize


def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    X = np.insert(X, 0, 1, axis=1)
    y = y.flatten()
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def regularized_cost(theta, X, y, l):
    thetaReg = theta[1:]
    first = (-y*np.log(sigmoid(X@theta))) + (y-1)*np.log(1-sigmoid(X@theta))
    reg = (thetaReg@thetaReg)*l / (2*len(X))
    return np.mean(first) + reg


def regularized_gradient(theta, X, y, l):
    thetaReg = theta[1:]
    first = (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)
    # 这里人为插入一维0，使得对theta_0不惩罚，方便计算
    reg = np.concatenate([np.array([0]), (l / len(X)) * thetaReg])
    return first + reg


def one_vs_all(X, y, l, K):
    all_theta = np.zeros((K, X.shape[1]))  # (10, 401)

    for i in range(1, K+1):
        theta = np.zeros(X.shape[1])
        y_i = np.array([1 if label == i else 0 for label in y])
        ret = minimize(fun=regularized_cost, x0=theta, args=(X, y_i, l), method='TNC',
                       jac=regularized_gradient, options={'disp': True})
        all_theta[i-1, :] = ret.x

    return all_theta


def predict_all(X, all_theta):

    h = sigmoid(X @ all_theta.T)  # 注意的这里的all_theta需要转置

    h_argmax = np.argmax(h, axis=1)

    h_argmax = h_argmax + 1

    return h_argmax


path = 'F:\PycharmProjects\MLcode\ex3-neural network\ex3data1.mat'
X, y = load_data(path)

all_theta = one_vs_all(X, y, 1, 10)

y_pred = predict_all(X, all_theta)
accuracy = np.mean(y_pred == y)
print('accuracy = {0}%'.format(accuracy * 100))
