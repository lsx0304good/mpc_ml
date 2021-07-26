import random
import numpy as np
from Option import *
from SigmoidFunction import *
from ThreeLayerNN import *
import struct
import os


def evaluate(network,X_full):
    for x in X_full:
        label = network.predict(secure(x))
        print(label)




def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

#标签one-hot处理
def onehot(targets, num):
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][targets[i]] = 1
    return result


if __name__ == '__main__':
    random.seed(123)
    np.random.seed(123)


    #测试安全性计算
    A_Test = SecureRational.secure(.2)
    B_Test = SecureRational.secure(-.62)
    AB_Test = A_Test * B_Test
    if(AB_Test.reveal() == (.2)*(-.62)):
        print("the Secure computing is ok")
        print(AB_Test.reveal())


    sigmoid = SigmoidFunction()
    secureRational = SecureRational()

    #读取训练数据
    X_train, Y_Train = load_mnist(r"/Users/simon_li/Desktop/aa/code/data",
                                  kind='train')
    #label标签做onhot编码
    Y_Train = onehot(Y_Train,60000)
    #预处理训练数据
    X_train =  X_train/255.0

    #读取测试数据
    X_test, y_test = load_mnist(r'/Users/simon_li/Desktop/aa/code/data',
                                 kind='t10k')
    X_test = X_test/255.0
    y_test = onehot(y_test,10000)

    #创建神经网络
    nn = ThreeLayerNN(sigmoid)

    #因为pdf说明了不用训练，所以这里仅仅取前5条数据进行代码测试而已
    nn.train(secure(X_train[0:5,:]),secure(Y_Train[0:5,:]),10)
#    nn.print_weights()

    print("测试")
    evaluate(nn,X_test)
