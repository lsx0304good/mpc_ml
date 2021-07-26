import numpy as np
from Option import *

#三层神经网络,包含了输入层，总共3层，第一层是784个节点，第二层是128个，第三层是10个
class ThreeLayerNN:
    def __init__(self, sigmoid):
        self.sigmoid = sigmoid



    def train(self, X, y, iterations, alpha=1):

        # prepare alpha value
        alpha = secure(alpha)

        # 初始化权重
        self.synapse0 = secure(2 * np.random.random((784, 128)) - 1)
        self.synapse1 = secure(2 * np.random.random((128, 10)) - 1)


        # 训练
        for i in range(iterations):

            # 前馈
            layer0 = X
            layer1 = self.sigmoid.evaluate(np.dot(layer0, self.synapse0))
            layer2 = self.sigmoid.evaluate(np.dot(layer1, self.synapse1))



            # 反向传播
            #layer2_error = np.sum((y - layer2)*(y-layer2))
            layer2_error = y - layer2
            layer2_delta = layer2_error * self.sigmoid.derive(layer2)
            layer1_error = np.dot(layer2_delta, self.synapse1.T)
            layer1_delta = layer1_error * self.sigmoid.derive(layer1)

            # p打印错误率
            if (i + 1) % (iterations // 10) == 0:
                print("Error: %s" % np.mean(np.abs(reveal(layer2_error))))

            # 更新权重
            self.synapse1 += np.dot(layer1.T, layer2_delta) * alpha
            self.synapse0 += np.dot(layer0.T, layer1_delta) * alpha


    def predict(self, X):
        layer0 = X
        layer1 = self.sigmoid.evaluate(np.dot(layer0, self.synapse0))
        layer2 = self.sigmoid.evaluate(np.dot(layer1, self.synapse1))
        score = reveal(layer2)       #解秘
        result = np.where(score==np.max(score))    #找到score中最大的，如果有多个，取第一个
        return result[0]

    def print_weights(self):
        print("Layer 0 weights: \n%s" % self.synapse0)
        print("Layer 1 weights: \n%s" % self.synapse1)