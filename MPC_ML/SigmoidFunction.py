from Option import *
import numpy as np

#这个地方采用麦克劳林公式来逼近正常的sigmod函数
class SigmoidFunction:
    def __init__(self):
        #使用麦克劳林公式逼近
        # ONE = SecureRational.secure(1)
        # W0 = SecureRational.secure(1/2)
        # W1 = SecureRational.secure(1 / 4)
        # W3 = SecureRational.secure(-1 / 48)
        # W5 = SecureRational.secure(1 / 480)
        # self.sigmoid = np.vectorize(lambda x: W0 + (x * W1) + (x ** 3 * W3) + (x ** 5 * W5))  #逼近sigmoid
        # self.sigmoid_deriv = np.vectorize(lambda x: (ONE - x) * x)       #sigmoid导数


        # #使用改进的relu函数
        def f(x):
            cc = reveal(x)
            if(cc<-0.5):
                return SecureRational.secure(0)
            elif (cc>0.5):
                return SecureRational.secure(1)
            else:
                return x

        def f2(x):
            cc = reveal(x)
            if(-0.5<=cc<=0.5):
                return SecureRational.secure(1)
            else:
                return SecureRational.secure(0)

        self.sigmoid = np.vectorize(lambda x: f(x))
        self.sigmoid_deriv = np.vectorize(lambda x: f2(x))  # sigmoid导数

    def evaluate(self, x):
        return self.sigmoid(x)

    def derive(self, x):
        return self.sigmoid_deriv(x)