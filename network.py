import random

def relu(x):
    if x > 1:
        return 1 + 0.01 * (x - 1)
    elif x < 0:
        return 0.01 * x
    else:
        return x


def der_relu(x):
    if x > 1:
        return 0.01
    elif x < 0:
        return 0.01
    else:
        return 1

def mse_loss(y_true, y_pred):
  dy1 = y_pred[0] - y_true[0]
  dy2 = y_pred[1] - y_true[1]
  dy3 = y_pred[2] - y_true[2]
  return (dy1**2 + dy2**2 + dy3**2)/3



class NeuralNetwork():

    def __init__(self):
        self.w1 = []
        self.w2 = []
        self.w3 = []
        self.flayer = [0,0,0,0,0,0,0,0,0,0]
        self.slayer = [0,0,0,0,0,0,0,0,0,0]
        self.output = [0,0,0]
        self.sig_xo = [0,0,0]
        self.sig_xs = [0,0,0,0,0,0,0,0,0,0]
        self.sig_xf = [0,0,0,0,0,0,0,0,0,0]
        self.e0 = 0
        for i in range(30):
            self.w1.append(random.uniform(-1,1))
            self.w3.append(random.uniform(-1,1))
        for i in range(100):
            self.w2.append(random.uniform(-1,1))


    def forward_prop(self, input_pix):
        input_pix = [input_pix[0] / 255, input_pix[1]/255, input_pix[2]/255]
        self.flayer = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(10):
            for j in range(3):
                self.flayer[i] += input_pix[j] * self.w1[i + 10 * j]

        self.slayer = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(10):
            for j in range(10):
                self.slayer[i] += relu(self.flayer[j])*self.w2[i + 10*j]

        self.output = [0, 0, 0]
        for i in range(3):
            for j in range(10):
                self.output[i] += relu(self.slayer[j])*self.w3[i+3*j]
            self.output[i] = relu(self.output[i])
        return self.output

    def back_prop(self, input_y, y_true):
        y_true = [y_true[0] / 255, y_true[1] / 255, y_true[2] / 255]
        eta = 0.002
        self.e0 = mse_loss(y_true, self.output)
        w3_copy = self.w3
        w2_copy = self.w2
        for num in range(30):
            i = num//3
            j = num%3
            self.sig_xo[j] = (y_true[j]-self.output[j])*der_relu(self.output[j])
            p_wij = self.sig_xo[j] * self.slayer[i]
            dwij = eta*p_wij
            self.w3[num] += dwij
        for num in range(100):
            i = num//10
            j = num%10
            sum_sigw = self.sig_xo[0] * w3_copy[3*j] + self.sig_xo[1] * w3_copy[3*j+1] + self.sig_xo[2] * w3_copy[3*j+2]
            self.sig_xs[j] = der_relu(self.slayer[j]) * (sum_sigw)
            p_wij = self.sig_xs[j] * self.flayer[i]
            dwij = eta * p_wij
            self.w2[num] += dwij
        for num in range(30):
            i = num%3
            j = num//3
            sum_sigw = 0
            for k in range(10):
                sum_sigw += self.sig_xs[k] * w2_copy[10*j+k]
            self.sig_xf[j] = der_relu(self.flayer[j]) * (sum_sigw)
            p_wij = self.sig_xf[j] * input_y[i]
            dwij = eta * p_wij
            self.w1[num] += dwij
