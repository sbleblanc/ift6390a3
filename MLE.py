import numpy as np
from recordclass import recordclass
from collections import namedtuple
from matplotlib import pyplot

FLayers = recordclass('FLayers', 'ha hs oa os')
BLayers = recordclass('BLayers', 'ha hs oa os W1 W2 b1 b2')
Params = recordclass('Params', 'W1 W2 b1 b2')
HyperParams = namedtuple('HyperParams', 'l11, l12, l21, l22, K, eta')
DataPoint = namedtuple('DataPoint', 'x y')

def softmax(x):
    return np.divide(np.exp(x), np.sum(np.exp(x)))


class MLP:

    def __init__(self, data, dh, m, hp):
        self.data_points = data
        self.d = len(data[0].x)
        self.hyper_params = hp
        self.params = Params(b1=np.zeros([dh, 1]), b2=np.zeros([m, 1]), W1=np.zeros([dh, self.d]), W2=np.zeros([m, dh]))
        self.params.W1 = np.random.uniform(-1./np.sqrt(dh), 1./np.sqrt(dh), (dh, self.d))
        self.params.W2 = np.random.uniform(-1. / np.sqrt(m), 1. / np.sqrt(m), (m, dh))
        self.dh = dh
        self.m = m

    def fprop(self, x):
        fl = FLayers(ha=np.zeros([self.dh, 1]), hs=np.zeros([self.dh, 1]), oa=np.zeros([self.m, 1]), os=np.zeros([self.m, 1]))
        fl.ha = self.params.b1 + self.params.W1 @ x
        fl.hs = np.maximum(0, fl.ha)
        fl.oa = self.params.b2 + self.params.W2 @ fl.hs
        fl.os = softmax(fl.oa)
        return fl

    def bprop(self, x,  y, fl):
        bl = BLayers(ha=np.zeros([self.dh, 1]), hs=np.zeros([self.dh, 1]), oa=np.zeros([self.m, 1]), os=np.zeros([self.m, 1]),
                     W1=np.zeros([self.dh, self.d]), W2=np.zeros([self.m, self.dh]), b1=np.zeros([self.dh, 1]), b2=np.zeros([self.m, 1]))
        bl.oa = fl.os - np.eye(1, self.m, int(y)).T
        bl.W2 = np.outer(bl.oa, fl.hs) \
                        + self.hyper_params.l21 * np.sign(self.params.W2) \
                        + 2*self.hyper_params.l22*self.params.W2
        bl.b2 = bl.oa
        bl.hs = (bl.oa.T @ self.params.W2).T
        bl.ha = np.multiply(bl.hs, np.where(fl.ha > 0, 1, 0))
        bl.W1 = np.outer(bl.ha, x) + self.hyper_params.l11 * np.sign(self.params.W1) + 2*self.hyper_params.l12*self.params.W1
        bl.b1 = bl.ha
        return bl

    def single_batch(self, batch_data):
        bl_avg = BLayers(ha=np.zeros([self.dh, 1]), hs=np.zeros([self.dh, 1]), oa=np.zeros([self.m, 1]),
                         os=np.zeros([self.m, 1]),
                         W1=np.zeros([self.dh, self.d]), W2=np.zeros([self.m, self.dh]), b1=np.zeros([self.dh, 1]),
                         b2=np.zeros([self.m, 1]))
        K = len(batch_data)
        for i in range(K):
            dp = batch_data[i]
            fl = self.fprop(dp.x)
            bl = self.bprop(dp.x, dp.y, fl)
            bl_avg.W1 += bl.W1
            bl_avg.W2 += bl.W2
            bl_avg.b1 += bl.b1
            bl_avg.b2 += bl.b2
        bl_avg.W1 /= K
        bl_avg.W2 /= K
        bl_avg.b1 /= K
        bl_avg.b2 /= K
        return bl_avg

    def current_cost(self):
        cummul = 0.
        for dp in self.data_points:
            fl = self.fprop(dp.x)
            cummul += -np.log(fl.os[int(dp.y)])
        cummul /= len(self.data_points)
        return cummul

    def train(self, tol=1e-05):
        epoch = 0
        last_cost = 0.
        current_cost = 1.
        while np.abs(last_cost - current_cost) > tol:
            last_cost = current_cost
            epoch += 1
            for i in range(0, len(self.data_points), self.hyper_params.K):
                end_batch = min(i + self.hyper_params.K, len(self.data_points))
                batch_data = self.data_points[i:end_batch]
                bl = self.single_batch(batch_data)
                self.params.W1 -= self.hyper_params.eta * bl.W1
                self.params.W2 -= self.hyper_params.eta * bl.W2
                self.params.b1 -= self.hyper_params.eta * bl.b1
                self.params.b2 -= self.hyper_params.eta * bl.b2
            current_cost = self.current_cost()
            print('Epoch {}: Loss={}'.format(epoch, current_cost))
            # print('W2 gradient norm: {}'.format(np.linalg.norm(bl.W2)))

    def predict(self, x):
        fl = self.fprop(x)
        return fl.os.argsort(axis=0)[-1]

    def log_likelihood(self, fl, y):
        return -np.log(fl.os[int(y)])

    def finite_difference(self, K):
        eps = 1e-05
        results = []
        param_dic = self.params._asdict()

        for p in self.params._fields:
            for i, x in np.ndenumerate(param_dic[p]):
                log_init_avg = 0.
                est = 0.
                for j in range(K):
                    dp = self.data_points[j]
                    fl = self.fprop(dp.x)
                    log_init = self.log_likelihood(fl, dp.y)
                    log_init_avg += log_init
                    param_dic[p][i] = param_dic[p][i] + eps
                    fl = self.fprop(dp.x)
                    log_pert = self.log_likelihood(fl, dp.y)
                    param_dic[p][i] = param_dic[p][i] - eps
                    est += (log_pert - log_init) / eps

                est /= K

                grad = self.single_batch(self.data_points[0:K])._asdict()[p][i]

                if est == 0:
                    ratio = 1
                else:
                    ratio = grad / est
                results.append((p, i, grad, est, ratio))
        return results


def plot_circle_data_decision(classifier, data):
    x1_0 = []
    x2_0 = []
    x1_1 = []
    x2_1 = []
    # for d in data:



raw_data = np.loadtxt(open('circles.txt','r'))
data = [DataPoint(x=rd[0:2].reshape(2,1), y=rd[2]) for rd in raw_data]

np.random.seed(1234)
hp = HyperParams(l11=0., l12=0., l21=0., l22=0., K=10, eta=1e-01)
test = MLP(data, 4, 2, hp)

#Faire attention au hyperparam l11,l12,l21,l22. Devrait etre 0 sinon la regularization embarque et le gradient se compare pas vraiment au finite difference
res = test.finite_difference(1)
for r in res:
    print('{}{}'.format(r[0],r[1]))
    print('\tGrad:{}'.format(r[2]))
    print('\tEst:{}'.format(r[3]))
    print('\tRatio:{}'.format(r[4]))

# test.train(tol=1e-06)
# print("some predictions...")
# for dp in data[0:20]:
#     print('True:{} // Prediction:{}'.format(dp.y, test.predict(dp.x)))
