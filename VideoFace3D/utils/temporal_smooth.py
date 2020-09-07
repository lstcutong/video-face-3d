import torch
import torch.nn as nn
import numpy as np
import copy


# from networks import TimeSmoothNetWork
class Model(nn.Module):
    def __init__(self, H, W, initation=None):
        super(Model, self).__init__()

        self.H = H
        self.W = W
        self.l1 = nn.Linear(self.W, self.H)

        if initation is not None:
            self.l1.weight.data = torch.from_numpy(initation)

        self.smo_param = self.l1.weight

    def divergence(self, x):
        g_x = x[:, 1:self.W] - x[:, 0:self.W - 1]

        g_xx = g_x[:, 1:self.W - 1] - g_x[:, 0:self.W - 2]

        return g_xx

    def gradient(self, x):
        g_x = x[:, 1:self.W] - x[:, 0:self.W - 1]
        return g_x

    def forward(self, ref_param):
        sim_loss = torch.sum((ref_param - self.smo_param.float()) ** 2)

        smo_loss = torch.norm(self.divergence(self.smo_param.float()), p=2)
        smo_loss2 = torch.norm(self.gradient(self.smo_param.float()), p=2)
        return sim_loss, smo_loss


'''
提供5种平滑方式
均值平滑，中值平滑，高斯平滑，，基于优化的平滑，卷积网络平滑
输入: ref_param 参考参数，类型 ndarray, shape:[param_num,frames]
输出: new_param 平滑参数, 类型 ndarray, shape:[param_num,frames]
'''


def smooth_optimize(ref_param):
    ref_param = ref_param.astype(np.float32)
    H, W = ref_param.shape

    model = Model(H, W, initation=ref_param).cuda()
    ref_param = torch.from_numpy(ref_param).float().cuda()

    iterations = 300
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # print(ref_param.shape)

    for it in range(iterations):
        sim_loss, smo_loss = model(ref_param)

        loss = 1 * sim_loss + 1.3 * smo_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    new_param = model.smo_param.cpu().detach().numpy()

    return new_param


def smooth_medium_filter(ref_param, k=5):
    assert k % 2 == 1, "未实现偶数步长"
    H, W = ref_param.shape

    s = int(k / 2)

    new_param = copy.deepcopy(ref_param)
    for i in range(0, W):
        start = np.maximum(0, i - s)
        end = np.minimum(W, i + s + 1)

        new_param[:, i] = np.median(ref_param[:, start:end], axis=1)

    return new_param


def smooth_mean_filter(ref_param, k=5):
    assert k % 2 == 1, "未实现偶数步长"
    H, W = ref_param.shape

    s = int(k / 2)

    new_param = copy.deepcopy(ref_param)
    for i in range(0, W):
        start = np.maximum(0, i - s)
        end = np.minimum(W, i + s + 1)

        new_param[:, i] = np.mean(ref_param[:, start:end], axis=1)

    return new_param


def smooth_gaussian_filter(ref_param, k=5):
    miu, sigma = 0, 1
    assert k % 2 == 1, "未实现偶数步长"
    H, W = ref_param.shape

    center = int(k / 2)
    x = np.array([i - center for i in range(k)])

    weights = np.exp(-(x - miu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    weights = weights / np.sum(weights)

    new_param = copy.deepcopy(ref_param)

    for i in range(center, W - center):
        start = np.maximum(0, i - center)
        end = np.minimum(W, i + center + 1)

        new_param[:, i] = ref_param[:, start:end] @ weights

    return new_param


def smooth_DCNN(ref_param):
    pass


def calculate_smooth_loss(x):
    H, W = x.shape

    g_x = x[:, 1:W] - x[:, 0:W - 1]

    g_xx = g_x[:, 1:W - 1] - g_x[:, 0:W - 2]

    return np.sum(g_xx ** 2)
    # return np.sum(np.abs(g_xx))


def calculate_sim_loss(x, y):
    return np.sum((x - y) ** 2)




