import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt, gridspec
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from scipy.special import gamma
from torch.autograd import Variable
from torchaudio.functional import gain
from tqdm import trange
import seaborn as sns


def set_seed(seed):
    torch.set_default_dtype(torch.float)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def is_cuda(data):
    if use_gpu:
        data = data.cuda()
    return data

def C(n, k):
    if n == 1:
        return sigma ** (1 - alpha)
    else:
        if k == 0:
            return ((1 + sigma) ** (2 - alpha) - sigma ** (2 - alpha)) / (2 - alpha) - (
                    (1 + sigma) ** (1 - alpha) - sigma ** (1 - alpha)) / 2
        elif 1 <= k <= n - 2:
            return ((k + 1 + sigma) ** (2 - alpha) - 2 * (k + sigma) ** (2 - alpha) + (k - 1 + sigma) ** (
                    2 - alpha)) / (2 - alpha) - (
                           (k + 1 + sigma) ** (1 - alpha) - 2 * (k + sigma) ** (1 - alpha) + (k - 1 + sigma) ** (
                           1 - alpha)) / 2
        else:
            return (3 * (n - 1 + sigma) ** (1 - alpha) - (n - 2 + sigma) ** (1 - alpha)) / 2 - (
                    (n - 1 + sigma) ** (2 - alpha) - (n - 2 + sigma) ** (2 - alpha)) / (2 - alpha)

# def exact_u(X):
#     return X ** (5 + alpha)
#
# def F(X):
#     return (gamma(6 + alpha) / 120) * X ** 5 + X ** (5 + alpha)

def exact_u(X):
    return 1 + X ** 2

def F(X):
    return (gamma(3)/gamma(3 - alpha)) * X ** (2- alpha) - X ** 2 - 1

def data_train():
    t = np.linspace(lb[0], ub[0], t_N)[:, None]

    return t

def data_test():
    t_test = np.linspace(lb[0], ub[0], t_test_N)[:, None]
    t_test = is_cuda(torch.from_numpy(t_test).float())
    t_test_exact = exact_u(t_test)

    return t_test, t_test_exact

class Net_Attention(nn.Module):
    def __init__(self, layers):
        super(Net_Attention, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.attention1 = nn.Linear(layers[0], layers[1])
        self.attention2 = nn.Linear(layers[0], layers[1])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)
        nn.init.xavier_normal_(self.attention1.weight.data, gain=1.0)
        nn.init.zeros_(self.attention1.bias.data)
        nn.init.xavier_normal_(self.attention2.weight.data, gain=1.0)
        nn.init.zeros_(self.attention2.bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        encoder_1 = self.activation(self.attention1(x))
        encoder_2 = self.activation(self.attention2(x))
        a = a * encoder_1 + (1 - a) * encoder_2
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
            a = a * encoder_1 + (1 - a) * encoder_2
        a = self.linear[-1](a)
        return a

class Model:
    def __init__(self, net, t, lb, ub,
                 t_test, t_test_exact
                 ):

        self.t_sigma = None
        self.t_t0 = None
        self.u_t0 = None
        self.F_t_sigma = None

        self.optimizer_u = None
        self.optimizer_LBGFS = None

        self.lambda_bc = 1.0

        self.net = net

        self.t = t
        self.t_N = len(t)
        self.dt = ((ub[0] - lb[0]) / (self.t_N - 1))
        self.lb = lb
        self.ub = ub

        self.t_test = t_test
        self.t_test_exact = t_test_exact

        self.i_loss_collect = []
        self.f_loss_collect = []
        self.total_loss_collect = []
        self.error_collect = []
        self.pred_u_collect = []

        self.logger_lambada = []

        self.t_test_estimate_collect = []

        self.init_data()

    def init_data(self):
        temp_t = self.t
        temp_t_sigma = temp_t + sigma * self.dt
        self.t = is_cuda(torch.from_numpy(temp_t).float())
        self.t_sigma = is_cuda(torch.from_numpy(temp_t_sigma).float())

        self.t_t0 = is_cuda(torch.from_numpy(temp_t[0]).float())
        self.u_t0 = exact_u(self.t_t0)
        self.F_t_sigma = F(self.t_sigma)

        self.lb = is_cuda(torch.from_numpy(lb).float())
        self.ub = is_cuda(torch.from_numpy(ub).float())

        pred_u = self.train_U(t_test).cpu().detach().numpy()
        pred_u = torch.from_numpy(pred_u).cpu()
        self.pred_u_collect.append([pred_u.tolist()])

    def train_U(self, x):
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(H)

    def predict_U(self, x):
        return self.train_U(x)

    def Lu_and_F(self):
        u_n = self.train_U(self.t)
        u_n = u_n.reshape(self.t_N, -1)

        x = Variable(self.t_sigma, requires_grad=True)
        u_n_sigma = self.train_U(x)
        u_n_sigma = u_n_sigma.reshape(self.t_N, -1)

        # F = self.F_t_sigma - u_n_sigma
        F = self.F_t_sigma + u_n_sigma
        F_n_sigma = F.reshape(self.t_N, -1)

        return u_n, u_n_sigma, F_n_sigma

    def PDE_loss(self):
        u_n, u_n_sigma, F_n_sigma = self.Lu_and_F()

        loss = is_cuda(torch.tensor(0.))
        # for n in range(1, self.t_N):
        #     if n == 1:
        #         pre_Ui = u_n[n - 1] + (cof / C(n, 0)) * (F_n_sigma[n - 1])
        #     else:
        #         pre_Ui = is_cuda(u_n[n - 1] + (cof / C(n, 0)) * (F_n_sigma[n - 1]))
        #         for k in range(1, n):
        #             pre_Ui += (C(n, k) / C(n, 0)) * (u_n[n - k - 1] - u_n[n - k])
        #     loss += torch.mean((pre_Ui - u_n[n]) ** 2)

        for n in range(1, self.t_N):
            if n == 1:
                pre_lu = C(n, 0) * (u_n[n] - u_n[n - 1])
            else:
                pre_lu = is_cuda(torch.zeros_like(u_n[0]))
                for k in range(n):
                    pre_lu += C(n, k) * (u_n[n - k] - u_n[n - k - 1])
            loss += torch.mean((pre_lu - cof * (F_n_sigma[n - 1])) ** 2)

        return loss

    def calculate_loss(self):

        loss_i = torch.mean((self.train_U(self.t_t0) - self.u_t0) ** 2)
        self.i_loss_collect.append([self.net.iter, loss_i.item()])

        loss_f = self.PDE_loss()
        self.f_loss_collect.append([self.net.iter, loss_f.item()])

        return loss_i, loss_f

    # computer backward loss
    def LBGFS_loss(self):
        self.optimizer_LBGFS.zero_grad()
        loss_i, loss_f = self.calculate_loss()
        loss = loss_i + 100 * loss_f
        self.total_loss_collect.append([self.net.iter, loss.item()])
        loss.backward()
        self.net.iter += 1
        print('Iter:', self.net.iter, 'Loss:', loss.item())
        pred = self.train_U(t_test).cpu().detach().numpy()
        exact = self.t_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        error = torch.tensor(error)
        self.error_collect.append([self.net.iter, error.item()])

        if self.net.iter % 10 == 0:
            pred_u = self.train_U(t_test).cpu().detach().numpy()
            pred_u = torch.from_numpy(pred_u).cpu()
            self.pred_u_collect.append([pred_u.tolist()])

        return loss

    def train(self, LBGFS_epochs=50000):
        # self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=1,
        #                                          max_iter=LBGFS_epochs)
        self.optimizer_LBGFS = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1,
            max_iter=LBGFS_epochs,
            max_eval=LBGFS_epochs,
            history_size=100,
            tolerance_grad=1e-9,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

        start_time = time.time()

        self.optimizer_LBGFS.step(self.LBGFS_loss)
        print('LBGFS done!')

        pred = self.train_U(t_test).cpu().detach().numpy()
        exact = self.t_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        print('LBGFS==Test_L2error:', '{0:.2e}'.format(error))

        elapsed = time.time() - start_time
        print('LBGFS==Training time: %.2f' % elapsed)

        save_loss(self.i_loss_collect, self.f_loss_collect, self.total_loss_collect)

        pred = self.train_U(t_test).cpu().detach().numpy()
        exact = self.t_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        print('Test_L2error:', '{0:.2e}'.format(error))

        elapsed = time.time() - start_time
        print('Training time: %.2f' % elapsed)
        return error, elapsed, self.LBGFS_loss().item()

def save_loss(i_loss_collect,  f_loss_collect, total_loss):
    np.savetxt('loss/i_loss_ODE_L2-1-sigma.txt', i_loss_collect)
    np.savetxt('loss/f_loss_ODE_L2-1-sigma.txt', f_loss_collect)
    np.savetxt('loss/total_loss_ODE_L2-1-sigma.txt', total_loss)

def draw_exact_pred():
    u_test_np = model.predict_U(t_test).cpu().detach().numpy()
    u_exact_np = t_test_exact.cpu().detach().numpy()
    plt.rc('legend', fontsize=16)
    plt.plot(t_test.cpu().detach().numpy(), u_exact_np, "b-", lw=5)
    plt.plot(t_test.cpu().detach().numpy(), u_test_np, "r--", lw=5)
    plt.legend(['Exact $u(t)$', 'Pred $u(t)$'], fontsize=12)
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel('$u(t)$', fontsize=20)
    # plt.title('Exact $u(x)$ and Pred $u(x)$ based on $L2-1$-sigma')
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig('Figure/ODE/ODE_L2-1-sigma.png')
    plt.show()

def draw_error():
    u_test_np = model.predict_U(t_test).cpu().detach().numpy()
    u_exact_np = t_test_exact.cpu().detach().numpy()
    error_draw = abs(u_test_np - u_exact_np)
    plt.rc('legend', fontsize=16)
    plt.plot(t_test.cpu().detach().numpy(), error_draw, "b-", lw=2)
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel('$Error$', fontsize=20)
    plt.yscale('log')
    # plt.title('$Error$ based on $L2-1$-sigma')
    plt.tight_layout()
    plt.savefig('Figure/ODE/ODE_L2-1-sigma_ERROR.png')
    plt.show()

def draw_epoch_loss():
    i_loss_collect = np.array(model.i_loss_collect)
    f_loss_collect = np.array(model.f_loss_collect)
    plt.rc('legend', fontsize=16)
    plt.yscale('log')
    plt.xlabel('$Epoch$')
    plt.ylabel('$Loss$')
    # plt.title('$Loss$ based on $L2-1$-sigma')
    plt.plot(i_loss_collect[:, 0], i_loss_collect[:, 1], 'b-', lw=2)
    plt.plot(f_loss_collect[:, 0], f_loss_collect[:, 1], 'r-', lw=2)
    plt.legend(['loss_i', 'loss_f'], fontsize=12)
    plt.savefig('Figure/ODE/ODE_L2-1-sigma_LOSS.png')
    plt.show()

def draw_epoch_loss_1():
    i_loss_collect = np.array(model.i_loss_collect)
    f_loss_collect = np.array(model.f_loss_collect)
    total_loss = i_loss_collect + f_loss_collect
    plt.yscale('log')
    plt.xlabel('$Epoch$')
    plt.ylabel('$Loss$')
    # plt.title('$Loss$ based on $L2-1$-sigma')
    plt.plot(f_loss_collect[:, 0], total_loss[:, 1], 'b-', lw=2)
    plt.savefig('Figure/ODE/ODE_L2-1-sigma_LOSS_1.png')
    plt.show()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    use_gpu = True
    # use_gpu = False
    # torch.cuda.is_available()
    set_seed(1234)

    layers = [1, 20, 20, 20, 20, 20, 1]
    # layers = [1, 20, 20, 20, 20, 20, 20, 20, 1]
    # layers = [1, 10, 10, 10, 10, 10, 10, 10, 1]
    # layers = [1, 20, 20, 20, 1]

    net = is_cuda(Net_Attention(layers))

    '''方程参数'''
    alpha = 0.5
    sigma = 1 - alpha / 2

    lb = np.array([0.0]) # low boundary
    ub = np.array([1.0]) # up boundary

    '''train data'''
    t_N = 41

    t = data_train()

    '''test data'''
    t_test_N = 500

    t_test, t_test_exact = data_test()


    '''Train'''
    model = Model(
        net=net,
        t=t,
        lb=lb,
        ub=ub,
        t_test=t_test,
        t_test_exact=t_test_exact,
    )
    cof = (gamma(2 - alpha) * (model.dt ** alpha))
    model.train(LBGFS_epochs=50000)

    # test time
    start_time0 = time.time()
    u_test_np = model.predict_U(t_test).cpu().detach().numpy()
    u_exact_np = t_test_exact.cpu().detach().numpy()
    error_draw = np.linalg.norm(u_test_np - u_exact_np, 2) / np.linalg.norm(u_exact_np, 2)
    print('L2error:', '{0:.2e}'.format(error_draw))
    elapsed0 = time.time() - start_time0
    print('Time: %.4f' % elapsed0)

    '''画图'''
    draw_exact_pred()
    draw_error()
    draw_epoch_loss()
    draw_epoch_loss_1()


