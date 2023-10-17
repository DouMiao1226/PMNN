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


def set_seed(seed):
    torch.set_default_dtype(torch.float)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def is_cuda(data):
    if use_gpu:
        data = data.cuda()
    return data

def C(n):
    return (n + 1) ** (1 - alpha) - n ** (1 - alpha)

def exact_u(X):
    return X[:, [1]] ** 2 + (2 * k * X[:, [0]] ** alpha) / gamma(alpha + 1)

def data_train():
    t = np.linspace(lb[0], ub[0], t_N)[:, None]
    x_data = np.linspace(lb[1], ub[1], x_N)[:, None]
    x_data = torch.from_numpy(x_data).float()

    return t, x_data

def data_test():
    t_test = np.linspace(lb[0], ub[0], t_test_N)[:, None]
    x_test = np.linspace(lb[1], ub[1], x_test_N)[:, None]
    t_star, x_star = np.meshgrid(t_test, x_test)
    t_star = t_star.flatten()[:, None]
    x_star = x_star.flatten()[:, None]
    tx_star = np.hstack((t_star, x_star))

    tx_test = is_cuda(torch.from_numpy(tx_star).float())
    tx_test_exact = exact_u(tx_test)

    return t_test, x_test, tx_test, tx_test_exact

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
    def __init__(self, net, x_data, t, lb, ub,
                 tx_test, tx_test_exact
                 ):

        self.tx = None
        self.tx_t0 = None
        self.tx_b1 = None
        self.tx_b2 = None
        self.u_t0 = None
        self.u_x_b1 = None
        self.u_x_b2 = None

        self.optimizer_u = None
        self.optimizer_LBGFS = None

        self.lambda_bc = 1.0
        self.lambda_ic = 1.0

        self.net = net

        self.x_data = x_data
        self.x_N = len(x_data)

        self.t = t
        self.t_N = len(t)
        self.dt = ((ub[0] - lb[0]) / (self.t_N - 1))
        self.lb = lb
        self.ub = ub

        self.tx_test = tx_test
        self.tx_test_exact = tx_test_exact

        self.i_loss_collect = []
        self.b_loss_collect = []
        self.f_loss_collect = []
        self.total_loss_collect = []
        self.error_collect = []
        self.pred_u_collect = []

        self.logger_lambada_b = []
        self.logger_lambada_i = []

        self.tx_test_estimate_collect = []

        self.init_data()

    def init_data(self):
        temp_t = torch.full_like(torch.zeros(self.x_N, 1), self.t[0][0])
        self.tx_t0 = is_cuda(torch.cat((temp_t, self.x_data), dim=1))
        self.tx = torch.cat((temp_t, self.x_data), dim=1)
        for i in range(self.t_N - 1):
            temp_t = torch.full_like(torch.zeros(self.x_N, 1), self.t[i + 1][0])
            temp_tx = torch.cat((temp_t, self.x_data), dim=1)
            self.tx = torch.cat((self.tx, temp_tx), dim=0)

        self.tx = is_cuda(self.tx)

        temp_t = torch.from_numpy(self.t).float()
        temp_lb = torch.full_like(torch.zeros(self.t_N, 1), self.lb[1])
        temp_ub = torch.full_like(torch.zeros(self.t_N, 1), self.ub[1])
        self.tx_b1 = is_cuda(torch.cat((temp_t, temp_lb), dim=1))
        self.tx_b2 = is_cuda(torch.cat((temp_t, temp_ub), dim=1))
        self.u_x_b1 = exact_u(self.tx_b1)
        self.u_x_b2 = exact_u(self.tx_b2)
        self.u_t0 = exact_u(self.tx_t0)

        self.lb = is_cuda(torch.from_numpy(lb).float())
        self.ub = is_cuda(torch.from_numpy(ub).float())

    def train_U(self, x):
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(H)

    def predict_U(self, x):
        return self.train_U(x)

    def Lu_and_F(self):
        x = Variable(self.tx, requires_grad=True)
        u_n = self.train_U(x)
        d = torch.autograd.grad(u_n, x, grad_outputs=torch.ones_like(u_n),
                                      create_graph=True)
        u_x = d[0][:, [1]]
        dd = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                         create_graph=True)
        u_xx = dd[0][:, [1]]

        u_n = u_n.reshape(self.t_N, -1)
        Lu = u_xx.reshape(self.t_N, -1)

        return u_n, Lu

    def PDE_loss(self):
        u_n, Lu = self.Lu_and_F()

        loss = is_cuda(torch.tensor(0.))

        for n in range(1, self.t_N):
            if n == 1:
                pre_Ui = cof * Lu[n] + (C(n - 1) / C(0)) * u_n[0]
            else:
                pre_Ui = is_cuda(cof * Lu[n] + (C(n - 1) / C(0)) * u_n[0])
                for k in range(1, n):
                    pre_Ui += ((C(n - k - 1) - C(n - k)) / C(0)) * u_n[k]
            loss += torch.mean((pre_Ui - u_n[n]) ** 2)

        return loss

    def calculate_loss(self):
        loss_i = torch.mean((self.train_U(self.tx_t0) - self.u_t0) ** 2)
        self.i_loss_collect.append([self.net.iter, loss_i.item()])
        loss_b1 = torch.mean((self.train_U(self.tx_b1) - self.u_x_b1) ** 2)
        loss_b2 = torch.mean((self.train_U(self.tx_b2) - self.u_x_b2) ** 2)
        loss_b = loss_b1 + loss_b2
        self.b_loss_collect.append([self.net.iter, loss_b.item()])

        loss_f = self.PDE_loss()
        self.f_loss_collect.append([self.net.iter, loss_f.item()])

        return 10 * loss_i, 10 * loss_b, loss_f

    # computer backward loss
    def LBGFS_loss(self):
        self.optimizer_LBGFS.zero_grad()
        loss_i, loss_b, loss_f = self.calculate_loss()
        loss = loss_i + loss_b + loss_f
        self.total_loss_collect.append([self.net.iter, loss.item()])
        loss.backward()
        self.net.iter += 1
        print('Iter:', self.net.iter, 'Loss:', loss.item())
        pred = self.train_U(tx_test).cpu().detach().numpy()
        exact = self.tx_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        error = torch.tensor(error)
        self.error_collect.append([self.net.iter, error.item()])


        if self.net.iter % 10 == 0:
            pred_u = self.train_U(tx_test).cpu().detach().numpy()
            pred_u = torch.from_numpy(pred_u).cpu()
            self.pred_u_collect.append([pred_u.tolist()])
            # self.pred_u_collect.append([self.net.iter, pred_u.tolist()])
            # self.pred_u_collect.append(pred_u.tolist())

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
            tolerance_grad=1e-12,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

        start_time = time.time()
        self.optimizer_LBGFS.step(self.LBGFS_loss)
        print('LBGFS done!')
        pred = self.train_U(tx_test).cpu().detach().numpy()
        exact = self.tx_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        print('LBGFS==Test_L2error:', '{0:.2e}'.format(error))

        elapsed = time.time() - start_time
        print('LBGFS==Training time: %.2f' % elapsed)

        save_loss(self.i_loss_collect, self.b_loss_collect, self.f_loss_collect, self.total_loss_collect)

        pred = self.train_U(tx_test).cpu().detach().numpy()
        exact = self.tx_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        print('Test_L2error:', '{0:.2e}'.format(error))

        elapsed = time.time() - start_time
        print('Training time: %.2f' % elapsed)
        return error, elapsed, self.LBGFS_loss().item()

def save_loss(i_loss_collect, b_loss_collect, f_loss_collect, total_loss):
    np.savetxt('loss/i_loss_1D_PDE_L1.txt', i_loss_collect)
    np.savetxt('loss/b_loss_1D_PDE_L1.txt', b_loss_collect)
    np.savetxt('loss/f_loss_1D_PDE_L1.txt', f_loss_collect)
    np.savetxt('loss/total_loss_1D_PDE_L1.txt', total_loss)

def draw_exact():
    u_exact_np = tx_test_exact.cpu().detach().numpy()
    TT, XX = np.meshgrid(t_test, x_test)
    e = np.reshape(u_exact_np, (TT.shape[0], TT.shape[1]))
    # fig = plt.figure(1, figsize=(10, 10))
    ax3 = plt.axes(projection='3d')
    fig1 = ax3.plot_surface(TT, XX, e, rstride=1, cstride=1, cmap='jet')  # cmap是颜色映射表
    # plt.pcolor(TT, XX, e, cmap='jet', shading='auto')
    plt.colorbar(fig1, fraction=0.1, pad=0.15, shrink=0.9, anchor=(0.0, 0.3))
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Exact $u(t,x)$')
    plt.tight_layout()
    plt.savefig('Figure/1D_PDE/1D_PDE_L1_exact.png')
    plt.show()

def draw_pred():
    u_test_np = model.predict_U(tx_test).cpu().detach().numpy()
    TT, XX = np.meshgrid(t_test, x_test)
    e = np.reshape(u_test_np, (TT.shape[0], TT.shape[1]))
    ax3 = plt.axes(projection='3d')
    fig2 = ax3.plot_surface(TT, XX, e, rstride=1, cstride=1, cmap='jet')  # cmap是颜色映射表
    plt.colorbar(fig2, fraction=0.1, pad=0.15, shrink=0.9, anchor=(0.0, 0.3))
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Pred $u(t,x)$')
    plt.tight_layout()
    plt.savefig('Figure/1D_PDE/1D_PDE_L1_pred.png')
    plt.show()

def draw_error():
    u_test_np = model.predict_U(tx_test).cpu().detach().numpy()
    u_exact_np = tx_test_exact.cpu().detach().numpy()
    TT, XX = np.meshgrid(t_test, x_test)
    e = np.reshape(abs(u_test_np - u_exact_np), (TT.shape[0], TT.shape[1]))
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(TT, XX, e, rstride=1, cstride=1, cmap='jet')  # cmap是颜色映射表
    # plt.pcolor(TT, XX, e, cmap='jet', shading='auto')
    # plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Error')
    plt.tight_layout()
    plt.savefig('Figure/1D_PDE/1D_PDE_L1_ERROR.png')
    plt.show()

def draw_error_1():
    u_test_np = model.predict_U(tx_test).cpu().detach().numpy()
    u_exact_np = tx_test_exact.cpu().detach().numpy()
    TT, XX = np.meshgrid(t_test, x_test)
    e = np.reshape(abs(u_test_np - u_exact_np), (TT.shape[0], TT.shape[1]))
    plt.pcolor(TT, XX, e, cmap='jet', shading='auto')#viridis #jet
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Error')
    plt.tight_layout()
    plt.savefig('Figure/1D_PDE/1D_PDE_L1_ERROR_1.png')
    plt.show()

def draw_some_t(t_num_index):
    u_test_np = model.predict_U(tx_test).cpu().detach().numpy()
    u_test_np = u_test_np.reshape((t_test_N, x_test_N))

    u_exact_np = tx_test_exact.cpu().detach().numpy()
    u_exact_np = u_exact_np.reshape((t_test_N, x_test_N))

    u_test_np = u_test_np[t_num_index, : ]
    u_exact_np = u_exact_np[t_num_index, : ]

    plt.plot(x_test, u_exact_np, 'b-', linewidth=5)
    plt.plot(x_test, u_test_np, 'r--', linewidth=5)
    plt.legend(['Exact $u(t,x)$', 'Pred $u(t,x)$'], fontsize=12)
    plt.xlabel('$x$')
    plt.ylabel('$u(t,x)$')
    # plt.title('$t = %.2f $' % (t_test[t_num_index]) + ' based on $L1$')
    plt.tight_layout()
    plt.savefig('Figure/1D_PDE/1D_PDE_L1_t_' + str(t_test[t_num_index]) + 'NN_learn.png')
    plt.show()

def draw_epoch_loss():
    i_loss_collect = np.array(model.i_loss_collect)
    b_loss_collect = np.array(model.b_loss_collect)
    f_loss_collect = np.array(model.f_loss_collect)
    total_loss = i_loss_collect + b_loss_collect + f_loss_collect
    plt.yscale('log')
    plt.xlabel('$Epoch$')
    plt.ylabel('$Loss$')
    # plt.title('$Loss$ based on $L1$')
    plt.plot(i_loss_collect[:, 0], i_loss_collect[:, 1], 'b-', lw=2)
    plt.plot(b_loss_collect[:, 0], b_loss_collect[:, 1], 'g-', lw=2)
    plt.plot(f_loss_collect[:, 0], f_loss_collect[:, 1], 'r-', lw=2)
    plt.legend(['loss_i', 'loss_b', 'loss_f'], fontsize=12)
    plt.savefig('Figure/1D_PDE/1D_PDE_L1_LOSS.png')
    plt.show()

def draw_epoch_loss_1():
    i_loss_collect = np.array(model.i_loss_collect)
    b_loss_collect = np.array(model.b_loss_collect)
    f_loss_collect = np.array(model.f_loss_collect)
    total_loss = i_loss_collect + b_loss_collect + f_loss_collect
    plt.yscale('log')
    plt.xlabel('$Epoch$')
    plt.ylabel('$Loss$')
    # plt.title('$Loss$ based on $L1$')
    plt.plot(i_loss_collect[:, 0], total_loss[:, 1], 'b-', lw=2)
    plt.savefig('Figure/1D_PDE/1D_PDE_L1_LOSS_1.png')
    plt.show()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    use_gpu = True
    # use_gpu = False  # torch.cuda.is_available()
    set_seed(1234)

    layers = [2, 20, 20, 20, 20, 20, 1]
    # layers = [2, 40, 40, 40, 40, 40, 40, 40, 1]
    # layers = [2, 20, 20, 20, 20, 20, 20, 20, 1]
    # layers = [2, 10, 10, 10, 10, 10, 10, 10, 1]

    net = is_cuda(Net_Attention(layers))

    alpha = 0.5
    sigma = 1 - alpha / 2
    k = 1

    lb = np.array([0.0, 0.0]) # low boundary
    ub = np.array([1.0, 1.0]) # up boundary

    '''train data'''
    t_N = 41
    x_N = 11

    t, x_data = data_train()

    '''test data'''
    t_test_N = 100
    x_test_N = 100

    t_test, x_test, tx_test, tx_test_exact = data_test()

    '''Train'''
    model = Model(
        net=net,
        x_data=x_data,
        t=t,
        lb=lb,
        ub=ub,
        tx_test=tx_test,
        tx_test_exact=tx_test_exact,
    )

    cof = (gamma(2 - alpha) / model.dt ** (-alpha)) / C(0)

    model.train(LBGFS_epochs=50000)

    '''画图'''
    # plot_t = int(t_test_N / 3)
    # t_num_index = [0, plot_t, 2 * plot_t, t_test_N-1]
    # for i in range(4):
    #     draw_some_t(t_num_index[i])
    #     plt.show()
    # draw_exact()
    # draw_pred()
    # draw_error()
    # draw_error_1()
    # draw_epoch_loss()
    # draw_epoch_loss_1()


