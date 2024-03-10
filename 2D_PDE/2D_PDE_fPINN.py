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
    return X[:, [0]] ** 2 * torch.exp(X[:, [1]] + X[:, [2]])

def F(X):
    return torch.exp(X[:, [1]] + X[:, [2]]) * ((2 * X[:, [0]] ** (2 - alpha)) / gamma(3 - alpha) - 2 * X[:, [0]] ** 2)

def data_train():
    t = np.linspace(lb[0], ub[0], t_N)[:, None]
    x = np.linspace(lb[1], ub[1], x_y_N)[:, None]
    y = np.linspace(lb[2], ub[2], x_y_N)[:, None]
    X, Y = np.meshgrid(x, y)
    vector_x = np.vstack(np.expand_dims(X, axis=2))
    vector_y = np.vstack(np.expand_dims(Y, axis=2))
    xy = np.concatenate((vector_y, vector_x), axis=-1)
    xlb = np.linspace(lb[1], lb[1], x_y_N)[:, None]
    xub = np.linspace(ub[1], ub[1], x_y_N)[:, None]
    ylb = np.linspace(lb[2], lb[2], x_y_N)[:, None]
    yub = np.linspace(ub[2], ub[2], x_y_N)[:, None]

    xy_xlb = np.concatenate((xlb, y), axis=-1)
    xy_xub = np.concatenate((xub, y), axis=-1)
    xy_ylb = np.concatenate((x, ylb), axis=-1)
    xy_yub = np.concatenate((x, yub), axis=-1)

    xy = torch.from_numpy(xy).float()
    xy_xlb = torch.from_numpy(xy_xlb).float()
    xy_xub = torch.from_numpy(xy_xub).float()
    xy_ylb = torch.from_numpy(xy_ylb).float()
    xy_yub = torch.from_numpy(xy_yub).float()

    return t, x, y, xy, xy_xlb, xy_xub, xy_ylb, xy_yub

def data_test():
    t_test = np.linspace(lb[0], ub[0], t_test_N)[:, None]
    x_test = np.linspace(lb[1], ub[1], x_y_test_N)[:, None]
    y_test = np.linspace(lb[2], ub[2], x_y_test_N)[:, None]
    X_test, Y_test = np.meshgrid(x_test, y_test)
    vector_x_test = np.vstack(np.expand_dims(X_test, axis=2))
    vector_y_test = np.vstack(np.expand_dims(Y_test, axis=2))
    xy_test = np.concatenate((vector_y_test, vector_x_test), axis=-1)
    xy_test = np.tile(xy_test, (t_test_N, 1))
    t_vector_test = t_test.repeat(x_y_test_N * x_y_test_N, axis=0)
    t_vector_test = t_vector_test.reshape(x_y_test_N * x_y_test_N * t_test_N, 1)
    txy_test = np.concatenate((t_vector_test, xy_test), axis=-1)
    txy_test = is_cuda(torch.from_numpy(txy_test).float())
    txy_test_exact = exact_u(txy_test)

    return x_test, y_test, t_test, txy_test, txy_test_exact

class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            # (self.linear[i].weight.data, gain(1.0) )
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
        a = self.linear[-1](a)
        return a

class Model:
    def __init__(self, net, xy, xy_xlb, xy_xub, xy_ylb, xy_yub, t, lb, ub,
                 txy_test, txy_test_exact
                 ):

        self.txy = None #txy放一起
        self.txy_t0 = None #初始时刻，txy放一起
        self.txy_xlb = None #边界条件
        self.txy_xub = None
        self.txy_ylb = None
        self.txy_yub = None
        self.u_t0 = None
        self.u_xlb = None
        self.u_xub = None
        self.u_ylb = None
        self.u_yub = None
        self.F_txy = None

        self.optimizer_u = None
        self.optimizer_LBGFS = None

        self.lambda_bc = 1.0

        self.net = net

        self.xy = xy
        self.xy_N = len(xy)

        self.xy_xlb = xy_xlb
        self.xy_xub = xy_xub
        self.xy_ylb = xy_ylb
        self.xy_yub = xy_yub
        self.x_N = len(xy_xlb)

        self.t = t
        self.t_N = len(t)
        self.dt = ((ub[0] - lb[0]) / (self.t_N - 1))
        self.lb = lb
        self.ub = ub

        self.txy_test = txy_test
        self.txy_test_exact = txy_test_exact

        self.i_loss_collect = []
        self.b_loss_collect = []
        self.f_loss_collect = []
        self.total_loss_collect = []
        self.error_collect = []
        self.pred_u_collect = []

        self.logger_lambada = []

        self.txy_test_estimate_collect = []

        self.init_data()

    def init_data(self):
        ## 控制方程 和 初始条件
        temp_t = torch.full_like(torch.zeros(self.xy_N, 1), self.t[0][0])
        self.txy_t0 = torch.cat((temp_t, self.xy), dim=1)
        self.txy_t0 = is_cuda(self.txy_t0)

        self.txy = torch.cat((temp_t, self.xy), dim=1)
        for i in range(self.t_N - 1):
            temp_t = torch.full_like(torch.zeros(self.xy_N, 1), self.t[i + 1][0])
            temp_txy = torch.cat((temp_t, self.xy), dim=1)
            self.txy = torch.cat((self.txy, temp_txy), dim=0)

        self.txy = is_cuda(self.txy)

        ## 边界条件
        temp_t = torch.full_like(torch.zeros(self.x_N, 1), self.t[0][0])
        self.txy_xlb = torch.cat((temp_t, self.xy_xlb), dim=1)
        self.txy_xub = torch.cat((temp_t, self.xy_xub), dim=1)
        self.txy_ylb = torch.cat((temp_t, self.xy_ylb), dim=1)
        self.txy_yub = torch.cat((temp_t, self.xy_yub), dim=1)
        for i in range(self.t_N - 1):
            temp_t = torch.full_like(torch.zeros(self.x_N, 1), self.t[i + 1][0])
            temp_txy_xlb = torch.cat((temp_t, self.xy_xlb), dim=1)
            temp_txy_xub = torch.cat((temp_t, self.xy_xub), dim=1)
            temp_txy_ylb = torch.cat((temp_t, self.xy_ylb), dim=1)
            temp_txy_yub = torch.cat((temp_t, self.xy_yub), dim=1)
            self.txy_xlb = torch.cat((self.txy_xlb, temp_txy_xlb), dim=0)
            self.txy_xub = torch.cat((self.txy_xub, temp_txy_xub), dim=0)
            self.txy_ylb = torch.cat((self.txy_ylb, temp_txy_ylb), dim=0)
            self.txy_yub = torch.cat((self.txy_yub, temp_txy_yub), dim=0)
        self.txy_xlb = is_cuda(self.txy_xlb)
        self.txy_xub = is_cuda(self.txy_xub)
        self.txy_ylb = is_cuda(self.txy_ylb)
        self.txy_yub = is_cuda(self.txy_yub)

        self.u_xlb = exact_u(self.txy_xlb)
        self.u_xub = exact_u(self.txy_xub)
        self.u_ylb = exact_u(self.txy_ylb)
        self.u_yub = exact_u(self.txy_yub)
        self.u_t0 = exact_u(self.txy_t0)
        self.F_txy = F(self.txy)

        self.lb = is_cuda(torch.from_numpy(lb).float())
        self.ub = is_cuda(torch.from_numpy(ub).float())

    def train_U(self, x):
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(H)

    def predict_U(self, x):
        return self.train_U(x)

    def Lu_and_F(self):
        x = Variable(self.txy, requires_grad=True)
        u_n = self.train_U(x)

        d = torch.autograd.grad(u_n, x, grad_outputs=torch.ones_like(u_n), create_graph=True)
        u_t = d[0][:, [0]]
        u_x = d[0][:, [1]]
        u_y = d[0][:, [2]]
        dd = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)
        u_xx = dd[0][:, [1]]
        dd_y = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)
        u_yy = dd_y[0][:, [2]]

        u_n = u_n.reshape(self.t_N, -1)
        u_t = u_t.reshape(self.t_N, -1)
        Lu = u_xx.reshape(self.t_N, -1) + u_yy.reshape(self.t_N, -1)

        F = self.F_txy
        F_n = F.reshape(self.t_N, -1)

        return u_n, u_t, Lu, F_n

    def PDE_loss(self):
        u_n, u_t, Lu, F_n = self.Lu_and_F()

        loss = is_cuda(torch.tensor(0.))

        for n in range(1, self.t_N):
            if n == 1:
                pre_Ui = cof * (Lu[n] + F_n[n]) + (C(n - 1) / C(0)) * u_n[0]
            else:
                pre_Ui = is_cuda(cof * (Lu[n] + F_n[n]) + (C(n - 1) / C(0)) * u_n[0])
                for k in range(1, n):
                    pre_Ui += ((C(n - k - 1) - C(n - k)) / C(0)) * u_n[k]
            loss += torch.mean((pre_Ui - u_n[n]) ** 2)

        return loss

    def calculate_loss(self):
        loss_i = torch.mean((self.train_U(self.txy_t0) - self.u_t0) ** 2)
        self.i_loss_collect.append([self.net.iter, loss_i.item()])
        loss_xlb = torch.mean((self.train_U(self.txy_xlb) - self.u_xlb) ** 2)
        loss_xub = torch.mean((self.train_U(self.txy_xub) - self.u_xub) ** 2)
        loss_ylb = torch.mean((self.train_U(self.txy_ylb) - self.u_ylb) ** 2)
        loss_yub = torch.mean((self.train_U(self.txy_yub) - self.u_yub) ** 2)
        loss_b = loss_xlb + loss_xub + loss_ylb + loss_yub
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

        pred = self.train_U(self.txy_test).cpu().detach().numpy()
        exact = self.txy_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        error = torch.tensor(error)
        self.error_collect.append([self.net.iter, error.item()])

        if self.net.iter % 10 == 0:
            pred_u = self.train_U(self.txy_test).cpu().detach().numpy()
            pred_u = torch.from_numpy(pred_u).cpu()
            self.pred_u_collect.append([pred_u.tolist()])

        return loss

    def train(self, LBGFS_epochs=50000):

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

        pred = self.train_U(txy_test).cpu().detach().numpy()
        exact = self.txy_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        print('LBGFS==Test_L2error:', '{0:.2e}'.format(error))

        elapsed = time.time() - start_time
        print('LBGFS==Training time: %.2f' % elapsed)

        save_error(self.error_collect)
        save_loss_fPINN(self.i_loss_collect, self.b_loss_collect, self.f_loss_collect, self.total_loss_collect)

        return error, elapsed, self.LBGFS_loss().item()


def save_error(error_collect):
    np.savetxt('loss/error_2D_PDE_fPINN.txt', error_collect)

def save_loss_fPINN(i_loss_collect, b_loss_collect, f_loss_collect, total_loss):
    np.savetxt('loss/i_loss_2D_PDE_fPINN.txt', i_loss_collect)
    np.savetxt('loss/b_loss_2D_PDE_fPINN.txt', b_loss_collect)
    np.savetxt('loss/f_loss_2D_PDE_fPINN.txt', f_loss_collect)
    np.savetxt('loss/total_loss_2D_PDE_fPINN.txt', total_loss)

def exact_draw_some_t(t_num_index):
    u_exact_np = txy_test_exact.cpu().detach().numpy()
    u_exact_np = u_exact_np.reshape((x_y_test_N, x_y_test_N, t_test_N))

    XX, YY = np.meshgrid(x_test, y_test)

    e = u_exact_np[:, :, t_num_index]
    ax3 = plt.axes(projection='3d')
    fig1 = ax3.plot_surface(XX, YY, e, rstride=1, cstride=1, cmap='jet')  # cmap是颜色映射表
    plt.colorbar(fig1, fraction=0.1, pad=0.15, shrink=0.9, anchor=(0.0, 0.3))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('$t = %.2f $' % (t_test[t_num_index]))
    plt.title('Exact $u(t,x,y)$')
    plt.tight_layout()
    plt.savefig('Figure/2D_PDE/2D_PDE_fPINN_t_' + str(t_test[t_num_index]) + 'NN_exact.png')
    plt.show()

def pred_draw_some_t(t_num_index):
    u_test_np = model.predict_U(txy_test).cpu().detach().numpy()
    u_test_np = u_test_np.reshape((x_y_test_N, x_y_test_N, t_test_N))

    XX, YY = np.meshgrid(x_test, y_test)

    e = u_test_np[:, :, t_num_index]
    ax3 = plt.axes(projection='3d')
    fig2 = ax3.plot_surface(XX, YY, e, rstride=1, cstride=1, cmap='jet')  # cmap是颜色映射表
    plt.colorbar(fig2, fraction=0.1, pad=0.15, shrink=0.9, anchor=(0.0, 0.3))
    plt.title('$t = %.2f $' % (t_test[t_num_index]))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Pred $u(t,x,y)$')
    plt.tight_layout()
    plt.savefig('Figure/2D_PDE/2D_PDE_fPINN_t_' + str(t_test[t_num_index]) + 'NN_learn.png')
    plt.show()

def error_draw_some_t(t_num_index):
    u_test_np = model.predict_U(txy_test).cpu().detach().numpy()
    u_test_np = u_test_np.reshape((x_y_test_N, x_y_test_N, t_test_N))
    u_exact_np = txy_test_exact.cpu().detach().numpy()
    u_exact_np = u_exact_np.reshape((x_y_test_N, x_y_test_N, t_test_N))

    XX, YY = np.meshgrid(x_test, y_test)

    e = np.reshape(abs(u_test_np[:, :, t_num_index] - u_exact_np[:, :, t_num_index]), (x_y_test_N, x_y_test_N))
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(XX, YY, e, rstride=1, cstride=1, cmap='jet')  # cmap是颜色映射表
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Error')
    plt.title('$t = %.2f $' % (t_test[t_num_index]))
    plt.tight_layout()
    plt.savefig('Figure/2D_PDE/2D_PDE_fPINN_t_' + str(t_test[t_num_index]) + 'NN_Error.png')
    plt.show()


def error_1_draw_some_t(t_num_index):
    u_test_np = model.predict_U(txy_test).cpu().detach().numpy()
    u_test_np = u_test_np.reshape((x_y_test_N, x_y_test_N, t_test_N))
    u_exact_np = txy_test_exact.cpu().detach().numpy()
    u_exact_np = u_exact_np.reshape((x_y_test_N, x_y_test_N, t_test_N))

    XX, YY = np.meshgrid(x_test, y_test)

    e = np.reshape(abs(u_test_np[:, :, t_num_index] - u_exact_np[:, :, t_num_index]), (x_y_test_N, x_y_test_N))
    plt.pcolor(XX, YY, e, cmap='jet', shading='auto')  # viridis
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('$t = %.2f $' % (t_test[t_num_index]))
    plt.title('Error')
    plt.tight_layout()
    plt.savefig('Figure/2D_PDE/2D_PDE_fPINN_t_' + str(t_test[t_num_index]) + 'NN_Error_1.png')
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
    plt.savefig('Figure/2D_PDE/2D_PDE_fPINN_LOSS.png')
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
    plt.savefig('Figure/2D_PDE/2D_PDE_fPINN_LOSS_1.png')
    plt.show()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    use_gpu = True
    # use_gpu = False
    # torch.cuda.is_available()
    set_seed(1234)

    layers = [3, 20, 20, 20, 20, 20, 1]
    net = is_cuda(Net(layers))

    alpha = 0.5
    lb = np.array([0.0, 0.0, 0.0]) # low boundary t,x,y
    ub = np.array([1.0, 1.0, 1.0]) # up boundary t,x,y

    '''train data'''
    t_N = 21
    x_y_N = 11

    t, x, y, xy, xy_xlb, xy_xub, xy_ylb, xy_yub = data_train()

    '''test data'''
    t_test_N = 100
    x_y_test_N = 100

    x_test, y_test, t_test, txy_test, txy_test_exact = data_test()

    '''Train'''
    model = Model(
        net=net,
        xy=xy,
        xy_xlb=xy_xlb,
        xy_xub=xy_xub,
        xy_ylb=xy_ylb,
        xy_yub=xy_yub,
        t=t,
        lb=lb,
        ub=ub,
        txy_test=txy_test,
        txy_test_exact=txy_test_exact,
    )

    cof = (gamma(2 - alpha) * model.dt ** alpha) / C(0)

    model.train(LBGFS_epochs=50000)


    '''画图'''
    # plot_t = int(t_test_N / 3)
    # t_num_index = [0, plot_t, 2 * plot_t, t_test_N - 1]
    # for i in range(4):
    #     error_draw_some_t(t_num_index[i])
    #     plt.show()
    #     pred_draw_some_t(t_num_index[i])
    #     plt.show()
    #     exact_draw_some_t(t_num_index[i])
    #     plt.show()
    #     error_1_draw_some_t(t_num_index[i])
    #     plt.show()
    #
    # draw_epoch_loss()
    # draw_epoch_loss_1()




