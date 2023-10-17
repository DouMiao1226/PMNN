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

def exact_u(X):
    return X[:, [0]] ** 2 + torch.cos(X[:, [1]]) + torch.cos(X[:, [2]]) + torch.cos(X[:, [3]])

def exact_u0(X):
    return torch.cos(X[:, [1]]) + torch.cos(X[:, [2]]) + torch.cos(X[:, [0]])

def make_boundary(t, x, y, z):
    t_star, x_star, y_star = np.meshgrid(t, x, y)
    t_star = torch.from_numpy(t_star.flatten()[:, None]).float()
    x_star = torch.from_numpy(x_star.flatten()[:, None]).float()
    y_star = torch.from_numpy(y_star.flatten()[:, None]).float()
    z_b1 = torch.cat((t_star, x_star, y_star, torch.full_like(torch.zeros_like(x_star), lb[3])),
                     dim=1)
    z_b2 = torch.cat((t_star, x_star, y_star, torch.full_like(torch.zeros_like(x_star), ub[3])),
                     dim=1)

    t_star, x_star, z_star = np.meshgrid(t, x, z)
    t_star = torch.from_numpy(t_star.flatten()[:, None]).float()
    x_star = torch.from_numpy(x_star.flatten()[:, None]).float()
    z_star = torch.from_numpy(z_star.flatten()[:, None]).float()
    y_b1 = torch.cat((t_star, x_star, torch.full_like(torch.zeros_like(x_star), lb[3]), z_star),
                     dim=1)
    y_b2 = torch.cat((t_star, x_star, torch.full_like(torch.zeros_like(x_star), lb[3]), z_star),
                     dim=1)

    t_star, y_star, z_star = np.meshgrid(t, y, z)
    t_star = torch.from_numpy(t_star.flatten()[:, None]).float()
    y_star = torch.from_numpy(y_star.flatten()[:, None]).float()
    z_star = torch.from_numpy(z_star.flatten()[:, None]).float()
    x_b1 = torch.cat((t_star, torch.full_like(torch.zeros_like(x_star), lb[3]), y_star, z_star),
                     dim=1)
    x_b2 = torch.cat((t_star, torch.full_like(torch.zeros_like(x_star), ub[3]), y_star, z_star),
                     dim=1)

    txyz_b = torch.cat((x_b1, x_b2, y_b1, y_b2, z_b1, z_b2), dim=0)
    return txyz_b

def data_train():
    t = np.linspace(lb[0], ub[0], t_N)[:, None]
    x = np.linspace(lb[1], ub[1], x_y_z_N)[:, None]
    y = np.linspace(lb[2], ub[2], x_y_z_N)[:, None]
    z = np.linspace(lb[3], ub[3], x_y_z_N)[:, None]
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    xyz = np.stack((X, Y, Z), axis=1)
    xyz = torch.from_numpy(xyz).float()

    xyz_N = int(np.cbrt(x_y_z_N))

    xb = np.linspace(lb[1], ub[1], xyz_N)[:, None]
    yb = np.linspace(lb[2], ub[2], xyz_N)[:, None]
    zb = np.linspace(lb[3], ub[3], xyz_N)[:, None]

    txyz_b = make_boundary(t, xb, yb, zb)

    return t, x, y, z, xyz, txyz_b

def data_test():
    t_test = np.linspace(lb[0], ub[0], t_test_N)[:, None]
    x_test = np.linspace(lb[1], ub[1], x_y_z_test_N)[:, None]
    y_test = np.linspace(lb[2], ub[2], x_y_z_test_N)[:, None]
    z_test = np.linspace(lb[3], ub[3], x_y_z_test_N)[:, None]
    X_test, Y_test, Z_test = np.meshgrid(x_test, y_test, z_test)
    X_test = X_test.flatten()
    Y_test = Y_test.flatten()
    Z_test = Z_test.flatten()
    xyz_test = np.stack((X_test, Y_test, Z_test), axis=1)
    xyz_test = torch.from_numpy(xyz_test).float()
    t_vector_test = torch.full_like(torch.zeros(x_y_z_test_N ** 3, 1), t_test[0][0])
    txyz_test = torch.cat((t_vector_test, xyz_test), dim=1)
    # txyz_test = torch.from_numpy(txyz_test).float()
    for i in range(t_test_N - 1):
        t_vector_test = torch.full_like(torch.zeros(x_y_z_test_N ** 3, 1), t_test[i + 1][0])
        txyz_test_temp = torch.cat((t_vector_test, xyz_test), dim=1)
        txyz_test = torch.cat((txyz_test, txyz_test_temp), dim=0)
    txyz_test = is_cuda(txyz_test)
    txyz_test_exact = exact_u(txyz_test)

    return x_test, y_test, z_test, t_test, txyz_test, txyz_test_exact

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
    def __init__(self, net, xyz, txyz_b, t, lb, ub,
                 txyz_test, txyz_test_exact
                 ):

        self.txyz = None #txyz放一起
        self.txyz_sigma = None  # t + sigma * dt时，txyz放一起
        self.txyz_b = txyz_b  # 边界条件
        self.u_b_exact = None

        self.txyz_t0 = None #初始条件
        self.u_t0 = None

        self.optimizer_u = None
        self.optimizer_LBGFS = None

        self.net = net

        self.xyz = xyz
        self.xyz_N = len(xyz)

        self.t = t
        self.t_N = len(t)
        self.dt = ((ub[0] - lb[0]) / (self.t_N - 1))
        self.lb = lb
        self.ub = ub

        self.txyz_test = txyz_test
        self.txyz_test_exact = txyz_test_exact

        self.i_loss_collect = []
        self.b_loss_collect = []
        self.f_loss_collect = []
        self.total_loss_collect = []
        self.error_collect = []
        self.pred_u_collect = []

        self.txyz_test_estimate_collect = []

        self.init_data()

    def init_data(self):
        ## 初始条件
        temp_t = torch.full_like(torch.zeros(self.xyz_N, 1), self.t[0][0])
        self.txyz_t0 = torch.cat((temp_t, self.xyz), dim=1)
        self.txyz_t0 = is_cuda(self.txyz_t0)
        self.u_t0 = exact_u0(self.xyz)
        self.u_t0 = is_cuda(self.u_t0)

        ## 控制方程
        self.txyz = torch.cat((temp_t, self.xyz), dim=1)
        self.txyz_sigma = torch.cat((temp_t + sigma * self.dt, self.xyz), dim=1)
        for i in range(self.t_N - 1):
            temp_t = torch.full_like(torch.zeros(self.xyz_N, 1), self.t[i + 1][0])
            xyz_temp = torch.cat((temp_t, self.xyz), dim=1)
            self.txyz = torch.cat((self.txyz, xyz_temp), dim=0)

            temp_txyz_sigma = torch.cat((temp_t + sigma * self.dt, self.xyz), dim=1)
            self.txyz_sigma = torch.cat((self.txyz_sigma, temp_txyz_sigma), dim=0)

        self.txyz = is_cuda(self.txyz)
        self.txyz_sigma = is_cuda(self.txyz_sigma)

        ## 边界条件
        self.txyz_b = is_cuda(self.txyz_b)
        self.u_b_exact = exact_u(self.txyz_b)

        self.lb = is_cuda(torch.from_numpy(lb).float())
        self.ub = is_cuda(torch.from_numpy(ub).float())

    def train_U(self, x):
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(H)

    def predict_U(self, x):
        return self.train_U(x)

    def Lu_and_F(self):
        u_n = self.train_U(self.txyz)

        x = Variable(self.txyz_sigma, requires_grad=True)
        u_n_sigma = self.train_U(x)

        d = torch.autograd.grad(u_n_sigma, x, grad_outputs=torch.ones_like(u_n_sigma), create_graph=True)
        u_t = d[0][:, [0]]
        u_x = d[0][:, [1]]
        u_y = d[0][:, [2]]
        u_z = d[0][:, [3]]
        u_tt = torch.autograd.grad(u_t, x, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, [0]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [1]]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, [2]]
        u_zz = torch.autograd.grad(u_z, x, grad_outputs=torch.ones_like(u_z), create_graph=True)[0][:, [3]]

        Lu = u_xx + u_yy + u_zz - u_x - u_y - u_z + gamma(3) / gamma(3 - alpha) * x[:, [0]] ** (2 - alpha) - (
                -torch.cos(x[:, [1]]) - torch.cos(x[:, [2]]) - torch.cos(x[:, [3]])) + (
                     -torch.sin(x[:, [1]]) - torch.sin(x[:, [2]]) - torch.sin(x[:, [3]]))

        u_n = u_n.reshape(self.t_N, -1)
        u_t_sigma = u_t.reshape(self.t_N, -1)
        u_tt_sigma = u_tt.reshape(self.t_N, -1)
        Lu_sigma = Lu.reshape(self.t_N, -1)

        return u_n, u_t_sigma, u_tt_sigma, Lu_sigma

    def PDE_loss(self):
        u_n, u_t_sigma, u_tt_sigma, Lu_sigma = self.Lu_and_F()

        loss = is_cuda(torch.tensor(0.))

        for n in range(1, self.t_N):
            if n == 1:
                pre_Ui = u_n[n - 1] + (cof / C(n, 0)) * Lu_sigma[n - 1]
            else:
                pre_Ui = is_cuda(u_n[n - 1] + (cof / C(n, 0)) * Lu_sigma[n - 1])
                for k in range(1, n):
                    pre_Ui += (C(n, k) / C(n, 0)) * (u_n[n - k - 1] - u_n[n - k])
            loss += torch.mean((pre_Ui - u_n[n]) ** 2)

        return loss

    def calculate_loss(self):
        loss_i = torch.mean((self.train_U(self.txyz_t0) - self.u_t0) ** 2)
        self.i_loss_collect.append([self.net.iter, loss_i.item()])
        loss_b = torch.mean((self.train_U(self.txyz_b) - self.u_b_exact) ** 2)
        self.b_loss_collect.append([self.net.iter, loss_b.item()])

        loss_f = self.PDE_loss()
        self.f_loss_collect.append([self.net.iter, loss_f.item()])

        return loss_i, loss_b, loss_f

    # computer backward loss
    def LBGFS_loss(self):
        self.optimizer_LBGFS.zero_grad()
        loss_i, loss_b, loss_f = self.calculate_loss()
        loss = loss_i + loss_b + loss_f
        self.total_loss_collect.append([self.net.iter, loss.item()])
        loss.backward()
        self.net.iter += 1
        print('Iter:', self.net.iter, 'Loss:', loss.item())

        pred = self.train_U(self.txyz_test).cpu().detach().numpy()
        exact = self.txyz_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        error = torch.tensor(error)
        self.error_collect.append([self.net.iter, error.item()])

        if self.net.iter % 10 == 0:
            pred_u = self.train_U(self.txyz_test).cpu().detach().numpy()
            pred_u = torch.from_numpy(pred_u).cpu()
            self.pred_u_collect.append([pred_u.tolist()])

        return loss

    def train(self, LBGFS_epochs=10000):
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

        pred = self.train_U(self.txyz_test).cpu().detach().numpy()
        exact = self.txyz_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        print('LBGFS==Test_L2error:', '{0:.2e}'.format(error))

        elapsed = time.time() - start_time
        print('LBGFS==Training time: %.2f' % elapsed)

        save_loss(self.i_loss_collect, self.b_loss_collect, self.f_loss_collect, self.total_loss_collect)

        return error, elapsed, self.LBGFS_loss().item()

def save_loss(i_loss_collect, b_loss_collect, f_loss_collect, total_loss):
    np.savetxt('loss/i_loss_3D_PDE_L1-2_sigma.txt', i_loss_collect)
    np.savetxt('loss/b_loss_3D_PDE_L1-2_sigma.txt', b_loss_collect)
    np.savetxt('loss/f_loss_3D_PDE_L1-2_sigma.txt', f_loss_collect)
    np.savetxt('loss/total_loss_3D_PDE_L1-2_sigma.txt', total_loss)


# def draw_epoch_loss():
#     b_loss_collect = np.array(model.b_loss_collect)
#     f_loss_collect = np.array(model.f_loss_collect)
#     total_loss = b_loss_collect + f_loss_collect
#     plt.yscale('log')
#     plt.xlabel('$Epoch$')
#     plt.ylabel('$Loss$')
#     # plt.title('$Loss$ based on $L1$')
#     plt.plot(b_loss_collect[:, 0], b_loss_collect[:, 1], 'g-', lw=2)
#     plt.plot(f_loss_collect[:, 0], f_loss_collect[:, 1], 'r-', lw=2)
#     plt.legend(['loss_b', 'loss_f'], fontsize=12)
#     plt.savefig('Figure/3D_PDE/3D_PDE_L1_LOSS.png')
#     plt.show()
#
# def draw_epoch_loss_1():
#     b_loss_collect = np.array(model.b_loss_collect)
#     f_loss_collect = np.array(model.f_loss_collect)
#     total_loss = b_loss_collect + f_loss_collect
#     plt.yscale('log')
#     plt.xlabel('$Epoch$')
#     plt.ylabel('$Loss$')
#     # plt.title('$Loss$ based on $L1$')
#     plt.plot(b_loss_collect[:, 0], total_loss[:, 1], 'b-', lw=2)
#     plt.savefig('Figure/3D_PDE/3D_PDE_L1_LOSS_1.png')
#     plt.show()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    use_gpu = True
    # use_gpu = False
    torch.cuda.is_available()
    set_seed(1234)

    layers = [4, 20, 20, 20, 20, 20, 1]
    # layers = [4, 40, 40, 40, 40, 40, 40, 40, 1]
    # layers = [4, 20, 20, 20, 20, 20, 20, 20, 1]
    # layers = [4, 10, 10, 10, 10, 10, 10, 10, 1]

    net = is_cuda(Net_Attention(layers))

    alpha = 0.5
    sigma = 1 - alpha / 2
    lb = np.array([0.0, 0.0, 0.0, 0.0]) # low boundary t,x,y
    ub = np.array([1.0, 1.0, 1.0, 1.0]) # up boundary t,x,y

    '''train data'''
    t_N = 51
    x_y_z_N = 11

    t, x, y, z, xyz, txyz_b = data_train()

    '''test data'''
    t_test_N = 51
    x_y_z_test_N = 51

    x_test, y_test, z_test, t_test, txyz_test, txyz_test_exact = data_test()

    '''Train'''
    model = Model(
        net=net,
        xyz=xyz,
        txyz_b=txyz_b,
        t=t,
        lb=lb,
        ub=ub,
        txyz_test=txyz_test,
        txyz_test_exact=txyz_test_exact,
    )

    cof = (gamma(2 - alpha) / model.dt ** (-alpha))

    model.train(LBGFS_epochs=5000)

    # test time
    start_time0 = time.time()
    u_test_np = model.predict_U(txyz_test).cpu().detach().numpy()
    u_exact_np = txyz_test_exact.cpu().detach().numpy()
    error_draw = np.linalg.norm(u_test_np - u_exact_np, 2) / np.linalg.norm(u_exact_np, 2)
    print('L2error:', '{0:.2e}'.format(error_draw))
    elapsed0 = time.time() - start_time0
    print('Time: %.4f' % elapsed0)

    '''画图'''
    # draw_epoch_loss()
    # draw_epoch_loss_1()




