import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import time

start_time = time.time()


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
    return 1 + X ** 2


def F(X):
    return (gamma(3) / gamma(3 - alpha)) * X ** (2 - alpha)-exact_u(X)


def data():
    t = np.linspace(lb[0], ub[0], t_N)[:, None]
    u_exact = exact_u(t)

    return t, u_exact


alpha = 0.75
sigma = 1 - alpha / 2

lb = np.array([0.0])  # low boundary
ub = np.array([1.0])  # up boundary

t_N = 21
dt = (ub[0] - lb[0]) / (t_N - 1)
cof = gamma(2 - alpha) / dt ** (-alpha)

t, u_exact = data()
F_n = F(t)

t_sigma = t + sigma * dt
F_n_sigma = F(t_sigma)
u_exact_sigma = exact_u(t_sigma)

pre_U = np.zeros((t_N, 1))
pre_U[0] = u_exact[0]
for n in range(1, t_N):
    if n == 1:
        pre_U[n] = u_exact[n - 1] + (cof / C(n, 0)) * (F_n_sigma[n - 1] + (1 - sigma) * u_exact[n - 1])
        pre_U[n]=pre_U[n]/(1-sigma*cof/ C(n, 0))
    else:
        pre_U[n] = pre_U[n - 1] + (cof / C(n, 0)) * (F_n_sigma[n - 1] + (1 - sigma) * pre_U[n - 1])
        for k in range(1, n):
            pre_U[n] += (C(n, k) / C(n, 0)) * (pre_U[n - k - 1] - pre_U[n - k])
        pre_U[n] = pre_U[n] / (1 - sigma * cof/ C(n, 0))


error_u = np.linalg.norm(u_exact - pre_U, 2) / np.linalg.norm(u_exact, 2)

print('L2error:', '{0:.2e}'.format(error_u))
elapsed1 = time.time() - start_time
print('Time: %.4f' % elapsed1)

