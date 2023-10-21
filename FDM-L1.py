import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import time

start_time = time.time()


def C(n):
    return (n + 1) ** (1 - alpha) - n ** (1 - alpha)


# def exact_u(X):
#     return X ** (5 + alpha)
#
# def F(X):
#     return (gamma(6 + alpha) / 120) * X ** 5 + X ** (5 + alpha)

def exact_u(X):
    return 1 + X ** 2


def F(X):
    return (gamma(3) / gamma(3 - alpha)) * X ** (2 - alpha) - X ** 2 - 1


alpha = 0.75

lb = np.array([0.0])  # low boundary
ub = np.array([1.0])  # up boundary

t_N = 201

t = np.linspace(lb[0], ub[0], t_N)[:, None]
u_exact = exact_u(t)
F = F(t)
dt = ((ub[0] - lb[0]) / (t_N - 1))
cof = (gamma(2 - alpha) / dt ** (-alpha)) / C(0)

pre_U = np.zeros((t_N, 1))
pre_U[0] = u_exact[0]
for n in range(1, t_N):
    if n == 1:
        pre_U[n] = cof * F[n] + (C(n - 1) / C(0)) * u_exact[0]
    else:
        pre_U[n] = cof * F[n] + (C(n - 1) / C(0)) * u_exact[0]
        for k in range(1, n):
            pre_U[n] += ((C(n - k - 1) - C(n - k)) / C(0)) * pre_U[k]
    pre_U[n] = pre_U[n] / (1 - cof)

error_u = np.linalg.norm(u_exact - pre_U, 2) / np.linalg.norm(u_exact, 2)
print('L2error:', '{0:.2e}'.format(error_u))
elapsed1 = time.time() - start_time
print('Time: %.4f' % elapsed1)
