import numpy as np
from quaternion import qmul, qconj, qexp

def increments(ts, gyro):
    # create rotation steps
    tau = np.diff(ts)
    return qexp(tau[:, None] * gyro[:-1] / 2.0)

def dead_reckon(E):
    # integrate gyroscope
    q = np.empty((len(E) + 1, 4))
    q[0] = [1.0, 0.0, 0.0, 0.0]
    for t in range(len(E)):
        q[t+1] = qmul(q[t], E[t])
        q[t+1] /= np.linalg.norm(q[t+1])
    return q

def gravity(q):
    # predict gravity
    w, x, y, z = q.T
    return np.stack([2*(x*z - w*y),
                     2*(y*z + w*x),
                     w*w - x*x - y*y + z*z], -1)

def cost(q, E, acc):
    # compute total cost
    s = np.clip((q[1:] * qmul(q[:-1], E)).sum(-1), -1.0, 1.0)
    return 2.0*np.sum(np.arccos(s)**2) + 0.5*np.sum((acc - gravity(q))**2)

def grad(q, E, acc):
    # compute gradient
    g = np.zeros_like(q)

    f = qmul(q[:-1], E)
    s = np.clip((q[1:] * f).sum(-1), -1.0, 1.0)
    theta = np.arccos(s)
    coef = -4.0 / np.sinc(theta/np.pi)
    g[1:] += coef[:, None] * f
    g[:-1] += coef[:, None] * qmul(q[1:], qconj(E))

    w, x, y, z = q.T
    e0, e1, e2 = (acc - gravity(q)).T
    g -= 2.0 * np.stack([-y*e0 + x*e1 + w*e2,
                          z*e0 + w*e1 - x*e2,
                         -w*e0 + z*e1 - y*e2,
                          x*e0 + y*e1 + z*e2], -1)

    g[0] = 0.0
    return g

def pgd(q, E, acc, alpha=1e-3, iters=1500):
    # optimize trajectory
    for it in range(iters):
        q = q - alpha * grad(q, E, acc)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return q
