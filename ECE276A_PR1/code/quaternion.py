import numpy as np

# quaternion math

EPS = 1e-12

def qmul(a, b):
    # multiply quaternions
    w1, v1 = a[..., 0:1], a[..., 1:]
    w2, v2 = b[..., 0:1], b[..., 1:]
    scalar = w1*w2 - (v1*v2).sum(-1, keepdims=True)
    vector = w1*v2 + w2*v1 + np.cross(v1, v2)
    return np.concatenate([scalar, vector], axis=-1)

def qconj(q):
    # conjugate quaternion
    return np.concatenate([q[..., 0:1], -q[..., 1:]], axis=-1)

def qexp(v):
    # quaternion exponential
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    safe = np.where(n > EPS, n, 1.0)
    sinc = np.where(n > EPS, np.sin(n)/safe, 1.0)
    return np.concatenate([np.cos(n), sinc*v], axis=-1)

def qlog(q):
    # quaternion logarithm
    w, v = q[..., 0:1], q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    theta = np.arccos(np.clip(w, -1.0, 1.0))
    safe = np.where(n > EPS, n, 1.0)
    return np.where(n > EPS, v * (theta/safe), 0.0)

def qrot(q, p):
    # rotate vector
    P = np.concatenate([np.zeros_like(p[..., 0:1]), p], axis=-1)
    return qmul(qmul(q, P), qconj(q))[..., 1:]
