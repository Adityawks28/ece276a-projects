import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp

from data import read, load_imu, load_vicon, find_static_end, calibrate
from tracker import increments, dead_reckon, pgd, cost
from panorama import build_panorama

FIG = '../figures'
TRAIN = range(1, 10)
TEST = (10, 11)
HAS_CAM = {1, 2, 8, 9, 10, 11}


def euler(q):
    # convert to Euler angles
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)


def estimate(ds, split):
    # estimate orientation
    ts, acc_raw, gyro_raw = load_imu(ds, split)
    n = find_static_end(gyro_raw)
    acc, gyro = calibrate(acc_raw, gyro_raw, n)
    E = increments(ts, gyro)
    q_dr = dead_reckon(E)
    q = pgd(q_dr.copy(), E, acc)
    print(f"  cost {cost(q_dr, E, acc):8.2f} -> {cost(q, E, acc):7.2f}   "
          f"static window {n} samples")
    return ts, acc, q_dr, q


def ground_truth(ds, split, ts):
    # align VICON data
    vts, rots = load_vicon(ds, split)
    inside = (ts >= vts[0]) & (ts <= vts[-1])
    return Slerp(vts, Rotation.from_matrix(rots))(ts[inside]), inside


def plot_rpy(ds, ts, q_dr, q, R_gt=None, inside=None):
    names = ['Roll', 'Pitch', 'Yaw']
    t = ts - ts[0]
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    e_dr, e_pg = euler(q_dr), euler(q)
    for k in range(3):
        if R_gt is not None:
            ax[k].plot(t[inside], R_gt.as_euler('xyz', degrees=True)[:, k],
                       'b-', lw=1.6, label='VICON (ground truth)')
        ax[k].plot(t, e_dr[:, k], color='0.6', lw=1.0, label='gyro integration only')
        ax[k].plot(t, e_pg[:, k], 'r-', lw=1.1, label='projected gradient descent')
        ax[k].set_ylabel(names[k] + ' (deg)')
        ax[k].grid(alpha=.3)
    ax[0].legend(loc='upper left', fontsize=9, ncol=3)
    ax[2].set_xlabel('time (s)')
    ax[0].set_title(f'Dataset {ds} — orientation estimate'
                    + ('' if R_gt is None else ' vs ground truth'))
    fig.tight_layout()
    fig.savefig(f'{FIG}/rpy_ds{ds}.png', dpi=110)
    plt.close(fig)


def errors(R_gt, q, inside):
    # compute errors
    R_es = Rotation.from_quat(q[inside][:, [1, 2, 3, 0]])
    d = np.abs(((R_gt.as_euler('xyz', degrees=True)
                 - R_es.as_euler('xyz', degrees=True) + 180) % 360) - 180)
    return d.mean(0), np.degrees((R_gt.inv() * R_es).magnitude().mean())


def panorama(ds, split, ts, q, tag='pgd', span=None):
    # create panorama
    cam = read(f'../data/{split}/cam/cam{ds}.p')
    imgs, cts = cam['cam'], cam['ts'][0]
    lo, hi = span if span else (ts[0], ts[-1])
    keep = (cts >= lo) & (cts <= hi)
    R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    pan = build_panorama(imgs[:, :, :, keep], cts[keep], ts, R)
    plt.imsave(f'{FIG}/pano_ds{ds}_{tag}.png', pan)
    return (pan.any(-1)).mean() * 100


def run(ds, split):
    print(f"\ndataset {ds} ({split})")
    ts, acc, q_dr, q = estimate(ds, split)

    R_gt, inside = (None, None)
    if split == 'trainset':
        R_gt, inside = ground_truth(ds, split, ts)
        d_dr, g_dr = errors(R_gt, q_dr, inside)
        d_pg, g_pg = errors(R_gt, q, inside)
        print(f"  gyro only : rpy {d_dr.round(2)}  geodesic {g_dr:5.2f} deg")
        print(f"  after pgd : rpy {d_pg.round(2)}  geodesic {g_pg:5.2f} deg")

    plot_rpy(ds, ts, q_dr, q, R_gt, inside)

    if ds in HAS_CAM:
        filled = panorama(ds, split, ts, q)
        print(f"  panorama  : {filled:.1f}% of canvas filled")
        if split == 'trainset':
            q_gt = np.empty_like(q)
            q_gt[:] = [1, 0, 0, 0]
            q_gt[inside] = R_gt.as_quat()[:, [3, 0, 1, 2]]
            covered = (ts[inside][0], ts[inside][-1])
            panorama(ds, split, ts, q_gt, tag='vicon', span=covered)
