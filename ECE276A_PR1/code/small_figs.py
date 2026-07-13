# create small data figures
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from scipy.spatial.transform import Rotation

from data import read, load_imu, find_static_end, calibrate
from panorama import FOV_H, FOV_V, align
from tracker import increments, dead_reckon, pgd


FIG = '../figures'
CREAM = '#f7f2e5'
RED = '#ff2020'
GRAY = '#555555'
os.makedirs(FIG, exist_ok=True)


def box(ax, x, y, w, h, text, edge='#888888', lw=1.0, size=10):
    # draw one box
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02',
        facecolor=CREAM,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size)


def arrow(ax, x1, y1, x2, y2):
    # draw one arrow
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=GRAY))


ts, acc_raw, gyro_raw = load_imu(1)
n_static = find_static_end(gyro_raw)
acc, gyro = calibrate(acc_raw, gyro_raw, n_static)
windows = np.lib.stride_tricks.sliding_window_view(gyro_raw, 50, axis=0)
std = windows.std(-1)


# sliding windows
rows = [n_static - 3, n_static - 2, n_static - 1, n_static]
fig, ax = plt.subplots(figsize=(6.2, 4.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.4)
ax.axis('off')
ax.set_title('Real sliding windows from dataset 1', fontsize=14, pad=8)

for r, k in enumerate(rows):
    y = 4.25 - r * 1.05
    ax.text(0.1, y + 0.31, f'{k}:{k + 49}', ha='left', va='center', fontsize=9)
    values = std[k]
    labels = [f'$\\sigma_x$\n{values[0]:.2f}',
              f'$\\sigma_y$\n{values[1]:.2f}',
              f'$\\sigma_z$\n{values[2]:.2f}']
    for d, label in enumerate(labels):
        box(ax, 1.8 + d * 1.25, y, 1.15, 0.62, label, size=9)
    dmax = int(np.argmax(values))
    ax.add_patch(Rectangle((1.75 + dmax * 1.25, y - 0.05), 1.25, 0.72,
                           fill=False, edgecolor=RED, linewidth=3))
    value = values[dmax]
    state = 'moving' if value > 1.5 else 'static'
    sign = '>' if value > 1.5 else '<'
    ax.text(5.9, y + 0.31, f'max = {value:.2f} {sign} 1.50\n{state}',
            ha='left', va='center', fontsize=10,
            color=RED if state == 'moving' else 'black')

ax.text(0.1, 0.15, 'The window slides by one sample. Motion starts when max std crosses 1.50.',
        fontsize=9)
fig.tight_layout()
fig.savefig(f'{FIG}/small_static_window.png', dpi=180, bbox_inches='tight')
plt.close(fig)


# calibration
raw_acc_mean = acc_raw[:n_static].mean(0)
raw_gyro_mean = gyro_raw[:n_static].mean(0)
cal_acc_mean = acc[:n_static].mean(0)
cal_gyro_mean = gyro[:n_static].mean(0)
cal_acc_mean[np.abs(cal_acc_mean) < 5e-4] = 0.0
cal_gyro_mean[np.abs(cal_gyro_mean) < 5e-4] = 0.0

fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.2)
ax.axis('off')
ax.set_title('Calibration with the stationary part of dataset 1', fontsize=14, pad=8)

box(ax, 0.2, 2.45, 3.0, 0.9,
    'raw accel mean (counts)\n' + np.array2string(raw_acc_mean, precision=2), size=9)
box(ax, 0.2, 0.85, 3.0, 0.9,
    'raw gyro mean (counts)\n' + np.array2string(raw_gyro_mean, precision=2), size=9)
box(ax, 4.25, 1.65, 3.0, 0.9, 'subtract bias\nand apply scale', edge=RED, lw=2.5, size=10)
box(ax, 8.2, 2.45, 3.2, 0.9,
    'calibrated accel mean (g)\n' + np.array2string(cal_acc_mean, precision=3, suppress_small=True), size=9)
box(ax, 8.2, 0.85, 3.2, 0.9,
    'calibrated gyro mean (rad/s)\n' + np.array2string(cal_gyro_mean, precision=3, suppress_small=True), size=9)
arrow(ax, 3.25, 2.9, 4.2, 2.2)
arrow(ax, 3.25, 1.3, 4.2, 2.0)
arrow(ax, 7.3, 2.2, 8.15, 2.9)
arrow(ax, 7.3, 2.0, 8.15, 1.3)
fig.tight_layout()
fig.savefig(f'{FIG}/small_calibration.png', dpi=180, bbox_inches='tight')
plt.close(fig)


# one motion step
E = increments(ts, gyro)
q_dr = dead_reckon(E)
k = n_static
dt = ts[k + 1] - ts[k]
half_step = dt * gyro[k] / 2.0

fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.0)
ax.axis('off')
ax.set_title('One real dead reckoning step from dataset 1', fontsize=14, pad=8)

box(ax, 0.3, 2.95, 3.7, 1.2,
    f'$\\tau_t$ = {dt * 1000:.3f} ms\n'
    f'$\\omega_t$ = [{gyro[k, 0]:.5f}, {gyro[k, 1]:.5f}, {gyro[k, 2]:.5f}]',
    size=9)
box(ax, 5.0, 2.75, 4.7, 1.6,
    '$\\tau_t\\omega_t/2$ =\n'
    f'[{half_step[0]:.4e}, {half_step[1]:.4e}, {half_step[2]:.4e}]\n'
    '$E_t$ = '
    f'[{E[k, 0]:.6f}, {E[k, 1]:.6f}, {E[k, 2]:.6f}, {E[k, 3]:.6f}]',
    edge=RED, lw=2.5, size=8.5)
arrow(ax, 4.05, 3.55, 4.95, 3.55)
ax.text(4.5, 3.8, 'exp', ha='center', va='bottom', fontsize=9)

box(ax, 0.3, 0.55, 4.1, 1.2,
    '$q_t$ =\n'
    f'[{q_dr[k, 0]:.6f}, {q_dr[k, 1]:.6f}, '
    f'{q_dr[k, 2]:.6f}, {q_dr[k, 3]:.6f}]', size=8.5)
box(ax, 5.5, 0.55, 4.2, 1.2,
    '$q_{t+1}=q_t\\circ E_t$\n'
    f'[{q_dr[k + 1, 0]:.6f}, {q_dr[k + 1, 1]:.6f}, '
    f'{q_dr[k + 1, 2]:.6f}, {q_dr[k + 1, 3]:.6f}]',
    edge=RED, lw=2.5, size=8.5)
arrow(ax, 4.45, 1.15, 5.45, 1.15)
fig.tight_layout()
fig.savefig(f'{FIG}/small_motion_step.png', dpi=180, bbox_inches='tight')
plt.close(fig)


# one panorama pixel
q_pgd = pgd(q_dr.copy(), E, acc)
rotations = Rotation.from_quat(q_pgd[:, [1, 2, 3, 0]]).as_matrix()
cam_data = read('../data/trainset/cam/cam1.p')
cam_ts = cam_data['ts'][0]
c = len(cam_ts) // 2
imu_k = align(ts, np.array([cam_ts[c]]))[0]
i, j = 120, 160
H, W = 240, 320
lam = -((j - (W - 1) / 2) / W) * FOV_H
phi = -((i - (H - 1) / 2) / H) * FOV_V
ray = np.array([np.cos(phi) * np.cos(lam),
                np.cos(phi) * np.sin(lam),
                np.sin(phi)])
world = rotations[imu_k] @ ray
theta = np.arctan2(world[1], world[0])
psi = np.arcsin(np.clip(world[2], -1.0, 1.0))
col = int((theta + np.pi) / (2 * np.pi) * 959)
row = int((np.pi / 2 - psi) / np.pi * 479)

fig, ax = plt.subplots(figsize=(8.0, 3.8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4.3)
ax.axis('off')
ax.set_title('One real camera pixel mapped into the panorama', fontsize=14, pad=8)

box(ax, 0.2, 1.75, 2.4, 1.1,
    f'image pixel\n$(i,j)=({i},{j})$', size=10)
box(ax, 3.25, 1.75, 3.2, 1.1,
    'body ray $r_{ij}$\n' + np.array2string(ray, precision=5), size=9)
box(ax, 7.1, 1.75, 3.2, 1.1,
    f'world ray $R_{{{imu_k}}}r_{{ij}}$\n' + np.array2string(world, precision=5), size=9)
box(ax, 10.95, 1.75, 2.7, 1.1,
    f'panorama pixel\n$(n,m)=({row},{col})$', edge=RED, lw=2.5, size=10)
arrow(ax, 2.65, 2.3, 3.2, 2.3)
arrow(ax, 6.5, 2.3, 7.05, 2.3)
arrow(ax, 10.35, 2.3, 10.9, 2.3)
ax.text(10.1, 0.75,
        f'$\\theta={np.degrees(theta):.2f}^\\circ$, '
        f'$\\psi={np.degrees(psi):.2f}^\\circ$',
        ha='center', va='center', fontsize=10)
ax.text(4.0, 0.75,
        f'camera frame {c} $\\rightarrow$ IMU sample {imu_k}',
        ha='center', va='center', fontsize=9)
fig.tight_layout()
fig.savefig(f'{FIG}/small_panorama_pixel.png', dpi=180, bbox_inches='tight')
plt.close(fig)

print('wrote four small data figures')
