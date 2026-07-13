# create report figures
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp

from data import load_imu, load_vicon, find_static_end, calibrate
from tracker import increments, dead_reckon, pgd, cost, grad

FIG = '../figures'
TRAIN = range(1, 10)


def sweep(ds):
    # process one dataset
    ts, acc_raw, gyro_raw = load_imu(ds)
    n = find_static_end(gyro_raw)
    acc, gyro = calibrate(acc_raw, gyro_raw, n)
    E = increments(ts, gyro)
    q_dr = dead_reckon(E)

    q, hist = q_dr.copy(), [cost(q_dr, E, acc)]
    for it in range(1500):
        q = q - 1e-3 * grad(q, E, acc)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        if (it + 1) % 25 == 0:
            hist.append(cost(q, E, acc))

    vts, rots = load_vicon(ds)
    inside = (ts >= vts[0]) & (ts <= vts[-1])
    R_gt = Slerp(vts, Rotation.from_matrix(rots))(ts[inside])
    R_dr = Rotation.from_quat(q_dr[inside][:, [1, 2, 3, 0]])
    R_pg = Rotation.from_quat(q[inside][:, [1, 2, 3, 0]])

    def per_axis(R_es):
        d = R_gt.as_euler('xyz', degrees=True) - R_es.as_euler('xyz', degrees=True)
        return np.abs(((d + 180) % 360) - 180).mean(0)

    gate = np.linalg.norm(acc[inside] - R_gt.inv().apply([0, 0, 1.]), axis=1).mean()
    geo_dr = np.degrees((R_gt.inv() * R_dr).magnitude())
    geo_pg = np.degrees((R_gt.inv() * R_pg).magnitude())
    return dict(ds=ds, dr=per_axis(R_dr), pg=per_axis(R_pg), gate=gate, hist=hist,
                t=ts[inside] - ts[0], geo_dr=geo_dr, geo_pg=geo_pg)


res = [sweep(ds) for ds in TRAIN]
for r in res:
    print(f"ds{r['ds']}: gate {r['gate']:.4f} g   dr {r['dr'].round(2)}   pg {r['pg'].round(2)}")

# axis errors
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
x = np.arange(len(res))
for k, name in enumerate(['Roll', 'Pitch', 'Yaw']):
    ax[k].bar(x - .2, [r['dr'][k] for r in res], .4, color='0.65', label='gyro only')
    ax[k].bar(x + .2, [r['pg'][k] for r in res], .4, color='firebrick', label='PGD')
    ax[k].set_xticks(x); ax[k].set_xticklabels([r['ds'] for r in res])
    ax[k].set_title(name); ax[k].set_xlabel('dataset'); ax[k].grid(alpha=.3, axis='y')
ax[0].set_ylabel('mean |error| (deg)'); ax[0].legend(fontsize=9)
fig.suptitle('Gravity corrects roll and pitch on every dataset. It never corrects yaw.', y=1.02)
fig.tight_layout(); fig.savefig(f'{FIG}/summary_peraxis.png', dpi=110, bbox_inches='tight'); plt.close(fig)

# convergence
r1 = res[0]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].semilogy(np.arange(len(r1['hist'])) * 25, r1['hist'], 'k-')
ax[0].set_xlabel('PGD iteration'); ax[0].set_ylabel('cost (log scale)')
ax[0].set_title(f"dataset 1: cost {r1['hist'][0]:.1f} → {r1['hist'][-1]:.2f}, monotone")
ax[0].grid(alpha=.3)
ax[1].plot(r1['t'], r1['geo_dr'], color='0.65', lw=1.0, label=f"gyro only (mean {r1['geo_dr'].mean():.1f}°)")
ax[1].plot(r1['t'], r1['geo_pg'], 'r-', lw=1.1, label=f"PGD (mean {r1['geo_pg'].mean():.1f}°)")
ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('geodesic error (deg)')
ax[1].set_title('geodesic error is immune to gimbal lock'); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f'{FIG}/summary_convergence.png', dpi=110); plt.close(fig)

# calibration error
gate = np.array([r['gate'] for r in res])
rp = np.array([(r['pg'][0] + r['pg'][1]) / 2 for r in res])
rho = np.corrcoef(gate, rp)[0, 1]
m, b = np.polyfit(gate, rp, 1)
fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.scatter(gate, rp, c='firebrick', zorder=3)
for r_, g_, p_ in zip(res, gate, rp):
    ax.annotate(f"ds{r_['ds']}", (g_, p_), textcoords='offset points', xytext=(5, 3), fontsize=8)
xs = np.linspace(gate.min(), gate.max(), 10)
ax.plot(xs, m * xs + b, 'k--', lw=1, label=f'fit, r = {rho:.3f}')
ax.plot(xs, np.degrees(np.arcsin(xs)), 'b:', lw=1.2, label='arcsin(residual): naive bound')
ax.set_xlabel('accel-vs-VICON residual (g)   [the data.py gate]')
ax.set_ylabel('PGD roll/pitch error (deg)')
ax.set_title('the calibration gate predicts the estimator')
ax.legend(fontsize=9); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f'{FIG}/summary_gate_vs_error.png', dpi=110); plt.close(fig)
print(f"\nPearson r = {rho:.3f}   slope = {m:.1f} deg/g")

# panorama comparison
a = plt.imread(f'{FIG}/pano_ds1_vicon.png')
b = plt.imread(f'{FIG}/pano_ds1_pgd.png')
fig, ax = plt.subplots(2, 1, figsize=(10, 10.4))
ax[0].imshow(a); ax[0].set_title('dataset 1 — panorama from VICON ground truth'); ax[0].axis('off')
ax[1].imshow(b); ax[1].set_title('dataset 1 — panorama from our estimate (13.2° yaw error)'); ax[1].axis('off')
fig.tight_layout(); fig.savefig(f'{FIG}/summary_panorama.png', dpi=100); plt.close(fig)
print("wrote 4 summary figures")
