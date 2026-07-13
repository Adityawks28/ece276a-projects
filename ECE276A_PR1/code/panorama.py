import numpy as np

# camera frame
FOV_H = np.deg2rad(60.0)
FOV_V = np.deg2rad(45.0)

def camera_rays(H=240, W=320):
    # create camera rays
    i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    lam = -((j - (W-1)/2) / W) * FOV_H
    phi = -((i - (H-1)/2) / H) * FOV_V
    rays = np.stack([np.cos(phi)*np.cos(lam),
                     np.cos(phi)*np.sin(lam),
                     np.sin(phi)])
    return rays.reshape(3, -1)

def align(imu_ts, cam_ts):
    # align timestamps
    idx = np.searchsorted(imu_ts, cam_ts, side='right') - 1
    return np.clip(idx, 0, len(imu_ts)-1)

def build_panorama(cam, cam_ts, imu_ts, R, pan_h=480, pan_w=960):
    # build panorama
    H, W, _, K = cam.shape
    rays = camera_rays(H, W)
    idx = align(imu_ts, cam_ts)
    pan = np.zeros((pan_h, pan_w, 3), np.uint8)

    for k in range(K):
        p = R[idx[k]] @ rays
        theta = np.arctan2(p[1], p[0])
        psi = np.arcsin(np.clip(p[2], -1.0, 1.0))

        col = ((theta + np.pi) / (2*np.pi) * (pan_w-1)).astype(int)
        row = ((np.pi/2 - psi) / np.pi * (pan_h-1)).astype(int)
        pan[row, col] = cam[:, :, :, k].reshape(-1, 3)
    return pan
