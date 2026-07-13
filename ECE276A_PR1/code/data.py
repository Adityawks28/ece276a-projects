import pickle
import numpy as np

VREF = 3300.0
ADC_MAX = 1023.0

ACC_SENS = 330.0
GYRO_SENS = 3.33

def read(fname) :
    with open(fname, 'rb') as f :
        d = pickle.load(f, encoding='latin1')
    return d

def load_imu(dataset=1, split='trainset'):
    # load IMU data
    imu = read(f"../data/{split}/imu/imuRaw{dataset}.p")

    ts = imu[0]
    acc_raw = imu[1:4].T
    gyro_raw=imu[4:7].T
    return ts, acc_raw, gyro_raw

def load_vicon(dataset=1, split='trainset'):
    # load VICON data
    vicon = read(f"../data/{split}/vicon/viconRot{dataset}.p")
    vts, rots = vicon['ts'][0], vicon['rots'].transpose(2, 0, 1)
    keep = np.isfinite(rots).all(axis=(1, 2))
    return vts[keep], rots[keep]

def find_static_end(gyro_raw, window=50, threshold=1.5):
    # find motion start
    window = np.lib.stride_tricks.sliding_window_view(gyro_raw, window, axis=0)
    spread = window.std(-1).max(1)
    moving = spread > threshold
    return int(np.argmax(moving)) if moving.any() else len(spread)


def calibrate(acc_raw, gyro_raw, n_static):
    # calibrate IMU data
    acc_scale  = VREF / ADC_MAX / ACC_SENS
    gyro_scale = VREF / ADC_MAX * (np.pi/180) / GYRO_SENS

    bias_acc = acc_raw[:n_static].mean(0)
    bias_acc[2] = bias_acc[2] - 1.0/acc_scale
    bias_gyro = gyro_raw[:n_static].mean(0)

    acc = (acc_raw - bias_acc) * acc_scale
    gyro = (gyro_raw - bias_gyro) * gyro_scale
    return acc, gyro
