import numpy as np 
import matplotlib.pyplot as plt
from load_data import load_data, curr_dir
from icp import icp
from pr2_utils import bresenham2D, tic, toc

LIDAR_X = 0.13323 # 133.23 mm in front of robot center (RobotConfiguration.pdf)
CAM_T = np.array([0.18, 0.005, 0.36])  # depth camera position on the robot (spec)
CAM_RPY = (0.0, 0.36, 0.021)  # depth camera roll, pitch, yaw [rad] (spec)
FLOOR_Z = -0.14  # floor height in the body frame, measured from the Kinect data (spec says 0)
RGBD_DIR = curr_dir / ".." / "data" / "dataRGBD"

def odometry(d):
    t = d["encoder_stamps"]
    imu_t = d["imu_stamps"]
    FR, FL, RR, RL = d["encoder_counts"]
    tau = np.diff(t) #time between consecuitve readings
    right_wheel= np.zeros(len(t))
    left_wheel= np.zeros(len(t))
    l = np.zeros(len(t))
    v = np.zeros(len(t))

    for i in range(1, len(t)) :
        right_wheel[i] = (FR[i] + RR[i]) / 2 * 0.0022 #in meter
        left_wheel[i] = (FL[i] + RL[i]) / 2 * 0.0022 #in meter
        l[i] = (right_wheel[i] + left_wheel[i])/2
        v[i] = (l[i]) / tau[i-1]

    yaw_rate = d["imu_angular_velocity"][2] #yaw rate
    matched_yaw = np.zeros(len(t))
    for index, value in enumerate(t): 
        min_index = np.argmin(np.abs(imu_t - value))
        matched_yaw[index] = yaw_rate[min_index]

    pose = np.zeros((len(t), 3))
    #pose of angle, x, and y
    teta = np.zeros(len(t))
    x = np.zeros(len(t))
    y = np.zeros(len(t))

    w = matched_yaw # for convention only 
    for j in range(1, len(t)) : 
        teta[j] = teta[j-1] + w[j] * tau[j-1]
        x[j] = x[j-1] + v[j] * tau[j-1] * np.cos(teta[j])
        y[j] = y[j-1] + v[j] * tau[j-1] * np.sin(teta[j])
        pose[j] = [x[j], y[j], teta[j]]

    return pose

def nearest(times, query):
    # for every query time, the index of the closest entry in times (same idea as matched_yaw)
    return np.array([np.argmin(np.abs(times - q)) for q in query])

def pose_to_T(p):
    # (x, y, theta) -> 3x3 matrix that moves body-frame points into the world frame
    x, y, th = p
    return np.array([[np.cos(th), -np.sin(th), x],
                     [np.sin(th),  np.cos(th), y],
                     [0, 0, 1]])

def T_to_pose(T):
    # 3x3 matrix -> (x, y, theta)
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])

def lidar_points(d, k):
    #scan k -> (N, 2) points in the body frame, ranges outside [range_min, range_max] dropped
    r = d["lidar_ranges"][:, k]
    angles = d["lidar_angle_min"] + np.arange(len(r)) * d["lidar_angle_increment"].item()
    valid = (r > d["lidar_range_min"]) & (r < d["lidar_range_max"])
    return np.column_stack((r[valid] * np.cos(angles[valid]) + LIDAR_X, r[valid] * np.sin(angles[valid])))

def scan_matching(d, odom, min_move=0.5) :
    odom_T = [pose_to_T(p) for p in odom[nearest(d["encoder_stamps"], d["lidar_stamps"])]]
    T = [odom_T[0]]
    keyframe, keyframe_pts = 0, lidar_points(d, 0)
    for k in range(1, len(odom_T)):
        current = lidar_points(d, k)
        #initial guess: previous ICP pose plus odometry step, relative to the keyframe
        T_initial = np.linalg.inv(T[keyframe]) @ T[k-1] @ np.linalg.inv(odom_T[k-1]) @ odom_T[k]
        T_relative, error = icp(current, keyframe_pts, T_initial, max_distance = 0.5)
        T.append(T[keyframe] @ T_relative)
        if np.hypot(T_relative[0, 2], T_relative[1, 2]) > min_move or abs(np.arctan2(T_relative[1, 0], T_relative[0, 0])) > 0.2:
            keyframe, keyframe_pts = k, current
    return np.array([T_to_pose(Tk) for Tk in T])

def new_map():
    MAP = {}
    MAP['resolution'] = 0.05
    MAP['min'] = np.array([-30.0, -30.0])
    MAP['max'] = np.array([50.0, 50.0])
    MAP['size'] = np.ceil((MAP['max'] - MAP['min']) / MAP['resolution']).astype(int)
    return MAP

def meters_to_cells(points_world, map_info):
    # (N, 2) points in meters -> (N, 2) integer cell numbers
    return np.floor((points_world - map_info["min"]) / map_info["resolution"]).astype(int)

def occupancy_map(d, poses, step=1, number_of_snapshots=4) :
    #poses: one(x, y, theta) per LiDAR scans. Log-odds grid built from every step-th scan
    map_info = new_map()
    grid = np.zeros(map_info["size"])
    snapshots = []
    log_odds = np.log(4) # a beam says 4 times more likely (occupied or free)

    scans = range(0, len(poses), step)
    snapshot_scans = set(scans[len(scans) * (s + 1) // number_of_snapshots - 1] for s in range(number_of_snapshots))
    for k in scans:
        T_world = pose_to_T(poses[k])
        points_body = lidar_points(d, k)
        points_world = points_body @ T_world[:2, :2].T + T_world[:2, 2]
        lidar_world = (T_world @ [LIDAR_X, 0, 1])[:2]
        start_x, start_y = meters_to_cells(points_world, map_info)
        end_cells = meters_to_cells(points_world, map_info)
        for end_x, end_y in end_cells :
            ray_cells = bresenham2D(start_x, start_y, end_x, end_y).astype(int)
            inside = np.all((ray_cells >= 0) & (ray_cells < map_info["size"][:, None]), axis = 0)
            free_cells = ray_cells[:, :, -1], inside[:-1]
            grid[free_cells[0], free_cells[1]] -= log_odds
            if inside[-1]:
                grid[ray_cells[0, -1], ray_cells[1, -1]] += log_odds
        np.clip(grid, -20, 20, out=grid)
        if k in snapshot_scans:
            snapshots.append((k, grid.copy()))
    return map_info, grid, snapshots

if __name__ == "__main__" :
    d = load_data(20)
    pose = odometry(d)
    print(pose.shape)
    plt.plot(pose[:, 0], pose[:, 1])
    plt.axis("equal")
    plt.show()