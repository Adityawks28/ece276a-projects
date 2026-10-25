import numpy as np 
import matplotlib.pyplot as plt
from load_data import load_data, curr_dir
from icp import icp
from pr2_utils import bresenham2D, tic, toc
import cv2
import gtsam
from scipy.spatial import cKDTree

LIDAR_X = 0.13323 # 133.23 mm in front of robot center (RobotConfiguration.pdf)
CAM_T = np.array([0.18, 0.005, 0.36])  # depth camera position on the robot (spec)
CAM_RPY = (0.0, 0.36, 0.021)  # depth camera roll, pitch, yaw [rad] (spec)
FLOOR_Z = -0.14  # floor height in the body frame, measured from the Kinect data (spec says 0)
RGBD_DIR = curr_dir / ".." / "data" / "dataRGBD"

def odometry(d):
    t = d["encoder_stamps"]
    imu_t = d["imu_stamps"]
    FR, FL, RR, RL = d["encoder_counts"]
    tau = np.diff(t) # time between consecutive readings
    right_wheel= np.zeros(len(t))
    left_wheel= np.zeros(len(t))
    l = np.zeros(len(t))
    v = np.zeros(len(t))

    for i in range(1, len(t)) :
        right_wheel[i] = (FR[i] + RR[i]) / 2 * 0.0022 #in meter
        left_wheel[i] = (FL[i] + RL[i]) / 2 * 0.0022 #in meter
        l[i] = (right_wheel[i] + left_wheel[i])/2
        v[i] = (l[i]) / tau[i-1]

    yaw_rate = d["imu_angular_velocity"][2]
    matched_yaw = np.zeros(len(t))
    for index, value in enumerate(t): 
        min_index = np.argmin(np.abs(imu_t - value))
        matched_yaw[index] = yaw_rate[min_index]

    pose = np.zeros((len(t), 3))
    # columns: x, y, theta
    teta = np.zeros(len(t))
    x = np.zeros(len(t))
    y = np.zeros(len(t))

    w = matched_yaw # omega in the motion model
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
        start_x, start_y = meters_to_cells(lidar_world, map_info)
        end_cells = meters_to_cells(points_world, map_info)
        for end_x, end_y in end_cells :
            ray_cells = bresenham2D(start_x, start_y, end_x, end_y).astype(int)
            inside = np.all((ray_cells >= 0) & (ray_cells < map_info["size"][:, None]), axis = 0)
            free_cells = ray_cells[:, :-1][:, inside[:-1]]
            grid[free_cells[0], free_cells[1]] -= log_odds
            if inside[-1]:
                grid[ray_cells[0, -1], ray_cells[1, -1]] += log_odds
        np.clip(grid, -20, 20, out=grid)
        if k in snapshot_scans:
            snapshots.append((k, grid.copy()))
    return map_info, grid, snapshots

def camera_to_body():
    # rotation from the camera frame to the body frame, from the spec's roll, pitch, yaw
    roll, pitch, yaw = CAM_RPY
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return R_z @ R_y @ R_x


def texture_map(d, dataset, poses, step=5, number_of_snapshots=4):
    # poses: one (x, y, theta) per LiDAR scan. Colors floor cells from every step-th disparity image.
    map_info = new_map()
    texture = np.zeros((*map_info["size"], 3), dtype=np.uint8)  # 1600 x 1600 x (red, green, blue); black = never seen
    snapshots = []
    R_camera_to_body = camera_to_body()
    rgb_index = nearest(d["rgb_stamps"], d["disp_stamps"])
    pose_index = nearest(d["lidar_stamps"], d["disp_stamps"])
    row, column = np.indices((480, 640))  # row[i, j] = i, column[i, j] = j, for every pixel
    frames = range(0, len(d["disp_stamps"]), step)
    snapshot_frames = set(frames[len(frames) * (s + 1) // number_of_snapshots - 1] for s in range(number_of_snapshots))

    for frame in frames:
        # image files are numbered from 1
        disparity = cv2.imread(str(RGBD_DIR / f"Disparity{dataset}" / f"disparity{dataset}_{frame + 1}.png"), cv2.IMREAD_UNCHANGED).astype(float)
        rgb = cv2.imread(str(RGBD_DIR / f"RGB{dataset}" / f"rgb{dataset}_{rgb_index[frame] + 1}.png"))[:, :, ::-1]  # BGR -> RGB
        dd = -0.00304 * disparity + 3.31  # spec's formulas
        depth = 1.03 / dd
        rgb_row = np.round((526.37 * row + 19276 - 7877.07 * dd) / 585.051).astype(int)
        rgb_column = np.round((526.37 * column + 16662) / 585.051).astype(int)
        # pixel -> optical frame (x right, y down, z forward) -> camera frame (x forward, y left, z up) -> body
        x_optical = (column - 315.84) * depth / 585.05
        y_optical = (row - 242.94) * depth / 585.05
        points_body = np.stack([depth, -x_optical, -y_optical], axis=-1) @ R_camera_to_body.T + CAM_T
        keep = ((depth > 0) & (depth < 5) & (np.abs(points_body[..., 2] - FLOOR_Z) < 0.05)
                & (rgb_row >= 0) & (rgb_row < 480) & (rgb_column >= 0) & (rgb_column < 640))
        T_world = pose_to_T(poses[pose_index[frame]])
        points_world = points_body[keep][:, :2] @ T_world[:2, :2].T + T_world[:2, 2]
        cells = meters_to_cells(points_world, map_info)
        inside = np.all((cells >= 0) & (cells < map_info["size"]), axis=1)
        texture[cells[inside, 0], cells[inside, 1]] = rgb[rgb_row[keep][inside], rgb_column[keep][inside]]
        if frame in snapshot_frames:
            snapshots.append((frame, texture.copy()))
    return map_info, texture, snapshots

def show_map(map_info, image, poses, title, filename):
    # crop to the area the robot saw and draw in meters, without smoothing (walls are 1 cell wide)
    extent = [map_info["min"][0], map_info["max"][0], map_info["min"][1], map_info["max"][1]]
    plt.figure(figsize=(8, 8))
    plt.imshow(np.swapaxes(image, 0, 1), origin="lower", extent=extent, interpolation="none",
               cmap="gray_r", vmin=0, vmax=1)
    plt.plot(poses[:, 0], poses[:, 1], "r", linewidth=0.8)
    plt.xlim(poses[:, 0].min() - 12, poses[:, 0].max() + 12)
    plt.ylim(poses[:, 1].min() - 12, poses[:, 1].max() + 12)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.savefig(filename, dpi=200)
    plt.close()

def match_scans(d, poses_T, i, j, max_distance):
    # ICP of scan j onto scan i, starting from the current relative pose.
    # Returns the relative pose and the share of scan j's points within 10 cm of scan i (inlier ratio).
    T_guess = np.linalg.inv(poses_T[i]) @ poses_T[j]
    source, target = lidar_points(d, j), lidar_points(d, i)
    T_relative, error = icp(source, target, T_guess, max_distance=max_distance)
    moved = source @ T_relative[:2, :2].T + T_relative[:2, 2]
    inlier_ratio = np.mean(cKDTree(target).query(moved)[0] < 0.1)
    return T_relative, inlier_ratio

def pose_graph(d, icp_poses, interval=10, proximity_step=20, proximity_radius=2.0,
               minimum_scan_gap=500, minimum_inlier_ratio=0.6):
    # Factor graph over all ICP poses: odometry factors between consecutive poses,
    # fixed-interval loop closures every `interval` poses, proximity loop closures between
    # poses that are close in space but far apart in time. Returns the optimized poses.
    number_of_poses = len(icp_poses)
    poses_T = [pose_to_T(p) for p in icp_poses]
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3]))
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.005]))  # small on purpose: 5 cm let loop closures shrink the hallways
    loop_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.0),
                                                gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05])))

    graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(*icp_poses[0]), prior_noise))
    for k in range(number_of_poses):
        initial.insert(k, gtsam.Pose2(*icp_poses[k]))
    for k in range(1, number_of_poses):
        T_step = np.linalg.inv(poses_T[k - 1]) @ poses_T[k]
        graph.add(gtsam.BetweenFactorPose2(k - 1, k, gtsam.Pose2(*T_to_pose(T_step)), odometry_noise))

    # (a) fixed-interval loop closures
    fixed_count = 0
    for i in range(0, number_of_poses - interval, interval):
        T_relative, inlier_ratio = match_scans(d, poses_T, i, i + interval, max_distance=0.5)
        if inlier_ratio >= minimum_inlier_ratio:
            graph.add(gtsam.BetweenFactorPose2(i, i + interval, gtsam.Pose2(*T_to_pose(T_relative)), loop_noise))
            fixed_count += 1

    # (b) proximity loop closures
    candidates = range(0, number_of_poses, proximity_step)
    proximity_tried, proximity_count = 0, 0
    for i in candidates:
        for j in candidates:
            close = np.hypot(*(icp_poses[i, :2] - icp_poses[j, :2])) < proximity_radius
            if j - i >= minimum_scan_gap and close:
                proximity_tried += 1
                T_relative, inlier_ratio = match_scans(d, poses_T, i, j, max_distance=1.0)
                if inlier_ratio >= minimum_inlier_ratio:
                    graph.add(gtsam.BetweenFactorPose2(i, j, gtsam.Pose2(*T_to_pose(T_relative)), loop_noise))
                    proximity_count += 1
    print(f"loop closures: fixed-interval {fixed_count}, proximity {proximity_count} of {proximity_tried} tried")

    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    return np.array([[result.atPose2(k).x(), result.atPose2(k).y(), result.atPose2(k).theta()]
                     for k in range(number_of_poses)])

if __name__ == "__main__" :
    dataset = 20
    d = load_data(dataset)
    pose = odometry(d)

    start_time = tic()
    icp_pose = scan_matching(d, pose)
    toc(start_time, "Scan matching")
    np.save(f"icp_pose_{dataset}.npy", icp_pose)

    start_time = tic()
    optimized_pose = pose_graph(d, icp_pose)
    toc(start_time, "Pose graph")
    np.save(f"optimized_pose_{dataset}.npy", optimized_pose)

    plt.figure()
    plt.plot(pose[:, 0], pose[:, 1], label="odometry")
    plt.plot(icp_pose[:, 0], icp_pose[:, 1], label="ICP scan matching")
    plt.plot(optimized_pose[:, 0], optimized_pose[:, 1], label="pose graph (GTSAM)")
    plt.axis("equal")
    plt.legend()
    plt.savefig(f"trajectory_{dataset}.png", dpi=200)
    plt.close()

    for name, poses in (("icp", icp_pose), ("optimized", optimized_pose)):
        start_time = tic()
        map_info, grid, snapshots = occupancy_map(d, poses, step=2)
        toc(start_time, f"Occupancy map ({name})")
        for k, snapshot in snapshots:
            probability = 1 - 1 / (1 + np.exp(snapshot))  # log-odds -> probability of occupied
            show_map(map_info, probability, poses[:k + 1], f"Occupancy ({name}), scans 0-{k}", f"occupancy_{name}_{dataset}_{k}.png")

        start_time = tic()
        map_info, texture, snapshots = texture_map(d, dataset, poses)
        toc(start_time, f"Texture map ({name})")
        for frame, snapshot in snapshots:
            k = nearest(d["lidar_stamps"], d["disp_stamps"][frame:frame + 1])[0]
            show_map(map_info, snapshot, poses[:k + 1], f"Floor texture ({name}), Kinect frames 0-{frame}", f"texture_{name}_{dataset}_{frame}.png")