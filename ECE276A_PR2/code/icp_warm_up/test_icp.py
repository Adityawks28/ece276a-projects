import numpy as np
import sys
from utils import read_canonical_model, load_pc, visualize_icp_result
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from icp import icp

def yaw_pose(yaw):
  T = np.eye(4)
  T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
  return T

def estimate_pose(source_pc, target_pc, n_yaw=12):
  # The view only shows one side of the object, so every view point has a partner on the
  # model but not the other way round. Run ICP view -> model, then invert.
  coarse = target_pc[::10]
  best_T, best_err = None, np.inf
  for yaw in np.linspace(0, 2 * np.pi, n_yaw, endpoint=False):
    T0 = yaw_pose(yaw)
    T0[:3, 3] = source_pc.mean(axis=0) - T0[:3, :3] @ target_pc.mean(axis=0)
    T, err = icp(coarse, source_pc, T0)
    if err < best_err:
      best_T, best_err = T, err
  T, err = icp(target_pc, source_pc, best_T)
  return np.linalg.inv(T), err

if __name__ == "__main__":
  obj_name = 'drill' # drill or liq_container
  num_pc = 4 # number of point clouds

  source_pc = read_canonical_model(obj_name)

  for i in range(num_pc):
    target_pc = load_pc(obj_name, i)

    # estimated_pose, you need to estimate the pose with ICP
    pose, err = estimate_pose(source_pc, target_pc)
    print(f'{obj_name} view {i}: mean pair distance {err * 1000:.1f} mm')

    # visualize the estimated result
    visualize_icp_result(source_pc, target_pc, pose)
