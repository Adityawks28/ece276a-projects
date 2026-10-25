import numpy as np
from scipy.spatial import cKDTree

def kabsch(P, Q) :
    assert P.shape == Q.shape, "Matrix dimensions must match!"

    centroid_P = np.mean(P, axis=0) # find centroids
    centroid_Q = np.mean(Q, axis=0) 

    #center the centroids
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    #define H as the covariance matrix
    H = P_centered.T @ Q_centered

    #perform Singular Value Decomposition at H
    U, Sigma, VT = np.linalg.svd(H)

    #optimal rotation matrix 
    R = VT.T @ U.T
    # R = U @ VT sabotage attempt to check self.test

    #check if the R values is negative
    if np.linalg.det(R) < 0 :
        VT[-1, :] *= -1
        R = VT.T @ U.T
    
    t = centroid_Q - R @ centroid_P # the translation
    # p rotated is (P - centroid_P) dot R

    return R, t, calculate_rmsd(apply_rotation(P, R), Q_centered)# return t too maybe?

def calculate_rmsd(P_rotated, Q_centered) : # root mean square deviation
    return np.sqrt(np.mean(np.sum((P_rotated - Q_centered) ** 2, axis = 1))) 

def apply_rotation(P, R) :
    centroid_P = np.mean(P, axis=0)
    P_centered = P - centroid_P
    P_rotated = P_centered @ R.T

    return P_rotated

def icp(source, target, T_initial, max_iterations = 50, max_distance = np.inf) :
    d = source.shape[1]
    T = T_initial.copy()
    tree = cKDTree(target)
    previous_error = np.inf

    for _ in range(max_iterations) :
        moved = source @ T[:d, :d].T + T[:d, d]
        dist, idx = tree.query(moved)
        keep = dist < max_distance
        R, t, rmsd = kabsch(source[keep], target[idx][keep])
        T[:d, :d] = R
        T[:d, d] = t
        error = dist[keep].mean()
        if abs(previous_error - error) < 10**(-6) : #tolerance is 10^-6
            break
        previous_error = error

    return T, error

if __name__ == "__main__" :
    print("Running icp.py self-tests...")
    rng = np.random.default_rng(0)
    P = rng.uniform(-1, 1, size=(100, 3)) # 100 random 3D points

    a = np.deg2rad(40)
    R_true = np.array([[np.cos(a), -np.sin(a), 0],
                       [np.sin(a), np.cos(a), 0],
                       [0, 0, 1]])
    
    t_true = np.array([0.5, -0.2, 0.1])

    Q = P @ R_true.T + t_true

    R, t, err = kabsch(P, Q)

    assert np.allclose(R, R_true), "R is wrong"
    assert np.allclose(t, t_true), "t is wrong"
    print("Kabsch passed, error is : ", err)
    print("The Kabsch in icp.py internal self-tests passed successfully! \n" \
    "--------------------------------------------------------------------")

    print("Running icp tests in icp.py...")
    rng = np.random.default_rng(0)
    P = rng.uniform(-1, 1, size=(100, 3)) # 100 random 3D points
    d = P.shape[1]

    a = np.deg2rad(10)
    R_true = np.array([[np.cos(a), -np.sin(a), 0],
                        [np.sin(a), np.cos(a), 0],
                        [0, 0, 1]])
    
    t_true = np.array([0.5, -0.2, 0.1])

    Q = P @ R_true.T + t_true
    Q = rng.permutation(Q)

    T, dist_mean = icp(P, Q, np.eye(d+1))

    assert np.allclose(R_true, T[:d, :d]), "R in T is wrong!"
    assert np.allclose(t_true, T[:d, d]), "t is T is wrong!"
    print("The icp is working! \n")
