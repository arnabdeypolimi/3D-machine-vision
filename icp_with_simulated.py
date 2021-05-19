import numpy as np
import icp
import open3d as o3d

# Simulate point clouds : Create reference and current 3D point clouds
T_GT = icp.SE3_exp([0.1,0.3,-0.2],[0.3,0.1,0.01])
P_ref = np.random.random((100,3)) * 50
# transform P_ref to homogeneous coordinates
P_ref = np.hstack((P_ref, np.ones((P_ref.shape[0], 1))))

# inv(T_GT) transforms from reference to current
P_current = np.dot(np.linalg.inv(T_GT), P_ref.T).T
noise_impact = 0 # try 0.01 0.1 1 and 10
noise = np.random.random((len(P_ref),3)) * noise_impact
P_current_noisy = P_current + np.hstack((noise, np.zeros((noise.shape[0], 1))))

T_est = icp.simpleicp(P_ref, P_current_noisy)
print("************ The ground truth pose ***************************")
print(T_GT)
print("************ The estimated pose ******************************")
print(T_est)
print("************ Checking the error of the estimation ************")
print(np.linalg.inv(T_GT) @ T_est)
