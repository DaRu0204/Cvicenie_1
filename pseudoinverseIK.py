import pybullet as p
import pybullet_data
import numpy as np
import time

def forward_kinematics(theta):
    theta1, theta2 = theta
    x = link_lengths[0]*np.cos(theta1) + link_lengths[1]*np.cos(theta1+theta2)
    y = link_lengths[0]*np.sin(theta1) + link_lengths[1]*np.sin(theta1+theta2)
    
    return np.array((x, y))

def jacobian (theta):
    theta1, theta2 = theta
    l1, l2 = link_lengths
    
    dx_dtheta1 = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
    dx_dtheta2 = -l2 * np.sin(theta1 + theta2)
    
    dy_dtheta1 = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    dy_dtheta2 = l2 * np.cos(theta1 + theta2)
    
    return np.array(([dx_dtheta1, dx_dtheta2], [dy_dtheta1, dy_dtheta2]))

def inverse_kinematics(target, theta_init, step=100):
    theta = np.array(theta_init, dtype=float)
    for _ in range(step):
        pos = forward_kinematics(theta)
        err = target - pos
        if np.linalg.norm(err) < 1e-3:
            break
        
        J = jacobian(theta)
        dtheta = np.linalg.pinv(J) @ err
        theta += dtheta
    return theta

if __name__=="__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0,0,-9.81)
    
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("urdf/2dof_planar_robot.urdf", [0, 0, 0], useFixedBase=True)
    
    joint_indices = [0, 1]
    link_lengths = [1, 1]
    
    target_pos = np.array((1.2, 1.2))
    
    target_id = p.loadURDF("sphere_small.urdf", [target_pos[0], target_pos[1], 0], globalScaling=0.1)
    theta_guess = [0.1, 0.1]
    
    theta_solution = inverse_kinematics(target_pos, theta_guess)
    theta_solution = (theta_solution + np.pi) % (2*np.pi) - np.pi
    
    