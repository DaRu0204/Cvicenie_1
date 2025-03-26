import pybullet as pb
import pybullet_data
import numpy as np
import time
import os

def dh_transformation(a, d, alfa, theta):
    R=np.array([[np.cos(theta), -np.sin(theta), 0, a],
                  [np.sin(theta)*np.cos(alfa), np.cos(theta)*np.cos(alfa), -np.sin(alfa), -d*np.sin(alfa)],
                  [np.sin(theta)*np.sin(alfa), np.cos(theta)*np.sin(alfa), np.cos(alfa), d*np.cos(alfa)],
                  [0, 0, 0, 1]])
    return R

if __name__ == "__main__":
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = pb.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    """
    joint_angles = [np.pi/2, np.pi/6, -np.pi/3, np.pi/4, -np.pi/8, 0, -np.pi/2]
    
    dh_params = [[0, 0.333, 0, joint_angles[0]],
                 [0, 0, -np.pi/2, joint_angles[1]],
                 [0, 0.316, np.pi/2, joint_angles[2]],
                 [0.0825, 0, np.pi/2, joint_angles[3]],
                 [-0.0825, 0.384, -np.pi/2, joint_angles[4]],
                 [0, 0, np.pi/2, joint_angles[5]],
                 [0.088, 0.107, np.pi/2, joint_angles[6]]]
    """
    dh_params = [[0, 0.333, 0, 0],
                 [0, 0, -np.pi/2, -np.pi/4],
                 [0, 0.316, np.pi/2, np.pi/4],
                 [0.0825, 0, np.pi/2, -np.pi/2],
                 [-0.0825, 0.384, -np.pi/2, np.pi/6],
                 [0, 0, np.pi/2, np.pi/2],
                 [0.088, 0.107, np.pi/2, np.pi/3]]
    
    num_dof = 7
    
    T = np.eye(4)
    T_list = np.zeros((num_dof, 4, 4))
    
    for i in range(num_dof):
        a, d, alpha, theta = dh_params[i]
        T_i = dh_transformation(*dh_params[i])
        T = T @ T_i
        T_list[i] = T
        
    fk_position_dh = T[:3,3]
    
    for i in range(num_dof):
        a, d, alfa, theta = dh_params[i]
        pb.setJointMotorControl2(robot_id, i, pb.POSITION_CONTROL, targetPosition=theta)
        # pb.setJointMotorControl2(robot_id, i, pb.POSITION_CONTROL, targetPosition=joint_angles[i])
        
    for _ in range(1000):
        pb.stepSimulation()
        time.sleep(1/240)
        
    ee_state = pb.getLinkState(robot_id, 8, computeForwardKinematics=True)
    fk_position_pybullet = np.array(ee_state[4])
    
    print(f"DH Computed: {fk_position_dh}")
    print(f"Pybullet computation: {fk_position_pybullet}")
    
    while True:
        pb.stepSimulation()
        time.sleep(1/240)