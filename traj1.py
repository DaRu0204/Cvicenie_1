import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt


def marker(pos, colour):
    shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.005, rgbaColor=colour)
    p.createMultiBody(baseVisualShapeIndex=shape, basePosition=pos)


# Cubic interpolation
def cubic_interpolation(q0, qf, T, steps):
    times = np.linspace(0, T, steps)
    q_traj = []
    for t in times:
        tau = t / T
        q_t = [(2*tau**3 - 3*tau**2 + 1)*q0[i] + (-2*tau**3 + 3*tau**2)*qf[i] for i in range(len(q0))]
        q_traj.append(q_t)
    return times, q_traj

if __name__ == "__main__":
    # Start PyBullet with GUI
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane = p.loadURDF("plane.urdf")
    robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

    # Joint setup
    num_joints = p.getNumJoints(robot)
    joint_indices = [i for i in range(num_joints) if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED]

    # Define start and end joint configurations in rad
    start_conf = [0.0] * len(joint_indices)
    end_conf = [0.5, -0.5, 0.3, -1.0, 0.8, 0.5, -0.2]
    
    # Generate trajectory
    T = 5  # seconds
    steps = 100
    times, joint_trajectory = cubic_interpolation(start_conf, end_conf, T, steps)

    # Simulate and record end-effector positions
    ee_positions = []
    target_ee_positions = []
    for q in joint_trajectory:
        for i, joint_index in enumerate(joint_indices):
            p.resetJointState(robot, joint_index, q[i])
        ee_target = p.getLinkState(robot, joint_indices[-1], computeForwardKinematics=True)[0]
        target_ee_positions.append(ee_target)
        p.stepSimulation()
        ee_state = p.getLinkState(robot, joint_indices[-1])
        ee_positions.append(ee_state[0])
        marker(ee_target[0], [0, 1, 0, 1]) 
        marker(ee_state[0], [1, 0, 0, 1]) 
        time.sleep(T / steps)

    # Convert for plotting
    ee_positions = np.array(ee_positions)
    target_ee_positions = np.array(target_ee_positions)
    # Plot joint trajectories
    plt.figure(figsize=(10, 5))
    for i in range(len(joint_indices)):
        plt.plot(times, [q[i] for q in joint_trajectory], label=f'Joint {i+1}')
    plt.title("Joint Space Trajectories (Cubic Interpolation)")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint Angle [rad]")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot end-effector path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], label='End-Effector Path')
    ax.plot(target_ee_positions[:, 0], target_ee_positions[:, 1], target_ee_positions[:, 2], label='End-Effector target Path')
    ax.set_title("End-Effector Path in Task Space")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    plt.show()
