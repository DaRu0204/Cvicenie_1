import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt


def marker(pos, colour):
    shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.005, rgbaColor=colour)
    p.createMultiBody(baseVisualShapeIndex=shape, basePosition=pos)


if __name__ == "__main__":
    # Start PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane = p.loadURDF("plane.urdf")
    robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

    # Joint indices (non-fixed)
    num_joints = p.getNumJoints(robot)
    joint_indices = [i for i in range(num_joints) if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED]

    # Define task space trajectory (start and end in XYZ)
    start_xyz = np.array([0.5, 0.0, 0.5])
    end_xyz = np.array([0.5, 0.3, 0.3])
    steps = 100
    T = 1  # seconds
    dt = T / steps

    # Linear interpolation in Cartesian space
    ee_path = np.linspace(start_xyz, end_xyz, steps)
    joint_trajectory = []
    actual_ee_path = []

    # Move robot to start pose (optional visual alignment)
    start_joint_angles = p.calculateInverseKinematics(
        robot,
        joint_indices[-1],
        start_xyz
    )
    for i, joint_index in enumerate(joint_indices):
        p.resetJointState(robot, joint_index, start_joint_angles[i])

    p.stepSimulation()
    time.sleep(1)  # Optional: pause to let the robot visibly reach start
    p.addUserDebugText("Red line: Actual EE trajectory", [0.4, -0.4, 1.2], textColorRGB=[1, 0, 0], textSize=1.2)
    p.addUserDebugText("Green line: Target EE trajectory",    [0.4, -0.4, 1.1], textColorRGB=[0, 1, 0], textSize=1.2)

    # Simulate the movement
    for pos in ee_path:
        # IK to get joint angles
        ik_solution = p.calculateInverseKinematics(
            robot,
            joint_indices[-1],  # end-effector link
            pos
        )
        # Apply only relevant joints (first 7)
        joint_angles = ik_solution[:len(joint_indices)]
        joint_trajectory.append(joint_angles)
        marker(pos, [0, 1, 0, 1]) 

        # Move joints
        for i, joint_index in enumerate(joint_indices):
            p.resetJointState(robot, joint_index, joint_angles[i])
        p.stepSimulation()

        # Record actual EE position
        ee_state = p.getLinkState(robot, joint_indices[-1])
        actual_ee_path.append(ee_state[0])
        marker(ee_state[0], [1, 0, 0, 1]) 

        time.sleep(dt)

    # Example legend text entries

    p.disconnect()

    # Convert to numpy for plotting
    actual_ee_path = np.array(actual_ee_path)
    joint_trajectory = np.array(joint_trajectory)
    times = np.linspace(0, T, steps)

    # Plot end-effector path in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ee_path[:, 0], ee_path[:, 1], ee_path[:, 2], '--', label='Desired Path')
    ax.plot(actual_ee_path[:, 0], actual_ee_path[:, 1], actual_ee_path[:, 2], label='Executed Path')
    ax.set_title("End-Effector Path in Task Space")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Plot joint angles over time
    plt.figure(figsize=(10, 5))
    for i in range(len(joint_indices)):
        plt.plot(times, joint_trajectory[:, i], label=f'Joint {i+1}')
    plt.title("Joint Angles from IK for Task Space Path")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint Angle [rad]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Compute tracking error
    errors = actual_ee_path - ee_path
    euclidean_error = np.linalg.norm(errors, axis=1)

    # Plot Euclidean error over time
    plt.figure()
    plt.plot(times, euclidean_error)
    plt.title("End-Effector Tracking Error (Euclidean Distance)")
    plt.xlabel("Time [s]")
    plt.ylabel("Position Error [m]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optional: plot error in X, Y, Z separately
    plt.figure()
    plt.plot(times, errors[:, 0], label='X error')
    plt.plot(times, errors[:, 1], label='Y error')
    plt.plot(times, errors[:, 2], label='Z error')
    plt.title("Axis-wise End-Effector Tracking Error")
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

