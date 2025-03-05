import pybullet as p
import pybullet_data
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# Create Cube
def create_cube(position, orientation, color):
    collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor = color)
    return p.createMultiBody(1, collision_shape_id, visual_shape_id, position, orientation)

def decompose_homogenous_matrix(T):
    translation = T[:3, 3]
    rotation = T[:3, :3]
    quaternion = R.from_matrix(rotation).as_quat()
    return translation, quaternion

def Rx(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def Rz(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
if __name__ == "__main__":
    # init gui
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    
    p.loadURDF("plane.urdf")
    
    initial_position = [0, 0, 0.1]
    initial_orientation = p.getQuaternionFromEuler([0, 0, 0])
    
    create_cube(initial_position, initial_orientation, [1, 0, 0, 1])
    
    Rxz = np.dot(Rx(np.pi/6), Rz(np.pi/4))
    pos = np.array([[0.5],[0.3],[0.4]])
    zero = np.array([0, 0, 0])
    
    print(Rxz[0, 0])
    print(pos[0])
    
    # Define homogenous transformation matrix
    
    T =  np.array([[np.cos(np.pi/4), -np.sin(np.pi/4), 0, 0.5],
                   [np.sin(np.pi/4), np.cos(np.pi/4), 0, 0.3],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    
    T2 = np.concatenate((Rxz, pos), axis=1)
    T2 = np.concatenate((T2, np.array([[zero[0], zero[1], zero[2], 1]])), axis=0)
    
    new_position, new_orientation = decompose_homogenous_matrix(T2)
    
    create_cube(new_position, new_orientation, [0, 1, 0, 1])    # new cube
    
    # another_position, another_orientation
    
    num_steps = 1000
    axis_lenght = 0.2
    
    for t in range(num_steps):
        p.addUserDebugLine(new_position, new_position + T2[:3, 0] * axis_lenght, [1, 0, 0], lineWidth=1.0)
        p.addUserDebugLine(new_position, new_position + T2[:3, 1] * axis_lenght, [0, 1, 0], lineWidth=1.0)
        p.addUserDebugLine(new_position, new_position + T2[:3, 2] * axis_lenght, [0, 0, 1], lineWidth=1.0)
        p.stepSimulation()
        time.sleep(1)
    
    
    p.disconnect()
    
    