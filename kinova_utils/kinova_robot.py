import math
import argparse
import numpy as np

from kinova_utils import kinova_api
from kinova_utils.utils_6drot import *

parser = argparse.ArgumentParser()

class KinovaRobot():
    def __init__(self,name):
        self.name = name

    def check_ready(self):
        try:
            self.cnnt, self.base, self.base_cyclic = kinova_api.connection(parser)
            return True
        except:
            return False
    
    def open_gripper(self):
        kinova_api.Open_gripper(self.base)
    
    def close_gripper(self):
        kinova_api.Close_gripper(self.base)

    def send_cartesian_waypoints(self, waypoints):
        kinova_api.Send_Cartesian_waypoints(self.base, self.base_cyclic, waypoints)


    def read_current_pose_6drot(self):
        """
        return dim: array(9),  xyz position + 6d rot
        """
        tmp = kinova_api.Get_cartesian_pose(
            base_cyclic=self.base_cyclic)
        euler = np.array(tmp[3:6])
        euler = np.radians(euler)
        euler = euler[None, :]  # Add batch dimension
        rotmat = convert_euler_to_rotation_matrix(euler)
        ortho6d = compute_ortho6d_from_rotation_matrix(rotmat)[0]
        return np.concatenate( [np.array(tmp[:3]), ortho6d])


    def read_current_pose(self):
        tmp = kinova_api.Get_cartesian_pose(
            base_cyclic=self.base_cyclic)
        return tmp
    
    def read_current_joint_angle(self):
        '''
        return 7DoF angle of actor
        '''
        return kinova_api.Get_joint_positions(
            self.base,self.base_cyclic)

    def read_current_joint_rad(self):
        '''
        return 7DoF angle of actor
        '''
        return np.array(self.read_current_joint_angle()) / 180 * math.pi

    def set_pose(self,pose):
        '''
        pose: [x,y,z,roll,pitch,yaw]
        '''
        kinova_api.Send_relative_cartesian_pose(
            self.base, self.base_cyclic, pose)

    def set_pose_6drot(self, pose):
        ortho6d= pose[3:]
        ortho6d = ortho6d[None, :]
        rotmat_recovered = compute_rotation_matrix_from_ortho6d(ortho6d)
        euler_recovered = convert_rotation_matrix_to_euler(rotmat_recovered)[0] * 180 / math.pi
        pose = np.concatenate([pose[:3] , euler_recovered])
        print("Kinova cartesian_pose API send in", pose, flush=True)
        kinova_api.Send_abosolute_cartesian_pose(
            self.base, self.base_cyclic, pose)

    def set_abosolute_pose(self, pose):
        kinova_api.Send_abosolute_cartesian_pose(
            self.base, self.base_cyclic, pose
        )
        
    def set_joint(self,angle):
        '''
        angle: 7DoF angle of actor
        '''
        return kinova_api.Send_joint_angles(
            self.base, angle)
        
    def close(self):
        kinova_api.disconnect()
    
    def move_to_zero_position(self):
        kinova_api.Move_to_home_position(self.base)

    def move_to_home_position(self):
        kinova_api.Move_to_home_position(self.base)

    def move_to_retract_position(self):
        kinova_api.Move_to_retract_position(self.base)
