import h5py
import numpy as np
from kinova_utils.utils_6drot import convert_euler_to_rotation_matrix, compute_ortho6d_from_rotation_matrix

def eef_process(path):
    '''
    Return 9d eef pose(xyz position + 6d rot) + gripper
    '''
    with h5py.File(path, 'r+') as f:
        data = f['observations']['tool_pose']
        eef_list = []
        for i in range(len(data)):
            euler = data[i][-3:].reshape(-1, 3)
            euler_radians = np.radians(euler)
            rotmat = convert_euler_to_rotation_matrix(euler_radians)
            ortho6d = compute_ortho6d_from_rotation_matrix(rotmat).reshape(-1)
            eef_9d = np.concatenate((data[i][:3].reshape(-1), ortho6d))

            gripper = f['observations']['gripper'][i]
            eef_all = np.append(eef_9d, gripper)
            eef_list.append(eef_all)
        eef_array = np.array(eef_list, dtype=np.float32)
        if 'observations/eef_9d' not in f:
            f.create_dataset('observations/eef_9d', data=eef_array)

def action_process(path):
    '''
    Return 9d action(xyz position + 6d rot) + gripper
    '''
    with h5py.File(path, 'r+') as f:
        action_new = np.empty((len(f['observations']['eef_9d']), 10))
        for j in range(len(f['observations']['eef_9d'])-1):
            action_new[j, :] = f['observations']['eef_9d'][j+1]
        action_new[-1, :] = f['observations']['eef_9d'][-1]
        del f['action']
        if 'action' not in f:
            f.create_dataset('action', data=action_new)

def main():
    for i in range (20):
        path = (f'/path/to/episode_{i}.hdf5')
        eef_process(path)
        action_process(path)
        print(f'episode_{i} done')

if __name__ == '__main__':
    main()