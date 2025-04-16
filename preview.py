import pickle
import numpy as np

# Replace 'your_file.pkl' with the path to your .pkl file
file_path = './data/datasets/redbird50_0325/ep01.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)
    q = data["get_current_pose"][1]
    print(q)

    # EPS = 1e-2
    # qpos_delta = np.abs(q - q[0:1])
    # # print(qpos_delta)
    # print(np.max(qpos_delta, axis=1))
    # indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
    # if len(indices) > 0:
    #     first_idx = indices[0]
    # else:
    #     raise ValueError("Found no qpos that exceeds the threshold.")
    # print(first_idx)

# Print the content of the .pkl file
# if isinstance(data, dict):
#     print(data.keys())