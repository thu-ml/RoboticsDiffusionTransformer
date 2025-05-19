import os
import fnmatch
import yaml
import numpy as np
import cv2

from configs.state_vec import STATE_VEC_IDX_MAPPING


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(self) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        HDF5_DIR = "data/datasets/airbot500_0427"
        self.DATASET_NAME = "airbot500_0427"

        self.file_paths = []
        subdirs = ["catch_redbird", "catch_cup", "pour_water", "mouse_moving", "pack_duck", "pack_duck_pink"]
        for subdir in subdirs:
            subdir_path = os.path.join(HDF5_DIR, subdir)
            if os.path.isdir(subdir_path):
                for root, dirs, files in os.walk(subdir_path):
                    # Check if the current directory is an episode folder (e.g., ep001, ep002, ...)
                    for dir_name in dirs:
                        if fnmatch.fnmatch(dir_name, 'ep*'):  # Match directories like ep001, ep002, etc.
                            episode_path = os.path.join(root, dir_name)  # Full path of the episode folder
                            self.file_paths.append(episode_path)  # Add the episode path to file_paths
            else:
                print(f"Warning: {subdir_path} does not exist or is not a directory.")
                
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
    
        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        # Load the meta.npz file
        meta_file_path = os.path.join(file_path, 'meta.npz')

        # Check if the meta file exists before attempting to load
        if os.path.exists(meta_file_path):
            # Load the .npz file
            data = np.load(meta_file_path)
            num_steps = len(data["timestamp"])
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None
            
            q = data['robot_arm_joint']
            eef = data['robot_gripper_joint']
            eef[eef < 0.05] = 0.0
            eef[eef > 0.95] = 1.0
            qpos = np.concatenate([q, eef], axis=1)

            # [Optional] We skip the first few still steps
            # Get the idx of the first qpos whose delta exceeds the threshold
            EPS = 1e-2
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            # We randomly sample a timestep
            step_id = np.random.randint(first_idx-1, num_steps)

            def get_instruction(file_path):
                INST_DIR = "data/datasets/airbot500_0427/lang"
                ep_name = os.path.basename(file_path)
                ep_num = int(ep_name[2:])
                task_path = os.path.dirname(file_path)
                task_name = os.path.basename(task_path)
                if task_name == "catch_redbird":
                    if 0 <= ep_num <= 24 or 50 <= ep_num <= 74:
                        task_type = "green_box"
                    elif 25 <= ep_num <= 49 or 75 <= ep_num <= 99:
                        task_type = "grey_box"
                elif task_name == "catch_cup":
                    if 0 <= ep_num <= 24 or 50 <= ep_num <= 74:
                        task_type = "green_box"
                    elif 25 <= ep_num <= 49 or 75 <= ep_num <= 99:
                        task_type = "grey_box"
                elif task_name == "pour_water":
                    task_type = "pour_water"
                elif task_name == "mouse_moving":
                    if 0 <= ep_num <= 14:
                        task_type = "forward"
                    elif 15 <= ep_num <= 29:
                        task_type = "backward"
                    elif 30 <= ep_num <= 44:
                        task_type = "right"
                    elif 45 <= ep_num <= 59:
                        task_type = "left"
                elif task_name == "pack_duck":
                    task_type = "blue_duck"
                elif task_name == "pack_duck_pink":
                    task_type = "pink_duck"
                instruction_type = np.random.choice([0, 1, 2])
                instruction = os.path.join(INST_DIR, task_name, "lang_embed", task_type, f"lang_embed_{str(instruction_type)}.pt")
                if not os.path.exists(instruction):
                    raise FileNotFoundError(f"Instruction file not found: {instruction}")

                return instruction

            # Load the instruction
            # dir_path = os.path.dirname(file_path)
            # with open(os.path.join(dir_path, 'expanded_instruction_gpt-4-turbo.json'), 'r') as f_instr:
            #     instruction_dict = json.load(f_instr)
            # We have 1/3 prob to use original instruction,
            # 1/3 to use simplified instruction,
            # and 1/3 to use expanded instruction.
            # instruction_type = np.random.choice([
            #     'instruction', 'simplified_instruction', 'expanded_instruction'])
            # instruction = instruction_dict[instruction_type]
            # if isinstance(instruction, list):
            #     instruction = np.random.choice(instruction)
            # You can also use precomputed language embeddings (recommended)
            instruction = get_instruction(file_path)
            
            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }

            target_q = data['teacher_arm_joint'][step_id:step_id+self.CHUNK_SIZE]
            target_eef = data['teacher_gripper_joint'][step_id:step_id+self.CHUNK_SIZE]
            target_eef[target_eef < 0.05] = 0.0
            target_eef[target_eef > 0.95] = 1.0
            target_qpos = np.concatenate([target_q, target_eef], axis=1)
            
            # Parse the state and action
            state = qpos[step_id:step_id+1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))
            actions = target_qpos
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE-actions.shape[0], 1))
                ], axis=0)
            
            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING["right_gripper_open"]
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            
            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            # If action's format is different from state's,
            # you may implement fill_in_action()
            actions = fill_in_state(actions)
            
            # Parse the images
            def parse_img(folder_name):
                imgs = []
                img_folder = os.path.join(file_path, folder_name)
                for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
                    img_path = os.path.join(img_folder, f"{i:04d}.png")
                    if os.path.isfile(img_path):
                        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        if img is not None:
                            imgs.append(img)
                imgs = np.stack(imgs) # shape: (img_history_size_partial, H, W, 3)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs
            # `cam_high` is the external camera image
            cam_high = parse_img('rgb_ex')
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_left_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
            cam_left_wrist_mask = [False] * self.IMG_HISORY_SIZE
            cam_right_wrist = parse_img('rgb_wrist')
            cam_right_wrist_mask = cam_high_mask.copy()
            
            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            }


    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        meta_file_path = os.path.join(file_path, 'meta.npz')

        # Check if the meta file exists before attempting to load
        if os.path.exists(meta_file_path):
            # Load the .npz file
            data = np.load(meta_file_path)
            num_steps = len(data["timestamp"])
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None

            q = data['robot_arm_joint']
            eef = data['robot_gripper_joint']
            eef[eef < 0.05] = 0.0
            eef[eef > 0.95] = 1.0
            qpos = np.concatenate([q, eef], axis=1)

            target_q = data['teacher_arm_joint']
            target_eef = data['teacher_gripper_joint']
            target_eef[target_eef < 0.05] = 0.0
            target_eef[target_eef > 0.95] = 1.0
            target_qpos = np.concatenate([target_q, target_eef], axis=1)

            # [Optional] We skip the first few still steps
            # Get the idx of the first qpos whose delta exceeds the threshold
            EPS = 1e-2
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")

            # Parse the state and action
            state = qpos[first_idx - 1:]
            action = target_qpos[first_idx - 1:]
            
            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING["right_gripper_open"]
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            state = fill_in_state(state)
            action = fill_in_state(action)
            
            # Return the resulting sample
            return True, {
                "state": state,
                "action": action
            }


if __name__ == "__main__":
    print("Testing HDF5VLADataset...")
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
