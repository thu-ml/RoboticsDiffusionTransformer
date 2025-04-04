import torch
import os

from scripts.agilex_model import RoboticDiffusionTransformerModel
from configs.state_vec import STATE_VEC_IDX_MAPPING


KINOVA7DOF_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_eef_pos_{i}"] for i in "xyz"
] + [
    STATE_VEC_IDX_MAPPING[f"right_eef_angle_{i}"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING["right_gripper_open"]
]

print("KINOVA7DOF_STATE_INDICES", KINOVA7DOF_STATE_INDICES, flush=True)
def create_model(args, **kwargs):
    model = KinovaDiffusionTransformerModel(args, **kwargs)
    pretrained = kwargs.get("pretrained", None)
    if (
        pretrained is not None
        and os.path.isfile(pretrained)
    ):
        model.load_pretrained_weights(pretrained)
    return model


class KinovaDiffusionTransformerModel(RoboticDiffusionTransformerModel):
    def __init__(self, *arg, **kwargs):
         super(KinovaDiffusionTransformerModel, self).__init__(*arg, **kwargs)

    def _format_joint_to_state(self, joints):
        """
        Format the joint proprioception into the unified action vector.

        Args:
            joints (torch.Tensor): The joint proprioception to be formatted.
                qpos ([B, N, 14]).

        Returns:
            state (torch.Tensor): The formatted vector for RDT ([B, N, 128]).
        """
        # Rescale the gripper to the range of [0, 1]
        # TODO
        joints = joints / torch.tensor(
            [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            device=joints.device, dtype=joints.dtype
        )

        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]),
            device=joints.device, dtype=joints.dtype
        )
        # Fill into the unified state vector
        state[:, :, KINOVA7DOF_STATE_INDICES] = joints
        # Assemble the mask indicating each dimension's availability
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device, dtype=joints.dtype
        )
        state_elem_mask[:, KINOVA7DOF_STATE_INDICES] = 1
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action):
        """
        Unformat the unified action vector into the joint action to be executed.

        Args:
            action (torch.Tensor): The unified action vector to be unformatted.
                ([B, N, 128])

        Returns:
            joints (torch.Tensor): The unformatted robot joint action.
                qpos ([B, N, 14]).
        """
        action_indices = KINOVA7DOF_STATE_INDICES
        joints = action[:, :, action_indices]

        # Rescale the gripper back to the action range
        # Note that the action range and proprioception range are different
        # for Mobile ALOHA robot
        joints = joints * torch.tensor(
            [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            device=joints.device, dtype=joints.dtype
        )

        return joints
