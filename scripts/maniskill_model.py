import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner


MANISKILL_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
] + [
    STATE_VEC_IDX_MAPPING[f"right_gripper_open"]
]


def create_model(args, pretrained, **kwargs):
    model = RoboticDiffusionTransformerModel(args, **kwargs)
    if pretrained is not None:
        model.load_pretrained_weights(pretrained)
    return model


DATA_STAT = {'state_min': [-0.31108009815216064, -0.0801204964518547, -0.20882299542427063, -2.364847183227539, -0.18066470324993134, 1.888562798500061, -0.13620014488697052, 0.018293146044015884], 'state_max': [0.29198548197746277, 0.9516910910606384, 0.21581684052944183, -1.3836885690689087, 0.18412861227989197, 2.8172154426574707, 1.6821379661560059, 0.03999999910593033], 'action_min': [-0.3116086423397064, -0.08631071448326111, -0.20903347432613373, -2.368969202041626, -0.18330387771129608, 1.882381558418274, -0.13761861622333527, -1.0], 'action_max': [0.29251915216445923, 0.954268753528595, 0.2164914458990097, -1.3772913217544556, 0.18632978200912476, 2.8199145793914795, 1.686010479927063, 1.0]}

class RoboticDiffusionTransformerModel(object):
    """A wrapper for the RDT model, which handles
            1. Model initialization
            2. Encodings of instructions
            3. Model inference
    """
    def __init__(
        self, args, 
        device='cuda',
        dtype=torch.bfloat16,
        image_size=None,
        control_frequency=25,
        pretrained_text_encoder_name_or_path=None,
        pretrained_vision_encoder_name_or_path=None,
    ):
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.device = device
        self.control_frequency = control_frequency
        self.text_tokenizer, self.text_model = self.get_text_encoder(pretrained_text_encoder_name_or_path)
        self.image_processor, self.vision_model = self.get_vision_encoder(pretrained_vision_encoder_name_or_path)
        self.policy = self.get_policy()

        self.state_min = torch.tensor(DATA_STAT['state_min']).to(device)
        self.state_max = torch.tensor(DATA_STAT['state_max']).to(device)
        self.action_min = torch.tensor(DATA_STAT['action_min']).to(device)
        self.action_max = torch.tensor(DATA_STAT['action_max']).to(device)

        self.reset()

    def get_policy(self):
        """Initialize the model."""
        # Initialize model with arguments
        img_cond_len = (self.args["common"]["img_history_size"] 
                        * self.args["common"]["num_cameras"] 
                        * self.vision_model.num_patches)
        
        _model = RDTRunner(
            action_dim=self.args["common"]["state_dim"],
            pred_horizon=self.args["common"]["action_chunk_size"],
            config=self.args["model"],
            lang_token_dim=self.args["model"]["lang_token_dim"],
            img_token_dim=self.args["model"]["img_token_dim"],
            state_token_dim=self.args["model"]["state_token_dim"],
            max_lang_cond_len=self.args["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                # No initial pos embed in the last grid size
                # since we've already done in ViT
                ("image", (self.args["common"]["img_history_size"], 
                    self.args["common"]["num_cameras"], 
                    -self.vision_model.num_patches)),  
            ],
            lang_pos_embed_config=[
                # Similarly, no initial pos embed for language
                ("lang", -self.args["dataset"]["tokenizer_max_length"]),
            ],
            dtype=self.dtype,
        )

        return _model

    def get_text_encoder(self, pretrained_text_encoder_name_or_path):
        text_embedder = T5Embedder(from_pretrained=pretrained_text_encoder_name_or_path, 
                                   model_max_length=self.args["dataset"]["tokenizer_max_length"], 
                                   device=self.device)
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        return tokenizer, text_encoder

    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder

    def reset(self):
        """Set model to evaluation mode.
        """
        device = self.device
        weight_dtype = self.dtype
        self.policy.eval()
        self.text_model.eval()
        self.vision_model.eval()

        self.policy = self.policy.to(device, dtype=weight_dtype)
        self.text_model = self.text_model.to(device, dtype=weight_dtype)
        self.vision_model = self.vision_model.to(device, dtype=weight_dtype)

    def load_pretrained_weights(self, pretrained=None):
        if pretrained is None:
            return 
        print(f'Loading weights from {pretrained}')
        filename = os.path.basename(pretrained)
        if filename.endswith('.pt'):
            checkpoint =  torch.load(pretrained)
            self.policy.load_state_dict(checkpoint["module"])
        elif filename.endswith('.safetensors'):
            from safetensors.torch import load_model
            load_model(self.policy, pretrained)
        else:
            raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")

    def encode_instruction(self, instruction, device="cuda"):
        """Encode string instruction to latent embeddings.

        Args:
            instruction: a string of instruction
            device: a string of device
        
        Returns:
            pred: a tensor of latent embeddings of shape (text_max_length, 512)
        """
        tokens = self.text_tokenizer(
            instruction, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"].to(device)

        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = self.text_model(tokens).last_hidden_state.detach()

        return pred

    def _format_joint_to_state(self, joints):
        """
        Format the robot joint state into the unified state vector.

        Args:
            joints (torch.Tensor): The joint state to be formatted. 
                qpos ([B, N, 14]).

        Returns:
            state (torch.Tensor): The formatted state for RDT ([B, N, 128]). 
        """
        # Rescale the gripper
        # joints = joints / torch.tensor(
        #     [[[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]]],
        #     device=joints.device, dtype=joints.dtype
        # )
        
        # normalize to -1,1
        joints = (joints - self.state_min) / (self.state_max - self.state_min) * 2 - 1
        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]), 
            device=joints.device, dtype=joints.dtype
        )
        # assemble the unifed state vector
        state[:, :, MANISKILL_INDICES] = joints
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device, dtype=joints.dtype
        )
        state_elem_mask[:, MANISKILL_INDICES] = 1
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action):
        action_indices = MANISKILL_INDICES
        joints = action[:, :, action_indices]
        
        # denormalize to action space 

        joints = (joints + 1) / 2 * (self.action_max - self.action_min) + self.action_min
        
        return joints

    @torch.no_grad()
    def step(self, proprio, images, text_embeds):
        """
        Args:
            proprio: proprioceptive states
            images: RGB images
            text_embeds: instruction embeddings

        Returns:
            action: predicted action
        """
        device = self.device
        dtype = self.dtype
        
        background_color = np.array([
            int(x*255) for x in self.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((
            self.image_processor.size["height"], 
            self.image_processor.size["width"], 3), dtype=np.uint8
        ) * background_color
        
        image_tensor_list = []
        for image in images:
            if image is None:
                # Replace it with the background image
                image = Image.fromarray(background_image)
            
            if self.image_size is not None:
                image = transforms.Resize(self.data_args.image_size)(image)
            
            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
                    
            if self.args["dataset"].get("image_aspect_ratio", "pad") == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)

        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)

        # history of actions
        joints = proprio.to(device).unsqueeze(0)   # (1, 1, 14)
        states, state_elem_mask = self._format_joint_to_state(joints)    # (1, 1, 128), (1, 128)
        states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        states = states[:, -1:, :]  # (1, 1, 128)
        ctrl_freqs = torch.tensor([self.control_frequency]).to(device)
        
        text_embeds = text_embeds.to(device, dtype=dtype)
        
        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(
                text_embeds.shape[:2], dtype=torch.bool,
                device=text_embeds.device),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),  
            ctrl_freqs=ctrl_freqs
        )
        trajectory = self._unformat_action_to_joint(trajectory).to(torch.float32)

        return trajectory
