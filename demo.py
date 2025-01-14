#-*- coding:utf-8 -*-

# Import a create function from the code base
from scripts.agilex_model import create_model
import torch
import time

# Names of cameras used for visual input
CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']
device = 'cuda'
config = {
    'episode_len': 1000,  # Max length of one episode
    'state_dim': 14,      # Dimension of the robot's state
    'chunk_size': 64,     # Number of actions to predict in one step
    'camera_names': CAMERA_NAMES,
}
pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384" 

tensor_type = torch.bfloat16
# tensor_type = torch.float32
# Create the model with the specified configuration
model = create_model(
    args=config,
    dtype=tensor_type, 
    pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
    pretrained='robotics-diffusion-transformer/rdt-1b',
    control_frequency=25,
    device=device
)

torch_model = model.policy.model
torch_model.eval()
print("Model Loaded!")

B = 1
Ta = 65
Ti = 2
# Da = 7 
D = 2048
Di = 1152 # Embedding dimension of the image.
x = torch.randn(B, Ta, D).to(device).type(tensor_type)  
t = torch.randint(0, 1000, (B,)).to(device).type(tensor_type)
f = torch.Tensor([10]).long().to(device).type(tensor_type)

lang_c = torch.randn(B, Ti, D).to(device).type(tensor_type)
img_c = torch.randn(B, Ti, D).to(device).type(tensor_type)

start = time.time()
o = torch_model(x, t, f, lang_c, img_c) # torch.Size([1, 64, 128])
end = time.time()
print(o.shape)
print(f"Time taken: {end - start} seconds")
