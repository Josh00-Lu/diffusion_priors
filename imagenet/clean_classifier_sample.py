import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

from unet import UNetModel, EncoderUNetModel
from diffusion import GaussianDiffusion

import tqdm
import matplotlib.pyplot as plt

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--save_base', type=str)
    parser.add_argument('--class_cond', type=str2bool)

    args = parser.parse_args()
    return args

args = create_args()

# Load DDPM model
device = torch.device('cuda:0')

if args.image_size == 256:
    diff_net = UNetModel(image_size=256, in_channels=3, out_channels=6,
                        num_classes=(1000 if args.class_cond else None),
                        model_channels=256, num_res_blocks=2, channel_mult=(1, 1, 2, 2, 4, 4),
                        attention_resolutions=[32,16,8], num_head_channels=64, dropout=0.0, resblock_updown=True, use_scale_shift_norm=True).to(device)
elif args.image_size == 128:
    diff_net = UNetModel(image_size=128, in_channels=3, out_channels=6, 
                        num_classes=(1000 if args.class_cond else None),
                        model_channels=256, num_res_blocks=2, channel_mult=(1, 1, 2, 3, 4),
                        attention_resolutions=[16,8,4], num_heads=4, dropout=0.0, resblock_updown=True, use_scale_shift_norm=True).to(device)
elif args.image_size == 64:
    diff_net = UNetModel(image_size=64, in_channels=3, out_channels=6, use_new_attention_order=True,
                        num_classes=(1000 if args.class_cond else None),
                        model_channels=192, num_res_blocks=3, channel_mult=(1, 2, 3, 4),
                        attention_resolutions=[8,4,2], num_head_channels=64, dropout=0.1, resblock_updown=True, use_scale_shift_norm=True).to(device)

diff_net.load_state_dict(torch.load(args.model_path))

def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=1000,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )

classifier = create_classifier(
    image_size=args.image_size, 
    classifier_use_fp16=False, classifier_width=128, classifier_depth=(4 if args.image_size == 64 else 2),
    classifier_attention_resolutions="32,16,8",  # 16
    classifier_use_scale_shift_norm=True,  # False
    classifier_resblock_updown=True,  # False
    classifier_pool="attention"
).to(device)

classifier.load_state_dict(torch.load(args.classifier_path))



######################## START SAMPLING ##########################

class InferenceModel(nn.Module):
    def __init__(self):
        super(InferenceModel, self).__init__()
        # Inferred image
        self.img = nn.Parameter(torch.randn(args.batch_size, 3, args.image_size, args.image_size))
        self.img.requires_grad = True

    def encode(self):
        return self.img

all_images = []
while len(all_images) * args.batch_size < args.num_samples:
    X0 = InferenceModel().to(device)
    # Inference procedure steps
    steps = 250  
    opt = torch.optim.Adamax(X0.parameters(), lr=1)
    diffusion = GaussianDiffusion(T=1000, schedule=('cosine' if args.image_size == 64 else 'linear'))
    diff_net.eval()
    classes = torch.randint(low=0, high=1000, size=(args.batch_size,), device=device)

    for i, _ in enumerate(range(steps)):
        # Select t      
        t = ((steps-i)/1.5 + (steps-i)/3*math.cos(i/10))/steps*800 + 200 # Linearly decreasing + cosine
        t = np.array([t + np.random.randint(-50, 51) for _ in range(1)]).astype(int) # Add noise to t
        t = np.clip(t, 1, diffusion.T)
        
        # Denoise
        sample_img = X0.encode()
        t = torch.tensor([t[0]] * args.batch_size, device=device)
        xt, epsilon = diffusion.sample(sample_img, t) 
        if args.class_cond:
            pred = diff_net(xt.float(), t, y=classes) 
        else:
            pred = diff_net(xt.float(), t) 
        epsilon_pred = pred[:,:3,:,:] # Use predicted noise only
        
        loss = F.mse_loss(epsilon_pred, epsilon)

        opt.zero_grad()        
        loss.backward()

        with torch.no_grad():
            grad_norm = torch.linalg.norm(X0.img.grad)
            if i > 0:
                alpha = 0.5
                norm_track = alpha*norm_track + (1-alpha)*grad_norm
            else:
                norm_track = grad_norm 
                    
        opt.step()

        zero_t = torch.zeros((args.batch_size,), device=device)
        logits = classifier(X0.encode(), zero_t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), classes.view(-1)] 
        Loss_label = selected.sum()

        opt.zero_grad()
        Loss_label.backward()
        torch.nn.utils.clip_grad_norm_(X0.parameters(), 0.1*norm_track)
        opt.step()

    sample = ((X0.encode() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous().cpu().numpy()
    all_images.append(sample)
    print(f"created {len(all_images) * args.batch_size} samples")

arr = np.concatenate(all_images, axis=0)
arr = arr[: args.num_samples]

shape_str = "x".join([str(x) for x in arr.shape])
if not os.path.exists(args.save_base):
    os.mkdir(args.save_base)
out_path = os.path.join(args.save_base, f"samples_{shape_str}.npz")
print(f"saving to {out_path}")
np.savez(out_path, arr)