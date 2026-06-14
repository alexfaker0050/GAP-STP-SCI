"""
Research GAP-STP: Dynamic Hybrid Feature Fusion for PnP Video SCI

This module integrates CACTI's forward model with a parallel structure:
Instead of disjoint stages (U-Net then PnP), each iteration computes BOTH:
1. U-Net branch: Learns to suppress structural mask artifacts.
2. FastDVDnet branch: Preserves high-frequency temporal details.
A learned Dynamic Fusion module calculates a spatial weight map to combine them.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the project root to sys.path so 'cacti' can be imported natively
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from cacti.utils.utils import A, At
from cacti.models.gap_net import Unet
from cacti.models.fastdvd import FastDVDnet

class DynamicFusionCell(nn.Module):
    """
    Computes a spatial confidence weight map to dynamically fuse U-Net and PnP outputs.
    """
    def __init__(self, nC=8):
        super().__init__()
        # Input is concatenated outputs (nC * 2) 
        self.attn = nn.Sequential(
            nn.Conv2d(nC * 2, nC, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nC, nC, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x_unet, x_pnp):
        # x_unet: (B, nC, H, W)
        # x_pnp: (B, nC, H, W)
        cat_x = torch.cat([x_unet, x_pnp], dim=1) # (B, nC*2, H, W)
        weight = self.attn(cat_x) # (B, nC, H, W) range [0, 1]
        
        # Soft-mix: 1 = Use entirely U-Net, 0 = Use entirely FastDVDnet PnP
        out = weight * x_unet + (1 - weight) * x_pnp
        return out

class GAP_STP_Net(nn.Module):
    """
    Parallel U-Net & FastDVDnet with dynamic fusion per stage.
    """
    def __init__(self, n_stages=9, nC=8):
        super().__init__()
        self.n_stages = n_stages
        
        # Parallel Task-specific U-Nets
        self.unets = nn.ModuleList([Unet(in_ch=nC, out_ch=nC) for _ in range(n_stages)])
        
        # Universal Prior
        self.fastdvd = FastDVDnet(num_input_frames=5, num_color_channels=1)
        
        # Fusion heads per stage
        self.fusions = nn.ModuleList([DynamicFusionCell(nC=nC) for _ in range(n_stages)])
        
        # Learned sigma map for each stage's PnP branch
        self.sigmas = nn.Parameter(torch.ones(n_stages) * (12/255))
        
    def forward(self, y, Phi, Phi_s):
        x = At(y, Phi)
        
        for k in range(self.n_stages):
            yb = A(x, Phi)
            v = x + At(torch.div(y - yb, Phi_s), Phi)
            
            # --- Branch 1: U-Net (Anti-Artifact) ---
            x_unet = self.unets[k](v)
            
            # --- Branch 2: FastDVDnet PnP ---
            B, nC, H, W = v.shape
            sigma_val = self.sigmas[k]
            out_frames = []
            
            # CRITICAL OOM FIX: Do not track gradients through the heavy frozen FastDVDnet
            with torch.no_grad():
                for b in range(B):
                    frames_b = v[b].unsqueeze(1) # (nC, 1, H, W)
                    noise_map = sigma_val.expand(nC).to(v.device)
                    denoised = self.fastdvd(frames_b, noise_map)
                    out_frames.append(denoised.squeeze(1))
            
            x_pnp = torch.stack(out_frames, dim=0) # (B, nC, H, W)
            
            # --- Convergence Fusion ---
            x = self.fusions[k](x_unet, x_pnp)
            
        return x
