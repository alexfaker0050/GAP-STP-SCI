"""
Training script for Research GAP-STP: Dynamic Hybrid Feature Fusion.
"""
import os
import sys
import torch
torch.backends.cudnn.benchmark = False # Disable dynamic algorithm search to prevent memory fragmentation OOM
torch.backends.cudnn.deterministic = True 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import einops

# Append project root to python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from cacti.datasets.builder import build_dataset
from cacti.utils.config import Config
from cacti.utils.mask import generate_masks
from model import GAP_STP_Net

# Setup Config matching CACTI
mask_path = os.path.join(SCRIPT_DIR, "datasets", "mask.mat")
mask_np_full, mask_s_np_full = generate_masks(mask_path)
NC = 4
mask_np = mask_np_full[:NC, :, :]
mask_s_np = np.maximum(np.sum(mask_np, axis=0), 1e-6).astype(np.float32)
data_root = os.path.join(SCRIPT_DIR, "datasets", "train")

train_cfg = Config({
    "type": "DavisData",
    "data_root": data_root,
    "mask_path": mask_path,
    "mask_shape": None,
    "pipeline": [ 
        dict(type='RandomResize'),
        dict(type='RandomCrop', crop_h=256, crop_w=256, random_size=True),
        dict(type='Flip', direction='horizontal', flip_ratio=0.5),
        dict(type='Flip', direction='diagonal', flip_ratio=0.5),
        dict(type='Resize', resize_h=256, resize_w=256),
    ],
    "gene_meas": dict(type='GenerationGrayMeas')
})

def main():
    print("=" * 60)
    print("Initializing GAP-STP (Dynamic Hybrid Fusion) Environment...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset Initialization
    print(f"Building Dataset from {data_root}...")
    dataset = build_dataset(train_cfg, {"mask": mask_np})
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    
    # 2. Model Initialization (12 fully parallel stages to match CACTI's 12 stages)
    n_stages = 12
    model = GAP_STP_Net(n_stages=n_stages, nC=NC).to(device)
    
    # --- CRITICAL: PRELOAD GAP-NET UNET WEIGHTS ---
    gapnet_path = os.path.join(BASE_DIR, "CACTI", "checkpoints", "gapnet", "gapnet.pth")
    if os.path.exists(gapnet_path):
        state = torch.load(gapnet_path, map_location=device)
        state = state.get("model_state_dict", state)
        
        # Manually extract UNet weights (unet1 -> unet12) to self.unets[0..11]
        print(f"Transferring GAP-Net U-Net 1~{n_stages} weights (adapting nC=8 to nC={NC})...")
        for i in range(n_stages):
            prefix = f"unet{i+1}."
            unet_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if unet_state:
                # Adapt nC if necessary (8 -> 4)
                if unet_state['dconv_down1.d_conv.0.weight'].shape[1] != NC:
                    unet_state['dconv_down1.d_conv.0.weight'] = unet_state['dconv_down1.d_conv.0.weight'][:, :NC, :, :]
                if unet_state['conv_last.weight'].shape[0] != NC:
                    unet_state['conv_last.weight'] = unet_state['conv_last.weight'][:NC, :, :, :]
                    unet_state['conv_last.bias'] = unet_state['conv_last.bias'][:NC]
                model.unets[i].load_state_dict(unet_state)
        print("Loaded pre-trained baseline U-Nets (Adapted to 4 frames)!")
    
    # --- PRELOAD FASTDVDNET (PnP Prior) ---
    fastdvd_path = os.path.join(BASE_DIR, "CACTI", "checkpoints", "fastdvd", "fastdvd_gray.pth")
    if os.path.exists(fastdvd_path):
        model.fastdvd.load_state_dict(torch.load(fastdvd_path, map_location=device))
        print("Loaded Universal Video PnP Prior (FastDVDnet)!")
    
    # Freeze FastDVDnet totally. It is a mathematical prior.
    for param in model.fastdvd.parameters():
        param.requires_grad = False
        
    print(f"Focusing training ONLY on Fusion cells and slight U-Net fine-tuning!")
        
    # 3. Dynamic Differential Optimizer
    optimizer = optim.Adam([
        {'params': model.unets.parameters(), 'lr': 1e-5},     # Slight fine-tuning
        {'params': model.fusions.parameters(), 'lr': 1e-4},   # Learn the fusion map fast
        {'params': model.sigmas, 'lr': 1e-4}                  # Learn PnP sigmas
    ])
    
    criterion = nn.MSELoss()
    
    # Phi logic moved inside loop to handle dynamic batch_size

    print("=" * 60)
    print("Environment OK! Ready for dry-run or full training.")
    # --- 4. Training Loop ---
    from torch.utils.tensorboard import SummaryWriter
    
    save_dir = os.path.join(BASE_DIR, "Research_GAP_STP_Dynamic_Hybrid", "checkpoints")
    log_dir = os.path.join(BASE_DIR, "Research_GAP_STP_Dynamic_Hybrid", "logs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    scaler = torch.cuda.amp.GradScaler() # Mixed precision for speed
    
    n_epochs = 20  # NC=4 has ~2x samples/epoch vs NC=8, so halve epochs to match total compute
    save_every_epochs = 1
    log_every_steps = 10
    
    # === RESUME LOGIC ===
    start_epoch = 0
    import glob
    existing_ckpts = glob.glob(os.path.join(save_dir, f"gap_hybrid_stp_nc{NC}_ep*.pth"))
    if len(existing_ckpts) > 0:
        latest_ckpt = sorted(existing_ckpts, key=lambda x: int(os.path.basename(x).split('ep')[1].split('.pth')[0]))[-1]
        print(f"[RESUME] Checkpoint detected! Resuming from: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False) # allow weights_only=False for optimizer state 
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"[FAST-FORWARD] Fast-forwarding to Epoch {start_epoch} / {n_epochs}")
    
    print("\n[START] Commencing Dynamic Hybrid Training Loop...")
    step = 0
    
    for epoch in range(start_epoch, n_epochs):
        model.train()
        # FastDVDnet should remain in eval mode to prevent batchnorm drifting if any
        model.fastdvd.eval() 
        
        epoch_loss = 0.0
        
        for i, data in enumerate(loader):
            gt, _ = data # gt: (B, 8, 256, 256), ignore pre-computed y
            gt = gt.float().to(device)
            B_current = gt.shape[0]
            
            # --- CURRICULUM LEARNING: OPERATOR AUGMENTATION (GPU ACCELERATED) ---
            import random
            import torch.nn.functional as F
            
            # Pre-load base mask to GPU
            base_mask_t = torch.from_numpy(mask_np).to(device) # (NC, H, W)
            base_mask_s_t = torch.from_numpy(mask_s_np).to(device) # (H, W)
            
            # First 20 epochs: Train on perfect mask to build strong baseline features
            # After 20 epochs: Introduce random sub-pixel shifts to build physical robustness
            if epoch >= 10:
                dx = random.uniform(-1.0, 1.0)
                dy = random.uniform(-1.0, 1.0)
                
                # affine_grid expects normalized coordinates [-1, 1]
                tx = 2.0 * dx / mask_np.shape[2]
                ty = 2.0 * dy / mask_np.shape[1]
                
                theta = torch.tensor([[
                    [1.0, 0.0, tx],
                    [0.0, 1.0, ty]
                ]], dtype=torch.float32, device=device).repeat(NC, 1, 1)
                
                # (NC, 1, H, W)
                mask_4d = base_mask_t.unsqueeze(1)
                grid = F.affine_grid(theta, mask_4d.size(), align_corners=False)
                # padding_mode='reflection' works well for edge continuity like 'wrap'
                aug_mask_t = F.grid_sample(mask_4d, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
                
                aug_mask_t = aug_mask_t.squeeze(1) # (NC, H, W)
                aug_mask_s_t = torch.clamp(torch.sum(aug_mask_t, dim=0), min=1e-6) # (H, W)
            else:
                aug_mask_t = base_mask_t
                aug_mask_s_t = base_mask_s_t
            
            # Dynamic batch allocation for mask arrays
            Phi = aug_mask_t.unsqueeze(0).repeat(B_current, 1, 1, 1) # (B, NC, H, W)
            Phi_s = aug_mask_s_t.unsqueeze(0).unsqueeze(0).repeat(B_current, 1, 1, 1) # (B, 1, H, W)
            
            # Re-generate measurement y dynamically using augmented mask
            # y = sum_t(gt_t * mask_t)
            y = torch.sum(gt * Phi, dim=1, keepdim=True) # (B, 1, 256, 256)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision to save VRAM and speed up
            with torch.cuda.amp.autocast():
                out = model(y, Phi, Phi_s)
                loss = criterion(out, gt)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping to ensure stability during hybrid fusion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            step += 1
            
            if step % log_every_steps == 0:
                print(f"Epoch [{epoch}/{n_epochs}] Step [{i}/{len(loader)}] | Loss (MSE): {loss.item():.6f}")
                writer.add_scalar('Train/Loss_MSE', loss.item(), step)
                
                # Monitor the learned sigmas
                mean_sigma = model.sigmas.data.mean().item()
                writer.add_scalar('Train/Mean_Learned_Sigma', mean_sigma, step)
                
        # Epoch Checkpointing
        avg_loss = epoch_loss / len(loader)
        print(f"🏁 Epoch {epoch} Complete | Avg Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % save_every_epochs == 0:
            ckpt_path = os.path.join(save_dir, f"gap_hybrid_stp_nc{NC}_ep{epoch:03d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"💾 Checkpoint saved to: {ckpt_path}")
            
    writer.close()
    print("🎉 Training Completed Successfully!")

if __name__ == "__main__":
    main()
