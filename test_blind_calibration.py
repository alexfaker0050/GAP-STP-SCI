"""
3D Blind Calibration Test (Translation + Rotation) on Kerr Effect Data.

Tests the NC=4 fine-tuned model against realistic physical mismatches
(sub-pixel X/Y shifts + Rotations). Uses GPU-accelerated affine transforms
to perform a 3D Coarse-to-Fine search.

Usage:
  python test_kerr_blind_calibration.py
"""
import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn.functional as F_torch
import einops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from cacti.utils.mask import generate_masks
from cacti.utils.metrics import compare_psnr
from model import GAP_STP_Net

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_STAGES = 12
NC = 4

# 3D Search Grid parameters
COARSE_STEP = 0.5
COARSE_RANGE_DX = np.arange(-1.0, 1.1, COARSE_STEP)
COARSE_RANGE_DY = np.arange(-1.0, 1.1, COARSE_STEP)
COARSE_RANGE_TH = np.arange(-1.5, 1.6, COARSE_STEP)
TOP_K = 2  # Trimmed for speed!

FINE_STEP_DX = np.arange(-0.4, 0.41, 0.1)
FINE_STEP_DY = np.arange(-0.4, 0.41, 0.1)
FINE_STEP_TH = np.arange(-0.4, 0.41, 0.1)

# ============================================================
# Mismatch Scenarios (dx, dy, theta)
# ============================================================
MISMATCH_SCENARIOS = [
    {"name": "Small_Shift",         "dx": 0.2,  "dy": 0.1,  "th": 0.3,  "scale": 1.0,  "blur": 0.0, "desc": "Small optical alignment error"},
    {"name": "Medium_Mis",          "dx": 0.5,  "dy": -0.3, "th": -0.8, "scale": 1.0,  "blur": 0.0, "desc": "Typical lab misalignment"},
    {"name": "Pure_Rotation",       "dx": 0.0,  "dy": 0.0,  "th": 2.5,  "scale": 1.0,  "blur": 0.0, "desc": "Significant lens rotation"},
    {"name": "Extreme_Diag",        "dx": 1.5,  "dy": -1.5, "th": 2.0,  "scale": 1.0,  "blur": 0.0, "desc": "Extreme spatial mismatch"},
    {"name": "Scale_Mismatch",      "dx": 0.0,  "dy": 0.0,  "th": 0.0,  "scale": 0.97, "blur": 0.0, "desc": "Magnification scaling error"},
    {"name": "Defocus_Blur",        "dx": 0.2,  "dy": 0.2,  "th": 0.5,  "scale": 1.0,  "blur": 1.0, "desc": "Out-of-focus blur error"},
    {"name": "Nightmare",           "dx": 1.0,  "dy": -0.5, "th": 1.2,  "scale": 1.02, "blur": 0.8, "desc": "Combined worst-case scenario"},
]


# ============================================================
# Dataset Loader for Kerr Test Data
# ============================================================
class KerrDataset(Dataset):
    def __init__(self, data_dir, nC=4):
        super().__init__()
        self.nC = nC
        self.samples = []
        seq_dirs = sorted(glob.glob(os.path.join(data_dir, "seq_*")))
        if len(seq_dirs) == 0:
            raise FileNotFoundError(f"No sequences found in {data_dir}")
        for seq_dir in seq_dirs:
            frames = sorted(glob.glob(os.path.join(seq_dir, "*.png")))
            n_frames = len(frames)
            for start in range(n_frames - nC + 1):
                self.samples.append((seq_dir, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_dir, start = self.samples[idx]
        frames = []
        for t in range(self.nC):
            img_path = os.path.join(seq_dir, f"{start + t:05d}.png")
            img = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
            frames.append(img)
        gt = np.stack(frames, axis=0)  # (NC, H, W)
        return torch.from_numpy(gt)


# ============================================================
# Core Physical Transform & Model Functions
# ============================================================

import torchvision.transforms.functional as TF

def apply_gpu_mismatch(mask_tensor, dx, dy, theta_deg, scale=1.0, blur_sigma=0.0):
    """
    Applies rigorous 3D spatial transformation (Shift X/Y + Rotation + Scale) and Optical Blur on GPU.
    mask_tensor: (1, NC, H, W)
    """
    B, cr, H, W = mask_tensor.shape
    rad = np.radians(theta_deg)
    cos_a = np.cos(rad) * scale
    sin_a = np.sin(rad) * scale
    
    # PyTorch affine_grid uses inverse mapping [cos, -sin, tx; sin, cos, ty]
    tx = -dx / (W / 2.0)
    ty = -dy / (H / 2.0)
    
    theta = torch.tensor([
        [cos_a, -sin_a, tx],
        [sin_a,  cos_a, ty]
    ], dtype=torch.float32, device=mask_tensor.device)
    
    theta = theta.unsqueeze(0).repeat(B * cr, 1, 1)
    mask_flat = mask_tensor.reshape(B * cr, 1, H, W)
    grid = F_torch.affine_grid(theta, mask_flat.size(), align_corners=False)
    shifted = F_torch.grid_sample(mask_flat, grid, mode='bilinear', padding_mode='border', align_corners=False)
    
    if blur_sigma > 0:
        kernel_size = int(2 * np.ceil(2 * blur_sigma) + 1)
        shifted = TF.gaussian_blur(shifted, kernel_size=[kernel_size, kernel_size], sigma=[blur_sigma, blur_sigma])
        
    return shifted.reshape(B, cr, H, W)


def generate_measurement(gt_tensor: torch.Tensor, mask_3d_tensor: torch.Tensor) -> torch.Tensor:
    """y = sum(x * mask). Adds realistic sensor noise."""
    y = torch.sum(gt_tensor * mask_3d_tensor, dim=1, keepdim=True)
    noise = torch.randn_like(y) * 0.005 # 0.5% sensor noise
    return torch.clamp(y + noise, 0.0, None)


def reconstruct_stp(model, y: torch.Tensor, mask_3d: torch.Tensor) -> torch.Tensor:
    """Run GAP-STP model reconstruction."""
    mask_s = torch.clamp(torch.sum(mask_3d, dim=1, keepdim=True), min=1e-6)
    with torch.no_grad():
        x_out = model(y, mask_3d, mask_s)
    return torch.clamp(x_out, 0.0, 1.0)


def compute_measurement_residual(y_true, x_hat, mask_3d):
    """Compute normalized measurement residual."""
    y_pred = torch.sum(x_hat * mask_3d, dim=1, keepdim=True)
    residual = torch.sum((y_true - y_pred) ** 2)
    norm = torch.sum(y_true ** 2) + 1e-10
    return (residual / norm).item()


def blind_calibrate_3d(model, mask_nom, calib_pairs):
    """
    Fast 3D Blind Calibration using Coarse Search + Multi-pass Coordinate Descent.
    mask_nom: Nominal simulation mask (1, NC, H, W) on GPU.
    calib_pairs: List of (y_true, gt) on GPU.
    """
    coarse_dx = np.arange(-1.0, 1.1, 0.5)
    coarse_dy = np.arange(-1.0, 1.1, 0.5)
    coarse_th = np.arange(-1.5, 1.6, 0.5)
    
    coarse_results = []
    for t_th in coarse_th:
        for t_dy in coarse_dy:
            for t_dx in coarse_dx:
                mask_cand = apply_gpu_mismatch(mask_nom, t_dx, t_dy, t_th)
                total_res = 0.0
                for y_cal, _ in calib_pairs:
                    recon = reconstruct_stp(model, y_cal, mask_cand)
                    total_res += compute_measurement_residual(y_cal, recon, mask_cand)
                avg_res = total_res / len(calib_pairs)
                coarse_results.append((avg_res, t_dx, t_dy, t_th))
                
    coarse_results.sort(key=lambda x: x[0])
    top_k = coarse_results[:2]
    
    best_res = 999.0
    best_dx, best_dy, best_th = 0.0, 0.0, 0.0
    fine_steps = np.arange(-0.4, 0.41, 0.1)
    
    for _, c_dx, c_dy, c_th in top_k:
        best_dx_local = c_dx
        best_dy_local = c_dy
        best_th_local = c_th
        current_best_res = 999.0
        
        for pass_idx in range(3):
            improved_in_pass = False
            
            # 1. Optimize dx
            res_dx_best = current_best_res
            best_step_dx = best_dx_local
            for step_dx in fine_steps:
                t_dx = best_dx_local + step_dx
                mask_cand = apply_gpu_mismatch(mask_nom, t_dx, best_dy_local, best_th_local)
                total_res = 0.0
                for y_cal, _ in calib_pairs:
                    recon = reconstruct_stp(model, y_cal, mask_cand)
                    total_res += compute_measurement_residual(y_cal, recon, mask_cand)
                res = total_res / len(calib_pairs)
                if res < res_dx_best:
                    res_dx_best = res
                    best_step_dx = t_dx
            if res_dx_best < current_best_res:
                best_dx_local = best_step_dx
                current_best_res = res_dx_best
                improved_in_pass = True
                    
            # 2. Optimize dy
            res_dy_best = current_best_res
            best_step_dy = best_dy_local
            for step_dy in fine_steps:
                t_dy = best_dy_local + step_dy
                mask_cand = apply_gpu_mismatch(mask_nom, best_dx_local, t_dy, best_th_local)
                total_res = 0.0
                for y_cal, _ in calib_pairs:
                    recon = reconstruct_stp(model, y_cal, mask_cand)
                    total_res += compute_measurement_residual(y_cal, recon, mask_cand)
                res = total_res / len(calib_pairs)
                if res < res_dy_best:
                    res_dy_best = res
                    best_step_dy = t_dy
            if res_dy_best < current_best_res:
                best_dy_local = best_step_dy
                current_best_res = res_dy_best
                improved_in_pass = True
                    
            # 3. Optimize th
            res_th_best = current_best_res
            best_step_th = best_th_local
            for step_th in fine_steps:
                t_th = best_th_local + step_th
                mask_cand = apply_gpu_mismatch(mask_nom, best_dx_local, best_dy_local, t_th)
                total_res = 0.0
                for y_cal, _ in calib_pairs:
                    recon = reconstruct_stp(model, y_cal, mask_cand)
                    total_res += compute_measurement_residual(y_cal, recon, mask_cand)
                res = total_res / len(calib_pairs)
                if res < res_th_best:
                    res_th_best = res
                    best_step_th = t_th
            if res_th_best < current_best_res:
                best_th_local = best_step_th
                current_best_res = res_th_best
                improved_in_pass = True
                
            if not improved_in_pass:
                break
                
        if current_best_res < best_res:
            best_res = current_best_res
            best_dx = best_dx_local
            best_dy = best_dy_local
            best_th = best_th_local
            
    return best_dx, best_dy, best_th


def evaluate_scenario(model, mask_recon, mask_true, calib_pairs):
    """Evaluate PSNR across all calibration scenes."""
    psnrs = []
    for y_true, gt in calib_pairs:
        recon = reconstruct_stp(model, y_true, mask_recon)
        # convert to numpy for PSNR
        recon_np = recon[0].cpu().numpy()
        gt_np = gt[0].cpu().numpy()
        
        if np.isnan(recon_np).any():
            print("NaN detected in recon_np!")
        
        p_sum = 0
        for t in range(NC):
            p_sum += compare_psnr(gt_np[t]*255, recon_np[t]*255)
        psnrs.append(p_sum / NC)
    return np.mean(psnrs)


# ============================================================
# Main Test Routine
# ============================================================
def main():
    print("=" * 80)
    print(f"3D Multi-Mismatch Blind Calibration Test (NC={NC})")
    print("=" * 80)
    
    # 1. Load Model
    print("\n[1/3] Loading Fine-tuned Kerr Model...")
    model = GAP_STP_Net(n_stages=N_STAGES, nC=NC).to(DEVICE)
    ckpt_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, f"gap_hybrid_stp_nc4_kerr_ep*.pth"))
    if not ckpts:
        raise FileNotFoundError("No fine-tuned Kerr checkpoint found! Please run finetune_kerr.py first.")
    latest = sorted(ckpts)[-1]
    print(f"  Loaded: {os.path.basename(latest)}")
    state = torch.load(latest, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    # 2. Load Mask
    print("\n[2/3] Loading Simulation Mask...")
    mask_path = os.path.join(SCRIPT_DIR, "datasets", "mask.mat")
    mask_full, _ = generate_masks(mask_path)
    mask_np = mask_full[:NC, :, :].astype(np.float32)
    mask_nom_gpu = torch.from_numpy(mask_np).unsqueeze(0).to(DEVICE) # (1, NC, H, W)
    
    # 3. Load Test Data
    print("\n[3/3] Loading Synthetic Kerr Test Data...")
    test_dir = os.path.join(SCRIPT_DIR, "datasets", "sample_data")
    dataset = KerrDataset(test_dir, nC=NC)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # We will use the first 5 samples as our "calibration/test batch" to speed up testing
    # In a real experiment, you only need 1-2 shots to calibrate.
    calib_gts = []
    for i, gt_batch in enumerate(loader):
        calib_gts.append(gt_batch.to(DEVICE))
        if i >= 4: # 5 scenes
            break
            
    print(f"  Selected {len(calib_gts)} scenes for evaluation.")
    
    # ============================================================
    # Run all mismatch scenarios
    # ============================================================
    results = []
    
    for idx, scenario in enumerate(MISMATCH_SCENARIOS):
        t_dx, t_dy, t_th = scenario["dx"], scenario["dy"], scenario["th"]
        scale, blur = scenario["scale"], scenario["blur"]
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{len(MISMATCH_SCENARIOS)}] {scenario['name']} | "
              f"dx={t_dx:.2f}, dy={t_dy:.2f}, th={t_th:.2f}°, s={scale}, b={blur}")
        print(f"Desc: {scenario['desc']}")
        print(f"{'='*80}")
        
        # Ground Truth physical mask
        mask_true = apply_gpu_mismatch(mask_nom_gpu, t_dx, t_dy, t_th, scale, blur)
        
        # Generate measurements
        calib_pairs = []
        for gt in calib_gts:
            y_true = generate_measurement(gt, mask_true)
            calib_pairs.append((y_true, gt))
            
        # ----------------------------------------------------
        # Evaluate Sc.II (Mismatch, Uncalibrated)
        # ----------------------------------------------------
        print("  Evaluating Sc.II (Mismatch)...", end=" ", flush=True)
        t0 = time.time()
        psnr_II = evaluate_scenario(model, mask_nom_gpu, mask_true, calib_pairs)
        print(f"{psnr_II:.2f} dB ({time.time()-t0:.1f}s)")
        
        # ----------------------------------------------------
        # Evaluate Sc.III (Oracle/Perfect Calibration)
        # ----------------------------------------------------
        print("  Evaluating Sc.III (Oracle)...", end=" ", flush=True)
        t0 = time.time()
        psnr_III = evaluate_scenario(model, mask_true, mask_true, calib_pairs)
        print(f"{psnr_III:.2f} dB ({time.time()-t0:.1f}s)")
        
        # ----------------------------------------------------
        # Run 3D Blind Calibration
        # ----------------------------------------------------
        print("  Running 3D Blind Calibration (Top-2 Search)...", flush=True)
        t0 = time.time()
        est_dx, est_dy, est_th = blind_calibrate_3d(model, mask_nom_gpu, calib_pairs)
        calib_time = time.time() - t0
        
        dx_err = abs(est_dx - t_dx)
        dy_err = abs(est_dy - t_dy)
        th_err = abs(est_th - t_th)
        
        print(f"    -> Est: dx={est_dx:.2f}, dy={est_dy:.2f}, th={est_th:.2f}°")
        print(f"    -> Err: dx={dx_err:.2f}, dy={dy_err:.2f}, th={th_err:.2f}° (Time: {calib_time:.1f}s)")
        
        # ----------------------------------------------------
        # Evaluate Sc.IV (Post-Blind Calibration)
        # ----------------------------------------------------
        mask_est = apply_gpu_mismatch(mask_nom_gpu, est_dx, est_dy, est_th)
        
        print("  Evaluating Sc.IV (Blind Calib)...", end=" ", flush=True)
        t0 = time.time()
        psnr_IV = evaluate_scenario(model, mask_est, mask_true, calib_pairs)
        print(f"{psnr_IV:.2f} dB ({time.time()-t0:.1f}s)")
        
        # Recovery rate
        delta_deg = psnr_III - psnr_II
        delta_rec = psnr_IV - psnr_II
        rho = delta_rec / max(delta_deg, 1e-6) * 100
        
        results.append({
            "name": scenario["name"],
            "true_dx": t_dx, "true_dy": t_dy, "true_th": t_th,
            "est_dx": est_dx, "est_dy": est_dy, "est_th": est_th,
            "dx_err": dx_err, "dy_err": dy_err, "th_err": th_err,
            "psnr_II": psnr_II, "psnr_III": psnr_III, "psnr_IV": psnr_IV,
            "rho": rho, "calib_time": calib_time
        })
        
        print(f"  >> Performance Recovery: {rho:.1f}%")
        
    # ============================================================
    # Output Summary Table & Chart
    # ============================================================
    print("\n" + "=" * 110)
    print("FINAL SUMMARY: 3D Blind Calibration (Shift + Rotation + Distortions)")
    print("=" * 115)
    print(f"{'Scenario':<16} {'True(x,y,th)':<20} {'Est(x,y,th)':<20} {'Err(x,y,th)':<20} "
          f"{'Sc.II':>7} {'Sc.IV':>7} {'Sc.III':>7} {'Recovery':>10} {'Time':>6}")
    print("-" * 115)
    
    for r in results:
        print(f"{r['name']:<16} ({r['true_dx']:+.2f},{r['true_dy']:+.2f},{r['true_th']:+.2f}) "
              f"({r['est_dx']:+.2f},{r['est_dy']:+.2f},{r['est_th']:+.2f}) "
              f"({r['dx_err']:.2f},{r['dy_err']:.2f},{r['th_err']:.2f}) "
              f"{r['psnr_II']:>7.2f} {r['psnr_IV']:>7.2f} {r['psnr_III']:>7.2f} "
              f"{r['rho']:>9.1f}% {r['calib_time']:>5.0f}s")
    
    print("-" * 115)
    
    # Chart Generation
    print("\nGenerating summary chart...")
    names = [r['name'] for r in results]
    p2 = [r['psnr_II'] for r in results]
    p4 = [r['psnr_IV'] for r in results]
    p3 = [r['psnr_III'] for r in results]
    rhos = [r['rho'] for r in results]
    
    x = np.arange(len(names))
    width = 0.25
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('3D Blind Calibration (Translation + Rotation) on Kerr Effect', fontsize=14, fontweight='bold')
    
    # PSNR Bars
    ax1.bar(x - width, p2, width, label='Sc.II (Mismatch)', color='#e74c3c')
    ax1.bar(x, p4, width, label='Sc.IV (Calibrated)', color='#2ecc71')
    ax1.bar(x + width, p3, width, label='Sc.III (Oracle)', color='#3498db')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Reconstruction Quality')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(20, 50)
    
    # Recovery Rate
    ax2.bar(x, rhos, 0.4, color='#9b59b6')
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel('Recovery Rate (%)')
    ax2.set_title('Performance Recovery via Blind Calib')
    ax2.set_ylim(0, 110)
    
    for i, v in enumerate(rhos):
        ax2.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
    plt.tight_layout()
    chart_path = os.path.join(SCRIPT_DIR, "kerr_3d_calibration_results.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    
    print(f"Saved Chart to: {chart_path}")
    print("Done!")


if __name__ == "__main__":
    main()
