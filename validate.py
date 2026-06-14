"""
Validation Script for 4-Frame (NC=4) Fine-tuned GAP-STP Model.

Runs the InverseNet 4-Scenario protocol identical to the 8-frame version,
adapted for the NC=4 model from finetune_8to4.py.

Scenarios:
  I   (Ideal)       : y = Phi_true @ x,  recon with Phi_true   → oracle upper bound
  II  (Mismatch)    : y = Phi_true @ x,  recon with Phi_nom    → mismatch baseline
  III (Oracle Calib) : y = Phi_true @ x,  recon with Phi_true   → = Scenario I
  IV  (Blind Calib)  : y = Phi_true @ x,  recon with Phi_est    → grid search estimated

Usage:
  python validate_nc4.py
"""

import os
import sys
import time
import glob
import numpy as np
import torch
import einops
from scipy.ndimage import shift as ndshift
from torch.utils.data import DataLoader

# ============================================================
# Path setup
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from cacti.datasets.builder import build_dataset
from cacti.utils.config import Config
from cacti.utils.mask import generate_masks
from cacti.utils.metrics import compare_psnr, compare_ssim
from model import GAP_STP_Net

# ============================================================
# Configuration
# ============================================================
DEVICE = torch.device("cuda:0")
N_STAGES = 12
NC = 4  # <--- 4-frame model

# InverseNet mismatch parameters (Rigorous off-grid values)
TRUE_DX = 0.52   # pixels
TRUE_DY = 0.28   # pixels

# Blind calibration grid search range
CALIB_DX_RANGE = np.arange(-1.0, 1.1, 0.2)  # 11 points
CALIB_DY_RANGE = np.arange(-1.0, 1.1, 0.2)  # 11 points


def apply_subpixel_shift(mask_3d: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Apply sub-pixel spatial shift to each frame of a 3D mask (cr, H, W)."""
    shifted = np.zeros_like(mask_3d)
    for t in range(mask_3d.shape[0]):
        shifted[t] = ndshift(mask_3d[t], [dy, dx], order=1, mode='wrap')
    return shifted.astype(np.float32)


def generate_measurement(gt: np.ndarray, mask_3d: np.ndarray) -> np.ndarray:
    """Generate CACTI compressed measurement: y = sum_t(mask_t * x_t).
    Added rigorous measurement noise (Gaussian sigma=0.01) matching InverseNet protocol.
    """
    y = np.sum(gt * mask_3d, axis=0).astype(np.float32)
    noise = np.random.normal(0, 0.01, y.shape).astype(np.float32)
    return np.clip(y + noise, 0, None)


def reconstruct_stp(model, y: np.ndarray, mask_3d: np.ndarray,
                      mask_s: np.ndarray, device: torch.device) -> np.ndarray:
    """Run GAP-STP model reconstruction."""
    y_t = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float().to(device)
    Phi = torch.from_numpy(einops.repeat(mask_3d, 'cr h w->b cr h w', b=1)).to(device)
    mask_s_safe = np.maximum(mask_s, 1e-6)
    Phi_s = torch.from_numpy(einops.repeat(mask_s_safe, 'h w->b 1 h w', b=1)).to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            x_out = model(y_t, Phi, Phi_s)
    
    out = x_out[0].cpu().numpy()
    return np.clip(out, 0.0, 1.0)


def compute_measurement_residual(y_true, x_hat, mask_3d):
    """Compute normalized measurement residual for blind calibration."""
    y_pred = generate_measurement(x_hat, mask_3d)
    residual = np.sum((y_true - y_pred) ** 2)
    norm = np.sum(y_true ** 2) + 1e-10
    return residual / norm


def compute_metrics(gt: np.ndarray, recon: np.ndarray, nC: int):
    """Compute average PSNR and SSIM over all frames."""
    psnr_sum, ssim_sum = 0.0, 0.0
    for t in range(nC):
        psnr_sum += compare_psnr(gt[t] * 255, recon[t] * 255)
        ssim_sum += compare_ssim(gt[t] * 255, recon[t] * 255)
    return psnr_sum / nC, ssim_sum / nC


def main():
    print("=" * 70)
    print(f"InverseNet 4-Scenario Validation for NC={NC} Fine-tuned Model")
    print("=" * 70)
    
    # ========================================
    # 1. Load 4-Frame Model
    # ========================================
    print(f"\n[1/5] Loading GAP_STP_Net (NC={NC})...")
    model = GAP_STP_Net(n_stages=N_STAGES, nC=NC).to(DEVICE)
    
    ckpt_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, f"gap_hybrid_stp_nc{NC}_ep*.pth"))
    if not ckpts:
        print(f"ERROR: No NC={NC} checkpoint found in {ckpt_dir}")
        return
    latest_ckpt = sorted(ckpts, key=lambda x: int(os.path.basename(x).split('ep')[1].split('.pth')[0]))[-1]
    print(f"  Loading: {os.path.basename(latest_ckpt)}")
    state = torch.load(latest_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    train_loss = state.get('loss', None)
    train_epoch = state.get('epoch', None)
    print(f"  [OK] Model loaded (epoch={train_epoch}, train_loss={train_loss:.6f})")
    
    # ========================================
    # 2. Load Test Data & Masks (4-frame sliced)
    # ========================================
    print(f"\n[2/5] Loading test data with NC={NC} mask...")
    mask_path = os.path.join(SCRIPT_DIR, "datasets", "mask.mat")
    mask_full, _ = generate_masks(mask_path)
    mask_nom = mask_full[:NC, :, :].astype(np.float32)
    mask_s_nom = np.maximum(np.sum(mask_nom, axis=0), 1e-6).astype(np.float32)
    print(f"  Mask shape: {mask_nom.shape} (sliced from {mask_full.shape[0]} to {NC} frames)")
    
    test_data_cfg = Config({
        "type": "SixGraySimData",
        "data_root": os.path.join(SCRIPT_DIR, "datasets", "test"),
        "mask_path": mask_path,
        "mask_shape": None
    })
    test_data = build_dataset(test_data_cfg, {"mask": mask_nom})
    loader = DataLoader(test_data, batch_size=1, shuffle=False)
    print(f"  [OK] Loaded {len(test_data)} test scenes")
    
    # ========================================
    # 3. Generate mismatched masks
    # ========================================
    print(f"\n[3/5] Generating mismatched operator (dx={TRUE_DX}px, dy={TRUE_DY}px)...")
    mask_true = apply_subpixel_shift(mask_nom, TRUE_DX, TRUE_DY)
    mask_s_true = np.sum(mask_true, axis=0, keepdims=False).astype(np.float32)
    print("  [OK] True operator Phi_true generated")
    
    # ========================================
    # 4. Run Scenarios I, II, III
    # ========================================
    print(f"\n[4/5] Running 4-Scenario Evaluation (NC={NC})...")
    print("-" * 70)
    
    results = {
        'scene': [],
        'psnr_I': [], 'ssim_I': [],
        'psnr_II': [], 'ssim_II': [],
        'psnr_III': [], 'ssim_III': [],
    }
    
    for data_iter, data in enumerate(loader):
        meas_orig, gt_batch = data
        gt_np = gt_batch[0].numpy()  # (compress_batch, NC, H, W)
        
        name = test_data.data_name_list[data_iter]
        _name = name.split("_")[0] if "_" in name else name.split(".")[0]
        
        compress_batch = gt_np.shape[0]
        
        scene_psnr = {'I': 0, 'II': 0, 'III': 0}
        scene_ssim = {'I': 0, 'II': 0, 'III': 0}
        n_total = 0
        
        for ii in range(compress_batch):
            gt_single = gt_np[ii]  # (NC, H, W)
            
            # Generate measurement using TRUE (mismatched) operator
            y_true = generate_measurement(gt_single, mask_true)
            
            # Scenario I (Ideal): Recon with TRUE operator
            recon_I = reconstruct_stp(model, y_true, mask_true, mask_s_true, DEVICE)
            p_I, s_I = compute_metrics(gt_single, recon_I, NC)
            
            # Scenario II (Mismatch): Recon with NOMINAL operator
            recon_II = reconstruct_stp(model, y_true, mask_nom, mask_s_nom, DEVICE)
            p_II, s_II = compute_metrics(gt_single, recon_II, NC)
            
            # Scenario III (Oracle): Same as I
            p_III, s_III = p_I, s_I
            
            scene_psnr['I'] += p_I
            scene_psnr['II'] += p_II
            scene_psnr['III'] += p_III
            scene_ssim['I'] += s_I
            scene_ssim['II'] += s_II
            scene_ssim['III'] += s_III
            n_total += 1
        
        for k in ['I', 'II', 'III']:
            scene_psnr[k] /= n_total
            scene_ssim[k] /= n_total
        
        delta_deg = scene_psnr['I'] - scene_psnr['II']
        delta_rec = scene_psnr['III'] - scene_psnr['II']
        rho = delta_rec / max(delta_deg, 1e-6) * 100
        
        results['scene'].append(_name)
        results['psnr_I'].append(scene_psnr['I'])
        results['ssim_I'].append(scene_ssim['I'])
        results['psnr_II'].append(scene_psnr['II'])
        results['ssim_II'].append(scene_ssim['II'])
        results['psnr_III'].append(scene_psnr['III'])
        results['ssim_III'].append(scene_ssim['III'])
        
        print(f"  {_name:<10} | I: {scene_psnr['I']:.2f} dB | "
              f"II: {scene_psnr['II']:.2f} dB | "
              f"III: {scene_psnr['III']:.2f} dB | "
              f"D_deg: {delta_deg:.2f} | rho: {rho:.1f}%")
    
    # Summary Table
    print("\n" + "=" * 70)
    print(f"SUMMARY: NC={NC} Model — InverseNet Scenarios I~III")
    print("=" * 70)
    print(f"{'Scene':<12} {'Sc.I (dB)':>10} {'Sc.II (dB)':>10} {'Sc.III (dB)':>12} {'D_deg':>8} {'D_rec':>8} {'rho(%)':>8}")
    print("-" * 70)
    
    for i, scene in enumerate(results['scene']):
        p1 = results['psnr_I'][i]
        p2 = results['psnr_II'][i]
        p3 = results['psnr_III'][i]
        dd = p1 - p2
        dr = p3 - p2
        rho = dr / max(dd, 1e-6) * 100
        print(f"{scene:<12} {p1:>10.2f} {p2:>10.2f} {p3:>12.2f} {dd:>8.2f} {dr:>8.2f} {rho:>7.1f}%")
    
    avg_I = np.mean(results['psnr_I'])
    avg_II = np.mean(results['psnr_II'])
    avg_III = np.mean(results['psnr_III'])
    avg_dd = avg_I - avg_II
    avg_dr = avg_III - avg_II
    avg_rho = avg_dr / max(avg_dd, 1e-6) * 100
    print("-" * 70)
    print(f"{'AVERAGE':<12} {avg_I:>10.2f} {avg_II:>10.2f} {avg_III:>12.2f} {avg_dd:>8.2f} {avg_dr:>8.2f} {avg_rho:>7.1f}%")
    print("=" * 70)
    
    # ========================================
    # 5. Scenario IV: Blind Calibration
    # ========================================
    print(f"\n[5/5] Running Scenario IV: Blind Calibration...")
    print(f"  Strategy: Multi-scene joint residual + Top-K multi-start fine search")
    
    # Prepare calibration data from ALL scenes
    calib_pairs = []
    for data_iter, data in enumerate(loader):
        meas_orig, gt_batch = data
        gt_first = gt_batch[0].numpy()[0]  # First group of each scene
        y_true = generate_measurement(gt_first, mask_true)
        calib_pairs.append((y_true, gt_first))
    print(f"  Using {len(calib_pairs)} scenes for joint residual computation")
    
    # --- Stage 1: Coarse Search ---
    print(f"  [Stage 1] Coarse search: dx, dy in [-1.0, 1.0] px, step 0.2 px")
    coarse_results = []
    
    t0 = time.time()
    total_candidates = len(CALIB_DX_RANGE) * len(CALIB_DY_RANGE)
    
    for i, test_dy in enumerate(CALIB_DY_RANGE):
        for j, test_dx in enumerate(CALIB_DX_RANGE):
            mask_cand = apply_subpixel_shift(mask_nom, test_dx, test_dy)
            mask_s_cand = np.maximum(np.sum(mask_cand, axis=0), 1e-6).astype(np.float32)
            
            total_res = 0.0
            for y_cal, _ in calib_pairs:
                recon_cand = reconstruct_stp(model, y_cal, mask_cand, mask_s_cand, DEVICE)
                total_res += compute_measurement_residual(y_cal, recon_cand, mask_cand)
            avg_res = total_res / len(calib_pairs)
            
            coarse_results.append((avg_res, test_dx, test_dy))
            
            done = i * len(CALIB_DX_RANGE) + j + 1
            if done % 5 == 0 or done == total_candidates:
                elapsed = time.time() - t0
                eta = elapsed / done * (total_candidates - done)
                cur_best = min(coarse_results, key=lambda x: x[0])
                print(f"\r  Progress: {done}/{total_candidates} | "
                      f"Best so far: dx={cur_best[1]:.1f}, dy={cur_best[2]:.1f} | "
                      f"ETA: {eta:.0f}s", end="", flush=True)
    
    # Top-K
    TOP_K = 3
    coarse_results.sort(key=lambda x: x[0])
    top_k = coarse_results[:TOP_K]
    print(f"\n  [Stage 1 Complete] Top-{TOP_K} coarse candidates:")
    for rank, (res, dx, dy) in enumerate(top_k):
        print(f"    #{rank+1}: dx={dx:.2f}, dy={dy:.2f} (residual={res:.6f})")
    
    # --- Stage 2: Fine Search ---
    print(f"  [Stage 2] Fine search around Top-{TOP_K} candidates, step 0.05 px...")
    best_dx, best_dy = top_k[0][1], top_k[0][2]
    best_residual = top_k[0][0]
    
    t1 = time.time()
    total_fine = 0
    for rank, (_, center_dx, center_dy) in enumerate(top_k):
        fine_dx_range = np.arange(center_dx - 0.2, center_dx + 0.21, 0.05)
        fine_dy_range = np.arange(center_dy - 0.2, center_dy + 0.21, 0.05)
        
        for test_dy in fine_dy_range:
            for test_dx in fine_dx_range:
                mask_cand = apply_subpixel_shift(mask_nom, test_dx, test_dy)
                mask_s_cand = np.maximum(np.sum(mask_cand, axis=0), 1e-6).astype(np.float32)
                
                total_res = 0.0
                for y_cal, _ in calib_pairs:
                    recon_cand = reconstruct_stp(model, y_cal, mask_cand, mask_s_cand, DEVICE)
                    total_res += compute_measurement_residual(y_cal, recon_cand, mask_cand)
                avg_res = total_res / len(calib_pairs)
                
                if avg_res < best_residual:
                    best_residual = avg_res
                    best_dx, best_dy = test_dx, test_dy
                
                total_fine += 1
        
        print(f"    Candidate #{rank+1} done. Global best: dx={best_dx:.2f}, dy={best_dy:.2f}")
    
    elapsed_fine = time.time() - t1
    print(f"  [Stage 2 Complete] {total_fine} fine candidates searched in {elapsed_fine:.0f}s")
    
    print(f"\n  >>> Blind calibration result:")
    print(f"     Estimated:  dx={best_dx:.2f} px, dy={best_dy:.2f} px")
    print(f"     True value: dx={TRUE_DX:.2f} px, dy={TRUE_DY:.2f} px")
    print(f"     Error:      d_dx={abs(best_dx - TRUE_DX):.2f} px, d_dy={abs(best_dy - TRUE_DY):.2f} px")
    
    # Evaluate Scenario IV
    print("\n  Evaluating Scenario IV with estimated operator...")
    mask_est = apply_subpixel_shift(mask_nom, best_dx, best_dy)
    mask_s_est = np.maximum(np.sum(mask_est, axis=0), 1e-6).astype(np.float32)
    
    psnr_IVs = []
    ssim_IVs = []
    
    for data_iter, data in enumerate(loader):
        meas_orig, gt_batch = data
        gt_np = gt_batch[0].numpy()
        name = test_data.data_name_list[data_iter]
        _name = name.split("_")[0] if "_" in name else name.split(".")[0]
        
        compress_batch = gt_np.shape[0]
        p_sum, s_sum, n = 0.0, 0.0, 0
        
        for ii in range(compress_batch):
            gt_single = gt_np[ii]
            y_true = generate_measurement(gt_single, mask_true)
            recon_IV = reconstruct_stp(model, y_true, mask_est, mask_s_est, DEVICE)
            p, s = compute_metrics(gt_single, recon_IV, NC)
            p_sum += p
            s_sum += s
            n += 1
        
        avg_p = p_sum / n
        avg_s = s_sum / n
        psnr_IVs.append(avg_p)
        ssim_IVs.append(avg_s)
        
        oracle_recovery = (avg_p - results['psnr_II'][data_iter]) / \
                          max(results['psnr_I'][data_iter] - results['psnr_II'][data_iter], 1e-6) * 100
        
        print(f"  {_name:<10} | IV: {avg_p:.2f} dB (SSIM: {avg_s:.4f}) | "
              f"Oracle recovery: {oracle_recovery:.1f}%")
    
    avg_IV = np.mean(psnr_IVs)
    avg_IV_ssim = np.mean(ssim_IVs)
    overall_recovery = (avg_IV - avg_II) / max(avg_I - avg_II, 1e-6) * 100
    
    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "=" * 70)
    print(f"FINAL SUMMARY: NC={NC} Fine-tuned Model under InverseNet Protocol")
    print("=" * 70)
    print(f"  Checkpoint: {os.path.basename(latest_ckpt)}")
    print(f"  Scenario I   (Ideal/Oracle)  : {avg_I:.2f} dB")
    print(f"  Scenario II  (Mismatch)      : {avg_II:.2f} dB")
    print(f"  Scenario III (Oracle Calib)   : {avg_III:.2f} dB")
    print(f"  Scenario IV  (Blind Calib)    : {avg_IV:.2f} dB (SSIM: {avg_IV_ssim:.4f})")
    print(f"  -----------------------------------")
    print(f"  D_degradation (I->II)        : {avg_dd:.2f} dB")
    print(f"  D_recovery (III-II)          : {avg_dr:.2f} dB")
    print(f"  rho_oracle                   : {avg_rho:.1f}%")
    print(f"  rho_blind (IV recovery)      : {overall_recovery:.1f}%")
    print(f"  Blind calibration accuracy   : dx_err={abs(best_dx-TRUE_DX):.2f}px, dy_err={abs(best_dy-TRUE_DY):.2f}px")
    print("=" * 70)
    print("  Done!")


if __name__ == "__main__":
    main()
