# GAP-STPNet: Dynamic Spatio-Temporal Parallel Network for Video SCI

## 📖 Introduction
Video Snapshot Compressive Imaging (SCI) faces a fundamental challenge: traditional convolutional networks (like U-Net) excel at spatial artifact suppression but often lack temporal coherence, while video-specific priors (like FastDVDnet) capture temporal dynamics well but struggle when faced with severe spatial aliasing.

**GAP-STPNet** resolves this bottleneck by introducing a novel **Spatio-Temporal Parallel (STP)** architecture. Embedded within the Generalized Alternating Projection (GAP) framework, our model processes spatial and temporal priors simultaneously in two parallel branches. A custom **Dynamic Fusion mechanism** then adaptively fuses these features, achieving superior reconstruction quality by leveraging the best of both domains.

Furthermore, real-world SCI hardware inevitably suffers from physical mask misalignment. To bridge the gap between simulation and optical experiments, this repository also introduces a robust **3D Blind Calibration** algorithm to computationally correct complex translation and rotation mismatches.

## 🧠 Architecture Overview
1. **Spatial Branch (U-Net)**: Dedicated to suppressing high-frequency spatial aliasing and learning robust intra-frame representations.
2. **Temporal Branch (FastDVDnet)**: Acts as a powerful temporal prior to maintain motion continuity and inter-frame consistency.
3. **Dynamic Fusion Cell**: A self-adaptive attention mechanism that computes spatial-temporal weights to optimally and dynamically combine the two branches at each GAP unfolding stage.
4. **3D Blind Calibration**: A coarse-to-fine physical mismatch optimization grid-search that corrects sub-pixel shifts and angular rotations of the coded aperture.

## 🚀 Getting Started
*(The following steps guide you on how to set up the environment and run the code.)*

### 1. Install Dependencies
Ensure you have PyTorch installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run 3D Blind Calibration
To execute the physical mask mismatch calibration (translation + rotation) on the Kerr effect data:
```bash
python test_blind_calibration.py
```

### 3. Validate Model
To run standard validation and evaluate the reconstruction quality of GAP-STPNet:
```bash
python validate.py
```

## 📁 Directory Structure
- `model.py`: The core architecture definition for the Hybrid Dynamic Fusion GAP-STP network.
- `test_blind_calibration.py`: Evaluates the robust 3D blind calibration strategy.
- `train.py` / `validate.py`: Training and evaluation scripts.
- `cacti/`: Core dependencies (extracted and refined from the CACTI framework).
- `checkpoints/`: Directory for pretrained model weights.
- `datasets/`: Includes masks and sample synthetic data.

## 🙏 Acknowledgements
* This project is built upon the open-source **[CACTI](https://github.com/ucaswangls/cacti)** framework and the foundational unfolding framework from **[GAP-net](https://github.com/mengziyi64/GAP-net)**.
* The 3D Blind Calibration algorithm is built upon the methodology and open-source code from **[Physics_World_Model](https://github.com/integritynoble/Physics_World_Model)**, as detailed in **[arXiv:2603.04538](https://arxiv.org/abs/2603.04538v1)**.
