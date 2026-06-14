# GAP-STP - Dynamic Hybrid Feature Fusion for Video SCI & 3D Blind Calibration

This repository contains the core implementation of **GAP-STP**, a dynamic hybrid fusion network for Video Snapshot Compressive Imaging (SCI), alongside a robust **3D Blind Calibration** algorithm.

## Features
- **Dynamic Hybrid Fusion**: Combines U-Net spatial detail suppression with FastDVDnet temporal continuity.
- **3D Blind Calibration**: Corrects complex mismatch scenarios (translation + rotation) using physical mask optimization.
- **Standalone**: Cleaned up to be independent, with local references and relative paths.

## Directory Structure
- `model.py`: The architecture definition for the Hybrid Dynamic Fusion GAP-STP network (U-Net + FastDVDnet).
- `test_blind_calibration.py`: Evaluates the blind calibration strategy.
- `train.py` / `validate.py`: Training and evaluation scripts.
- `cacti/`: Core dependencies (extracted from CACTI framework).
- `checkpoints/`: Pretrained weights.
- `datasets/`: Includes masks and sample synthetic data.

## Getting Started

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run Blind Calibration**
```bash
python test_blind_calibration.py
```

3. **Validate Model**
```bash
python validate.py
```

## Acknowledgements
* This project is built upon the open-source **[CACTI](https://github.com/ucaswangls/cacti)** framework and the foundational unfolding framework from **[GAP-net](https://github.com/mengziyi64/GAP-net)**.
* The 3D Blind Calibration algorithm is built upon the methodology and open-source code from **[Physics_World_Model](https://github.com/integritynoble/Physics_World_Model)**, as detailed in **[arXiv:2603.04538](https://arxiv.org/abs/2603.04538v1)**.
