# Spatiotemporal Multi-View Trend-Aware Network for Traffic Flow Prediction (SMVTAN)

This repository provides the official PyTorch implementation of the SMVTAN model, proposed for traffic flow prediction by jointly modeling spatiotemporal dependencies and multi-view temporal trends.

**Authors:** Linlong Chen, Linbiao Chen, Hongyan Wang, Jian Zhao

---

## 📁 Project Structure

```
.
├── config/                    # Dataset-specific configuration files
│   ├── PEMS03.json
│   ├── PEMS04.json
│   ├── PEMS07.json
│   └── PEMS08.json
│
├── data/                      # Traffic data and adjacency matrices
│   ├── PEMS03/
│   ├── PEMS04/
│   ├── PEMS07/
│   ├── PEMS08/
│   ├── adj_PEMS03_001.csv
│   ├── adj_PEMS04_001.csv
│   ├── adj_PEMS07_001.csv
│   └── adj_PEMS08_001.csv
│
├── params/                    # Temporal sampling parameter sets
│   ├── PEMS03_TS_4_0.95_32
│   ├── PEMS04_TS_4_0.95_32
│   ├── PEMS07_TS_4_0.95_32
│   └── PEMS08_TS_4_0.95_32
│
├── utils/                     # Utility functions and model runner
│   ├── data_process.py
│   ├── evaluation.py
│   ├── logs.py
│   ├── model_fit.py
│   └── setting.py
│
├── SMVTAN.py                  # Model architecture
├── main.py                    # Training and testing entry point
├── README.md                  # This documentation
```

---

## 🔧 Environment Setup

**Python ≥ 3.8**  
**PyTorch ≥ 1.10**  
(Optional) CUDA ≥ 11.3 for GPU acceleration

### Install required packages:

```bash
pip install torch torchvision
pip install numpy pandas matplotlib
pip install scikit-learn tqdm
```

---

## 📊 Dataset Access

This project supports the following datasets:

- PEMS03 (358 nodes)
- PEMS04 (307 nodes)
- PEMS07 (883 nodes)
- PEMS08 (170 nodes)

Relevant datasets for **standard prediction tasks** are available in the repository or can be reproduced.  
However, the **long-term prediction datasets (60-step horizon)** are too large to upload.  
If needed, please feel free to contact me to request access.

## 🚀 How to Run

### Train the model:

```bash
python main.py --mode train --dataset PEMS03 --device cuda
```

or for CPU:

```bash
python main.py --mode train --dataset PEMS08 --device cpu
```

### Test the model:

```bash
python main.py --mode test --dataset PEMS03 --checkpoint ./checkpoints/PEMS03_best.pth
```

All results and logs are saved automatically.

---

## 📈 Evaluation Metrics

During testing, the following metrics are reported:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error

---

## ✏️ Citation

If this project is helpful to your research, please cite:

```
bibtex
@article{chen2025smvtan,
  title={Spatiotemporal Multi-View Trend-Aware Network for Traffic Flow Prediction},
  author={Chen, Linlong and Chen, Linbiao and Wang, Hongyan and Zhao, Jian},
  journal={Under Review},
  year={2025}
}
```

---

## 🔒 License

This repository is released for **academic research use only**.  
For commercial usage, please contact the authors.
