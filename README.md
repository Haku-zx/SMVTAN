# Spatiotemporal Multi-View Trend-Aware Network for Traffic Flow Prediction (SMVTAN)

This repository provides the official PyTorch implementation of the SMVTAN model, proposed for traffic flow prediction by jointly modeling spatiotemporal dependencies and multi-view temporal trends.


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

## 🚀 How to Run

### Train the model:

```bash
python main.py --mode train --dataset PEMS03 --device cuda
```

or for CPU:

```bash
python main.py --mode train --dataset PEMS08 --device cpu
```


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
