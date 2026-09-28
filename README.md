# ARTI 502 – Deep Learning: Assignment 1
### Training-Testing a Neural Net on the MNIST Dataset using PyTorch
Imam Abdulrahman Bin Faisal University – College of Computer Science & IT · Year 2026, Term 1 · **Group 1**

| # | Member | Academic ID |
|---|---|---|
| 1 | Khalid Waleed Alhilal (Group Leader) | 2230000788 |
| 2 | Ali Khalid Albinali | 2230003813 |
| 3 | Mishary Waleed AlAwadh | 2230006217 |
| 4 | Nawaf Alsabr | 2230004159 |
| 5 | Yasir Saleh Alghamdi | 2230004157 |
| 6 | Ibrahim Alkuwaiz | 2230002878 |
| 7 | Meshal Alrubaie | 2230005991 |

**Repository:** https://github.com/KhalidExe/ARTI502-Assignment1

## Contents
| Path | Description |
|---|---|
| `Assigment-1 DL.ipynb` | Main notebook (all tasks, executed with outputs) – file name as required by the assignment |
| `Assigment-1 DL.py` | Plain-Python export of the notebook source |
| `screenshots/` | All figures and code/output screenshots, see `screenshots/INDEX.md` |
| `models/best_mlp.pt` | Weights of the best tuned MLP (Task 8c) |
| `requirements.txt` | Python dependencies (official PyTorch CUDA 13.0 wheels, torch 2.14.0) |
| `report/` | Filled assignment document (PDF + Word) with all screenshots and explanations |

## Tasks
1. **Install PyTorch** – project `.venv`, PyTorch 2.14.0 CUDA 13.0 build (Blackwell needs CUDA ≥ 12.8), GPU check, fixed seeds.
2. **Download & load MNIST** – torchvision download; 60k official train set split 50,000 train / 10,000 validation; official 10,000 test set.
3. **Inspect the dataset** – DataLoader batch tensor size `[32, 1, 28, 28]`, sample grid, class distribution.
4. **Simple network** – `Linear(784→512) → ReLU → Linear(512→10)` with `torch.flatten` (32×1×28×28 → 32×784).
5. **Training function** – `train(net, train_loader, epochs, lr, momentum, ...)` with `nn.CrossEntropyLoss` and SGD (learning rate + momentum); prints epoch / batch / loss / accuracy and returns all mini-batch losses.
6. **Training** – debug log, mini-batch loss curve, per-epoch loss & accuracy (train vs validation).
7. **Testing** – test loss/accuracy, predictions on test images, confusion matrix, per-class accuracy, misclassified examples.
8. **Tuning** – learning rate × momentum grid, hidden layers × neurons search; selection on the validation set only; single final test evaluation.
9. **Save** – source code, requirements, best model weights.

**Beyond requirements:** CNN with BatchNorm/Dropout + augmentation, Dropout/BatchNorm comparison on the MLP, LR scheduler + early stopping, CPU vs GPU timing, final comparison table.

## Results
All numbers come from the executed notebook (RTX 5070 Ti, seed 42). Hyper-parameters were selected on the **validation** split only; the test set was used for reporting.

| Model | Params | Val acc | Test loss | Test acc | Test errors |
|---|---|---|---|---|---|
| Task 6 baseline MLP 784-512-10 (lr 0.01, momentum 0.9, 10 epochs) | 407,050 | 97.92 % | 0.0603 | 98.26 % | 174 |
| **Task 8c tuned MLP 1×512 (lr 0.1, momentum 0.5, best epoch 15/20)** | 407,050 | 98.26 % | 0.0682 | **98.55 %** | 145 |
| Beyond: CNN + BatchNorm + Dropout + augmentation (early stopping) | 468,138 | 99.67 % | 0.0087 | **99.65 %** | 35 |

**Task 8 best results (validation):**

| Search | Highest accuracy | Lowest loss |
|---|---|---|
| Learning rate × momentum (8 epochs) | lr 0.1, momentum 0.5 → 98.12 % (loss 0.0795) | lr 0.01, momentum 0.9 → loss 0.0725 (97.87 %) |
| Hidden layers × neurons (8 epochs) | 1 × 512 → 98.12 % | 1 × 512 → loss 0.0795 |

Why not 100 %: a few MNIST test digits are ambiguous or arguably mislabelled, and reaching "100 %" would normally mean tuning on the test set (data leakage).

## Reproduce
```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```
Open `Assigment-1 DL.ipynb` in VS Code / Jupyter, select the `.venv` kernel and **Run All**. MNIST is downloaded to `./data` automatically. All seeds are fixed (`SEED = 42`).

## Hardware
AMD Ryzen 9 9950X · NVIDIA GeForce RTX 5070 Ti (16 GB) · 64 GB DDR5 · Windows. Exact software versions are printed in the notebook's *Hardware & Environment* section.
