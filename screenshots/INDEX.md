# Screenshots index – ARTI 502 Assignment 1 (Group 1)

* `*_vscode_*.jpg` – screenshots of the notebook **code cells and their outputs in VS Code** (taken on the RTX 5070 Ti machine after *Run All*).
* `*.png` – figures saved by the notebook itself with `plt.savefig(..., dpi=200)`.

| File | Task / deliverable it satisfies |
|---|---|
| `task01_vscode_load_pytorch.jpg` | Task 1 – loading PyTorch in the source code, library versions, `CUDA ready: True` |
| `task01_vscode_gpu_check.jpg` | Task 1 – GPU verification (RTX 5070 Ti, tensor op on `cuda:0`), fixed seeds |
| `task02_vscode_dataset_loaded.jpg` | Task 2 – MNIST downloaded from code and loaded (50,000 train / 10,000 val / 10,000 test) |
| `task03_vscode_tensor_size_and_images.jpg` | Task 3 (1) images tensor size `[32, 1, 28, 28]` and (2) matplotlib visualisation |
| `task03_samples.png` | Task 3 – grid of one mini-batch with labels |
| `task03_vscode_class_distribution.jpg`, `task03_class_distribution.png` | Task 3 – class distribution of train / validation / test |
| `task03_vscode_gpu_data_pipeline.jpg` | Data pipeline used from Task 5 on (tensors pre-loaded on the GPU) |
| `task04_vscode_simple_network.jpg` | Task 4 – network source code (Linear → ReLU → Linear, `torch.flatten`) and shape verification (32×1×28×28 → 32×784) |
| `task05_vscode_train_function_1.jpg`, `task05_vscode_train_function_2.jpg` | Task 5 – training function (CrossEntropyLoss + SGD with learning rate & momentum) |
| `task05_vscode_train_function_2.jpg` | Task 6 – instantiating a new network and starting training (bottom of the image) |
| `task06_vscode_debug_log.jpg` | Task 6 – debug output: epoch, batch number, current loss, accuracy |
| `task06_vscode_loss_accuracy_graphs.jpg`, `task06_minibatch_loss.png`, `task06_epoch_curves.png` | Task 6 – graphs of all mini-batch losses and per-epoch loss / accuracy |
| `task07_vscode_test_results.jpg`, `task07_test_predictions.png` | Task 7 – testing source code, test loss/accuracy, predictions on test images |
| `task07_vscode_confusion_matrix.jpg`, `task07_vscode_per_class_accuracy.jpg`, `task07_confusion_matrix.png` | Task 7 – confusion matrix and per-class accuracy |
| `task07_vscode_misclassified.jpg`, `task07_misclassified.png` | Task 7 – misclassified test images |
| `task08a_vscode_lr_momentum_table.jpg`, `task08a_vscode_heatmaps.jpg`, `task08a_lr_momentum_heatmaps.png` | Task 8a – learning rate × momentum tuning, best results |
| `task08b_vscode_mlp_code.jpg`, `task08b_vscode_architecture_table.jpg`, `task08b_vscode_architecture_plot.jpg`, `task08b_architecture.png` | Task 8b – number of layers × neurons tuning, best results |
| `task08c_vscode_final_training.jpg`, `task08c_vscode_summary_code.jpg`, `task08c_vscode_best_results.jpg`, `task08c_final_model.png` | Task 8c – final model selected on validation, best loss / accuracy summary |
| `task09_vscode_save_model.jpg` | Task 9 – saving the best model and reloading it |
| `beyond_b1_*` | Beyond – Dropout / BatchNorm on the MLP |
| `beyond_b2_*` | Beyond – CNN with BatchNorm, Dropout, augmentation, LR scheduler, early stopping |
| `beyond_b3_*` | Beyond – CPU vs GPU training time |
| `beyond_b4_*` | Beyond – comparison of all models |
| `hardware_vscode_specs.jpg`, `hardware_vscode_timings.jpg` | Hardware & Environment – detected specs and training times |
