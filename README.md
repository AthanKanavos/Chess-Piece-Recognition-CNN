# Chess Piece Recognition using Deep Convolutional Neural Networks

TensorFlow/Keras implementation of the convolutional neural network architectures presented in the paper:

**Chess Piece Recognition using Deep Convolutional Neural Networks**

This repository accompanies the associated publication and provides a TensorFlow/Keras implementation of the proposed methodology.

## Task

Six-class chess piece image classification:

- `bishop`
- `king`
- `knight`
- `pawn`
- `queen`
- `rook`

## Dataset

The paper uses the Kaggle **Chessman Image Dataset**:

```text
https://www.kaggle.com/datasets/niteshfre/chessman-image-dataset
```

The publication reports a total of 556 images:

| Class | Images |
|---|---:|
| Bishop | 87 |
| King | 76 |
| Knight | 106 |
| Pawn | 107 |
| Queen | 78 |
| Rook | 102 |
| **Total** | **556** |

Expected directory structure:

```text
dataset/
├── bishop/
├── king/
├── knight/
├── pawn/
├── queen/
└── rook/
```

Folder names are matched case-insensitively.

## CNN Architectures

### Architecture 1

```text
(Conv2D ×2 → MaxPooling2D → Dropout) ×3
→ GlobalAveragePooling2D
→ Flatten
→ Dense(256)
→ Dropout
→ Softmax Output
```

### Architecture 2

```text
((Conv2D → BatchNormalization) → Conv2D → MaxPooling2D → Dropout) ×3
→ GlobalAveragePooling2D
→ Flatten
→ Dense(256)
→ Dropout
→ Softmax Output
```

### Architecture 3

```text
(Conv2D ×3 → MaxPooling2D → Dropout) ×3
→ GlobalAveragePooling2D
→ Flatten
→ Dense(256)
→ Dropout
→ Softmax Output
```

## Implementation Details

The implementation follows the methodology presented in the paper and uses the following configuration:

- Input size: `224 × 224 × 3`
- Six-class softmax classification
- Kernel size: `3 × 3`
- Filter progression: `32 → 64 → 128`
- Activation: ReLU
- Optimizer: Adam
- Learning rate: `1e-3`
- Loss: Sparse categorical cross-entropy
- Block dropout: `0.25`
- Dense layer: `256` units
- Dense dropout: `0.50`
- Default epochs: `100`
- Data augmentation for robustness to viewpoint and illumination variation
- Class weighting
- Random seed: `42`

## Project Structure

```text
Chess-Piece-Recognition-CNN/
├── README.md
├── LICENSE
├── requirements.txt
├── train.py
├── evaluate.py
├── predict.py
├── .gitignore
├── outputs/
└── src/
    ├── __init__.py
    ├── data.py
    ├── models.py
    └── utils.py
```

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Training

Architecture 1:

```bash
python train.py \
  --data-dir "path/to/dataset" \
  --architecture 1 \
  --batch-size 128 \
  --epochs 100
```

Architecture 2:

```bash
python train.py \
  --data-dir "path/to/dataset" \
  --architecture 2 \
  --batch-size 128 \
  --epochs 100
```

Architecture 3:

```bash
python train.py \
  --data-dir "path/to/dataset" \
  --architecture 3 \
  --batch-size 128 \
  --epochs 100
```

Batch sizes evaluated in the paper:

```text
128, 256, 512, 1024, 2048, 4096
```

## Evaluation

```bash
python evaluate.py \
  --data-dir "path/to/dataset" \
  --model-path "outputs/architecture_2/best_model.keras"
```

The evaluation script produces:

- Loss
- Accuracy
- Confusion matrix
- Classification report
- Per-class precision, recall, and F1-score

## Prediction

```bash
python predict.py \
  --model-path "outputs/architecture_2/best_model.keras" \
  --image-path "path/to/chess-piece.jpg"
```

## Published Results

At 100 epochs, the publication reports the following results for batch size 128:

| Architecture | Final Loss | Final Accuracy |
|---|---:|---:|
| Architecture 1 | 0.1108 | 98.92% |
| Architecture 2 | 0.1173 | 98.98% |
| Architecture 3 | 0.1457 | 98.87% |

Results may vary depending on the software environment, preprocessing pipeline, random initialization, hyperparameter configuration, and hardware platform.

## Citation

If you use this implementation in your research, please cite the original paper.

```bibtex
@inproceedings{papadimitriou2023chess,
  title={Chess Piece Recognition using Deep Convolutional Neural Networks},
  author={Papadimitriou, Orestis and Kanavos, Athanasios and Maragoudakis, Manolis and Gerogiannis, Vassilis C.},
  year={2023}
}
```

## License

This project is released under the MIT License.
