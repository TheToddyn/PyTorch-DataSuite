# IA Training Pipeline

This project implements a highly efficient and flexible AI training pipeline using only PyTorch and native Python resources. The pipeline supports classification and analysis of large-scale datasets, including images, text, and tabular data.

**Version:** 2.0.0

## Features
- Modular structure
- GPU and multi-GPU support
- Early stopping
- Advanced logging (CSV, console)
- Automatic checkpoints
- Easy adaptation for different problems (images, text, tabular)
- Minimal external libraries (only PyTorch)
- Efficient processing for large datasets

## How to use
1. Install the main dependency:
   ```bash
   pip install torch
   ```
2. Organize your data in `./data/train` and `./data/val` (for images), or provide CSV/text files for tabular or NLP data.
3. Configure the pipeline in `config.yaml` for the desired data type.
4. Run training:
   ```bash
   python main.py
   ```

## Structure
- `main.py`: Main pipeline
- `model.py`: Model definitions (images, text, tabular)
- `dataset.py`: Flexible dataset and dataloaders
- `train.py`: Training function
- `utils.py`: Early stopping, logging, checkpoints
- `config.yaml`: Pipeline configuration
- `test_pipeline.py`: Automated tests

## License
MIT
