# Quick Start Guide

This guide will help you get started with the Monocular 3D Object Detection project in 5 minutes.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- KITTI dataset downloaded

## Installation (2 minutes)

```bash
# 1. Install dependencies
pip install -r require.txt

# 2. Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Dataset Setup (3 minutes)

1. Download KITTI dataset from: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

2. Extract to `data/kitti/training/`:
   ```
   data/kitti/training/
   ├── image_2/
   ├── label_2/
   └── calib/
   ```

3. Verify:
   ```bash
   python -c "from dataset import KITTIDataset; from config import cfg; print('Dataset OK:', len(KITTIDataset(cfg.DATA_ROOT, 'training')) > 0)"
   ```

## Quick Training (30 seconds to start)

### Option 1: Default Configuration (Attention Model)
```bash
python main.py
```

### Option 2: Specific Model
```python
# Edit config.py
BACKBONE_TYPE = "transformer"  # simple, resnet, attention, transformer
EPOCHS = 50
BATCH_SIZE = 4
```

Then run:
```bash
python main.py
```

## Quick Evaluation

```bash
# Evaluate trained models
python evaluate.py

# Run cross-validation
python cross_validation.py

# Explore features
python feature_exploration.py
```

## Data Exploration

```bash
# Launch Jupyter notebook
jupyter notebook data_exploration.ipynb
```

## Project Structure

```
group_project/
├── main.py                    # ← Start here for training
├── evaluate.py                # ← Model evaluation
├── cross_validation.py        # ← Cross-validation
├── data_exploration.ipynb     # ← Data analysis
├── config.py                  # ← Configuration
├── model.py                   # ← Model architectures
├── dataset.py                 # ← Data loading
└── PROJECT_REPORT.md          # ← Full documentation
```

## Expected Results

After training for 50 epochs:
- Training time: ~2-4 hours (depends on GPU)
- Checkpoints saved to: `checkpoints/`
- Logs saved to: `logs/`
- Visualizations saved to: `visualizations/`

## Troubleshooting

### CUDA Out of Memory
```python
# In config.py, reduce batch size
BATCH_SIZE = 2  # or even 1
```

### Dataset Not Found
```bash
# Check dataset path
python -c "from config import cfg; print(cfg.DATA_ROOT)"
# Should print: /path/to/group_project/data/kitti
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r require.txt --upgrade
```

## Next Steps

1. **Read the full report**: [PROJECT_REPORT.md](PROJECT_REPORT.md)
2. **Explore the data**: Open `data_exploration.ipynb`
3. **Train multiple models**: Edit `config.py` and run `main.py`
4. **Compare models**: Run `evaluate.py`
5. **Validate robustness**: Run `cross_validation.py`

## Key Configuration Options

### Model Selection
```python
# config.py
BACKBONE_TYPE = "attention"  # simple, resnet, attention, transformer, ensemble
```

### Training Parameters
```python
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-4
WEIGHT_DECAY = 1e-5
```

### Loss Weights
```python
LOSS_WEIGHTS = {
    "location": 1.0,
    "dimension": 0.5,
    "orientation": 0.3
}
```

### Data Augmentation
```python
# In dataset.py, modify build_transforms()
RandomFlip(prob=0.5)
RandomBrightnessContrast()
```

## Getting Help

- Check [README.md](README.md) for detailed documentation
- Read [PROJECT_REPORT.md](PROJECT_REPORT.md) for methodology
- Review code comments in each module
- Open an issue on GitHub

## Performance Tips

1. **Use GPU**: Ensure CUDA is available
2. **Adjust batch size**: Based on GPU memory
3. **Enable mixed precision**: For faster training (advanced)
4. **Use data augmentation**: Improves generalization
5. **Monitor training**: Check `logs/train.log`

## Common Commands

```bash
# Training
python main.py

# Evaluation
python evaluate.py

# Cross-validation
python cross_validation.py

# Feature exploration
python feature_exploration.py

# Data analysis
jupyter notebook data_exploration.ipynb

# Check logs
tail -f logs/train.log

# View checkpoints
ls -lh checkpoints/

# View visualizations
ls -lh visualizations/
```

## Success Indicators

✅ Training loss decreases consistently  
✅ Validation loss stabilizes  
✅ AP@0.5 > 0.3 (reasonable baseline)  
✅ No CUDA errors  
✅ Checkpoints saved successfully  
✅ Visualizations generated  

## Congratulations!

You're now ready to explore monocular 3D object detection. Happy coding! 🚀
