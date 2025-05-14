# Deep Learning Module

This module implements deep learning models for Dota 2 match prediction, focusing on temporal sequence modeling of match data.

## 🧠 Models

### WinPredictor

A ReccurentDeepNetwork-based model architecture that processes match data as a time series to predict win probability.

Architecture:
- Input: Match features over time (shape: `[batch_size, seq_len, n_features]`)
- Reccurent deep network layers
- Dense layers with dropout
- Output: Win probability (shape: `[batch_size, 1]`)

## 🛠️ Usage

### Data Preparation

Before training models, you need to prepare the dataset:
```bash
tar -xzvf ../../data_for_live_dl_training.tgz -C ../../data_new/fetched_datasets/
```

### Training

```bash
python train.py --data_path=../data_new/fetched_datasets/match_data__[start_time_start-1743800400].json --output_dir=training/output/
```

### Model Configuration

Edit `config.py` file:

```python
@dataclass
class ModelConfig:
    input_size: int = 17 # [2, 6]  # Number of features per timestep
    ...

    # For deserrializing when predicting
    @classmethod
    def from_json(cls, json_path: str) -> 'ModelConfig':
        with open(json_path, 'r') as f:
            config_dict = json.load(f)["model"]
        return cls(**config_dict)

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 3
    ...
    

@dataclass
class DataConfig:
    max_game_length: int = 60  # Maximum game length in minutes
    min_game_length: int = 10  # Minimum game length to consider
    time_window_size: int = 10  # Size of time windows for validation
    ...

@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    experiment_name: str = "win_predictor"
    ...

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()
```
See config.py for full config

## 📊 Data Processing

### MatchDataProcessor

Handles preprocessing of match data into model-ready format:

```python
from src.experiments.src.dl.data.processor import MatchDataProcessor

processor = MatchDataProcessor(config=DataConfig())
processed_matches = processor.process_matches(match_data)
```

### Dataset

PyTorch dataset implementation for efficient data loading:

```python
from src.experiments.src.dl.data.dataset import create_dataloaders

train_loader, val_loader = create_dataloaders(
    processed_matches,
    batch_size=32,
    validation_split=0.1
)
```

## 🎯 Training

### Trainer

The `Trainer` class handles model training with features like:
- Early stopping
- Model checkpointing
- logging
- Validation metrics

```python
from src.experiments.src.dl.training.trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=TrainingConfig()
)

trainer.train()
```

## 📈 Evaluation

### Metrics

- Binary Cross-Entropy Loss
- Accuracy
- AUC-ROC

### Visualization

- Win probability plots

## 💾 Model Checkpoints

The model is saved automatically and can be loaded:

```python
# Save model
model.save("training/outputs/<run_dir>/model_checkpoint.pt")

# Load model
model = WinPredictor.load("training/outputs/<run_dir>/model_checkpoint.pt", config)
```

## 🔍 Model Performance

The current model achieves:
| Game Stage | Accuracy | AUC-ROC |
|------------|----------|---------|
| 1-10 min   | 73.4%    | 84.3    |
| 11-20 min  | 81.8%    | 92.1    |
| 21-30 min  | 90.8%    | 97.3    |
| 31-40 min  | 87.1%    | 96.4    |
| 41-50 min  | 83.5%    | 93.1    |
| 51-60 min  | 87.5%    | 86.7    |
- Average inference time: < 50ms

## ⚠️ Limitations

- Requires significant training data
- Sensitive to patch changes
- May need retraining for different skill brackets
- GPU recommended for training

## 📝 Notes

- Model performance varies by game phase
- Early game predictions are less reliable
- Feature engineering is crucial for model success
- Regular retraining recommended after major patches 