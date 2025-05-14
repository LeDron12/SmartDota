# Experiments Module

This module provides tools for training, evaluating, and analyzing Dota 2 match prediction models. It includes both traditional machine learning and deep learning approaches.

## 🎯 Features

- Model training and evaluation
- Time-based cross-validation
- Feature engineering pipeline
- Model comparison and analysis
- Experiment tracking and logging

## 🛠️ Usage

### Training Models

```bash
# Train a model using configuration file
python __main__.py --config configs/train_config.yaml

# Run time-based cross-validation
python time_cv.py --config configs/cv_config.yaml --n-windows 5 --window-hours 24
```

### Configuration

Create a `train_config.yaml` file:
**Examples in configs/**

Example:
```yaml
dataset:
  type: "starz_public_matches"  # Type of dataset to load
  path: "../data_new/fetched_datasets/stratz_matches.json"

transformers:
  - name: StarzDatasetConverter
    enabled: true
    description: "Converts raw match data to DataFrame format"

  - name: HeroFeaturesTransformer
    enabled: false
    heroes_path: data/heroes.json
    description: "Transforms hero picks into one-hot encoded features"

  - name: HeroStatsTransformer
    enabled: true
    heroes_path: "data/hero_stats.json"
    use_extra_features: false
    description: "Calculates hero statistics features like winrates, pickrates, and banrates"

  - name: PlayerStatsTransformer
    enabled: true
    use_diff: false
    description: "Calculates player statistics features"

model:
  do_shap: true
  use_param_grid: false
  use_scaling: false

  # Base model parameters
  base_params:
    LogisticRegression:
      max_iter: 1000
      random_state: 42
    
    RandomForest:
      n_estimators: 100
      random_state: 42
    
    CatBoost:
      iterations: 2000
      learning_rate: 0.03
      depth: 5
      loss_function: "Logloss"
      eval_metric: "Accuracy"
      random_seed: 42
```

## 📊 Data Processing Pipeline

### Transformers

The module includes several transformers for feature engineering:

1. `HeroFeaturesTransformer`:
   - Hero win rates
   - Pick rates
   - Hero synergies

2. `PlayerStatsTransformer`:
   - Player performance metrics
   - Historical win rates
   - Role-specific statistics

3. `PlayerHeroStatsTransformer`:
   - Hero-specific player performance
   - Hero mastery metrics

4. `PlayerRatingsTransformer`:
   - Estimated Player skill ratings
   - Estimated Role-based ratings

### Usage

Training implements factory patters. You can add your Data Transformer to transformers/ and make expriment

## 📈 Model Training

### Supported Models

1. **CatBoost**
   - Best performance for tabular data
   - Handles categorical features well
   - Good for feature importance analysis

2. **Random Forest**
   - Robust to overfitting
   - Good for feature importance
   - Fast training and inference

3. **Logistic Regression**
   - Simple and interpretable
   - Good baseline model
   - Fast training and inference

### Training Process

Just run a script with specified config

## 📊 Evaluation

### Metrics

- Accuracy
- AUC-ROC
- Precision/Recall
- F1 Score
- Feature Importance
- SHAP Values

### Visualization

- Learning curves
- Feature importance plots
- SHAP summary plots
- Metrics over thresholds plots

## 📝 Experiment Tracking

Each experiment run is saved with:
- Model checkpoints
- Configuration files
- Training metrics
- Validation metrics
- Feature importance analysis
- SHAP analysis (if enabled)

## 🔍 Model Performance

Current best models achieve:
- CatBoost: ~60 accuracy
- Random Forest: ~55% accuracy
- Logistic Regression: ~52% accuracy

## ⚠️ Limitations

- Training data quality affects performance
- Models may need retraining after patches
- Feature engineering is crucial
- Some features require API access