# Live Prediction Module

This module provides two types of predictions for Dota 2 matches:
1. **Draft/Picks Stage** (`predict.py`): Predicts win probability based on hero picks and player statistics
2. **Live Game Stage** (`predict_pro.py`): Predicts win probability during the match using real-time game state

## 🎯 Usage

### Draft/Picks Game Stage Predictions

To make predictions after heroes pick/bans stage:

⚠️ **Important**: Before running predictions, make sure to configure `predict_config.yaml` with your desired settings. The config file controls:
- `match_id`: Dota 2 match ID to predict
- `model_path`: Path to the trained CatBoost model (saved after training)
- `pipeline_path`: Path to the feature processing pipeline (saved after training)

**Config already contains correct model and pipeline paths**

#### Parameters

```bash
python3 predict.py --match_id=<MATCH_ID>
```

- `--match_id`: (Required) Dota 2 match ID to predict

#### How It Works

1. **Data Collection**:
   - Fetches match data from OpenDota API
   - Retrieves player statistics from Stratz API
   - Collects hero pick information

2. **Feature Processing**:
   - Processes hero picks using saved pipeline
   - Calculates hero win rates and synergies
   - Computes player performance metrics
   - Applies the trained feature pipeline

3. **Prediction**:
   - Uses the trained CatBoost model
   - Generates win probability for Radiant team
   - Provides confidence score

#### Model Performance


| Metric         | Value       |
|----------------|-------------|
| ✅ AUC-ROC     | 0.6054      |
| 🎯 F1 Score    | 0.6179      |
| 🎯 Precision   | 0.6441      |
| 🎯 Recall      | 0.5938      |
| 🎯 Accuracy    | 0.5937      |
| ❗ Brier Score | 0.2444      |

- Accuracy: ~60% on high MMR matches

#### Interpreting Results

The script outputs:
1. **Win Probability**: Single probability value for Radiant win
2. **Confidence Score**: Model's confidence in the prediction

### Live Game Stage Predictions

To make predictions during a live (pro) match:

```bash
python3 predict_pro.py --model_path=<PATH> --match_id=<MATCH_ID> [--show_plot]
```

#### Parameters

- `--model_path`: (Required) Path to the trained LSTM model checkpoint
- `--match_id`: (Required) Dota 2 match ID to predict
- `--show_plot`: (Optional) Display real-time prediction plot. Otherwise it would be save to current folder

#### How It Works

1. **Data Collection**:
   - Fetches live match data for every game minute
   - Collects real-time game state:
     - Gold and XP advantages
     - Objective status (towers, Roshan)
     - Teamfight metrics

2. **Feature Processing**:
   - Normalizes game state features
   - Creates time series sequences
   - Handles missing data points

3. **Prediction**:
   - Uses LSTM model
   - Processes temporal game state
   - Outputs prediction plot for every game minute

#### Model Performance

#### Model Performance by Game Stage

| Game Stage | Accuracy | AUC-ROC |
|------------|----------|---------|
| 1-10 min   | 73.4%    | 84.3    |
| 11-20 min  | 81.8%    | 92.1    |
| 21-30 min  | 90.8%    | 97.3    |
| 31-40 min  | 87.1%    | 96.4    |
| 41-50 min  | 83.5%    | 93.1    |
| 51-60 min  | 87.5%    | 86.7    |

The model shows strong predictive power across all game stages, with peak performance during the mid-game (21-30 minutes) where accuracy reaches 90.8% and AUC-ROC hits 97.3.
Performance remains consistently high through the late game, though with slightly lower metrics in the very late stages (51-60 minutes).


- Early Game (0-15 min): ~75% accuracy
- Mid Game (15-35 min): ~90% accuracy
- Late Game (35+ min): ~85% accuracy
- Features used:
  - Gold advantage
  - XP advantage
  - Objective control
  - Teamfight metrics

#### Interpreting Results

The script generates:
1. **Real-time Plot**:
   - Blue line: Win probability over time
   - Red dashed line: Decision boundary (0.5)
   - Green dotted line: True outcome (if available)
   - X-axis: Game time in minutes
   - Y-axis: Radiant win probability

2. **Output File**:
   - Saved as `predictions_<match_id>_<timestamp>.png`
   - Contains full prediction timeline

## 📊 Finding Matches

- High MMR matches: https://www.opendota.com/matches/highMmr
- Pro matches: https://www.opendota.com/matches/pro

## ⚠️ Limitations

### Draft/Picks Predictions
- Less reliable for new heroes/patches
- Player statistics may be incomplete
- Team strategies not considered

### Live Predictions
- Early game predictions less reliable
- Requires stable internet connection
- May miss very quick game events
- Performance varies by patch version

## 📝 Notes

- Both models are trained on high MMR/pro matches
- Regular updates recommended after major patches
- Predictions are more reliable for standard game modes
- Consider using both models for different insights

