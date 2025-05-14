# SmartDota

SmartDota is a machine learning project for Dota 2 match analysis and prediction. It provides tools for training models on historical match data and making real-time predictions during live matches.

## 🎮 Features

- **Match Data Collection**: Fetch and process match data from OpenDota and Stratz APIs
- **Win Probability Prediction**: Real-time win probability prediction during live matches
- **Model Training**: Train custom models on historical match data
- **Feature Engineering**: Comprehensive feature extraction for heroes, players, and team statistics
- **Model Evaluation**: Time-based cross-validation and performance metrics

## 📋 Prerequisites

- Python 3.10
- CUDA-capable GPU (recommended for training)
- Conda package manager

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SmartDota.git
cd SmartDota
```

2. Set up the environment:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

3. Activate the environment:
```bash
conda activate smartdota
```

## 📁 Project Structure

```
SmartDota/
├── src/
│   ├── data_new/           # Data collection and processing
│   │   └── README.md       # Data fetching and preprocessing guide
│   ├── experiments/        # Model training and evaluation
│   │   ├── src/
│   │   │   ├── dl/        # Deep learning models
│   │   │   │   └── README.md
│   │   │   └── README.md  # Training scripts guide
│   │   └── runs/          # Training runs and checkpoints
│   └── live/              # Live prediction scripts
│       └── README.md      # Prediction usage guide
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
└── setup_env.sh          # Environment setup script
```

## 🎯 Usage

### Draft/Picks Game Stage Predictions

To make predictions after heroes pick/bans stage:

⚠️ **Important**: Before running predictions, make sure to configure `predict_config.yaml` with your desired settings. The config file controls:
- match_id - which one to predict
- model_path - saved automatically after model training experiment
- pipeline_path - saved automatically after model training experiment
**Config already contains correct model and pipeline paths**


Example config:

```bash
cd src/live/
echo "Contents of predict_config.yaml:"
cat predict_config.yaml

# Template
python3 predict.py --match_id=<MATCH_ID>
# Sample
python3 predict.py --match_id=8290631276
```

List of latest public matches: https://www.opendota.com/matches/highMmr

### Live Game Stage Predictions

To make predictions during a live (pro) match:

```bash
cd src/live/

# Template
python3 predict_pro.py \
--model_path=../experiments/src/dl/training/output/win_predictor_2025.05.13_22.35.13[xp_gold_objectives_teamfights]/final_model.pt \
--match_id=<MATCH_ID>
# Sample 
python3 predict_pro.py \
--model_path=/Users/ankamenskiy/SmartDota/src/experiments/src/dl/training/output/win_predictor_2025.05.13_22.14.46/best_model.pt \
--match_id=8290631276
```

List of latest pro matches: https://www.opendota.com/matches/pro

### Model Training

To train a new model:

```bash
cd src/experiments/src/

# Template
python3 __main__.py --config=../configs/<CONFIG>.yaml
# Sample [Sample configs are listed in ../configs/]
python3 __main__.py --config=../hero_ohe_3.78c_hero-stats_player-stats.yaml
```

For time-based cross-validation (better precision):

```bash
cd src/experiments/src/

# Template
python3 time_cv.py \
--config=../configs/<CONFIG>.yaml \
--n-windows=3 \
--window-hours=6
# Sample [Sample configs are listed in ../configs/]
python3 time_cv.py \
--config=../configs/hero_ohe_3.78c_hero-stats_player-stats.yaml \
--n-windows=3 \
--window-hours=6
```

## 📊 Pretrained Models

The repository includes two pretrained model checkpoints:

1. `src/experiments/src/dl/training/output/win_predictor_2025.05.13_22.35.13[xp_gold_objectives_teamfights]/final_model.pt`: Best performing model for [live] predictions
2. `/Users/ankamenskiy/SmartDota/src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21`: Best model (time-based cross-validation) for [After picks] predictionsfrom

## 📝 Documentation

Detailed documentation for each component:

- [Data Collection Guide](src/data_new/README.md)
- [Draft Stage Training Guide](src/experiments/src/README.md)
- [Live Stage Training Guide](src/experiments/src/dl/README.md)
- [Predictions Guide](src/live/README.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenDota API](https://docs.opendota.com/) for match data
- [Stratz API](https://stratz.com/api) for additional match statistics
- The Dota 2 community for inspiration and feedback

---

# Summary

Date : 2025-05-14 17:06:21

Directory /Users/ankamenskiy/SmartDota

Total : 534 files,  2103128 codes, 1759 comments, 2090 blanks, all 2106977 lines

Summary / [Details](details.md) / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Languages
| language | files | code | comment | blank | total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| JSON | 327 | 1,978,749 | 0 | 24 | 1,978,773 |
| Log | 22 | 112,435 | 0 | 133 | 112,568 |
| Python | 54 | 5,945 | 1,687 | 1,486 | 9,118 |
| XML | 3 | 3,132 | 0 | 13 | 3,145 |
| YAML | 28 | 1,298 | 49 | 86 | 1,433 |
| Markdown | 9 | 723 | 0 | 242 | 965 |
| Excel | 4 | 376 | 0 | 4 | 380 |
| CSV | 79 | 310 | 0 | 79 | 389 |
| MS SQL | 5 | 79 | 0 | 0 | 79 |
| Shell Script | 1 | 56 | 17 | 18 | 91 |
| pip requirements | 2 | 25 | 6 | 5 | 36 |

## Directories
| path | files | code | comment | blank | total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| . | 534 | 2,103,128 | 1,759 | 2,090 | 2,106,977 |
| . (Files) | 4 | 203 | 23 | 69 | 295 |
| data | 3 | 14,233 | 0 | 1 | 14,234 |
| sandbox | 36 | 50,414 | 1,150 | 518 | 52,082 |
| sandbox/data | 21 | 7,981 | 1,043 | 293 | 9,317 |
| sandbox/data/api | 8 | 7,640 | 47 | 219 | 7,906 |
| sandbox/data/api (Files) | 3 | 352 | 2 | 79 | 433 |
| sandbox/data/api/OpenDota | 3 | 479 | 45 | 139 | 663 |
| sandbox/data/api/SteamWebAPI | 2 | 6,809 | 0 | 1 | 6,810 |
| sandbox/data/dataclasses | 8 | 184 | 972 | 38 | 1,194 |
| sandbox/data/dataset | 5 | 157 | 24 | 36 | 217 |
| sandbox/data/dataset (Files) | 3 | 77 | 24 | 30 | 131 |
| sandbox/data/dataset/configs | 2 | 80 | 0 | 6 | 86 |
| sandbox/ids_only | 1 | 3,123 | 0 | 1 | 3,124 |
| sandbox/ids_to_hero-stats | 2 | 35,809 | 0 | 1 | 35,810 |
| sandbox/ids_w_teams | 1 | 0 | 0 | 1 | 1 |
| sandbox/lib | 2 | 99 | 0 | 34 | 133 |
| sandbox/sandbox | 3 | 2,634 | 0 | 8 | 2,642 |
| sandbox/test_exps | 6 | 768 | 107 | 180 | 1,055 |
| sandbox/test_exps/exp1 | 2 | 285 | 41 | 68 | 394 |
| sandbox/test_exps/exp2 | 4 | 483 | 66 | 112 | 661 |
| src | 491 | 2,038,278 | 586 | 1,502 | 2,040,366 |
| src (Files) | 1 | 3 | 0 | 0 | 3 |
| src/data_new | 24 | 1,955,502 | 45 | 108 | 1,955,655 |
| src/data_new (Files) | 7 | 91,744 | 36 | 94 | 91,874 |
| src/data_new/constants | 4 | 15,682 | 0 | 0 | 15,682 |
| src/data_new/fetched_datasets | 5 | 1,819,311 | 0 | 1 | 1,819,312 |
| src/data_new/sandbox | 2 | 28,705 | 9 | 13 | 28,727 |
| src/data_new/sql | 6 | 60 | 0 | 0 | 60 |
| src/experiments | 461 | 82,378 | 504 | 1,270 | 84,152 |
| src/experiments/configs | 7 | 390 | 43 | 62 | 495 |
| src/experiments/runs | 412 | 66,127 | 0 | 231 | 66,358 |
| src/experiments/runs/hero_features_config | 6 | 449 | 0 | 4 | 453 |
| src/experiments/runs/hero_features_config/2025.05.04-19.55.59 | 6 | 449 | 0 | 4 | 453 |
| src/experiments/runs/hero_features_config/2025.05.04-19.55.59 (Files) | 4 | 418 | 0 | 3 | 421 |
| src/experiments/runs/hero_features_config/2025.05.04-19.55.59/models | 2 | 31 | 0 | 1 | 32 |
| src/experiments/runs/hero_ohe_3.78c | 12 | 819 | 0 | 10 | 829 |
| src/experiments/runs/hero_ohe_3.78c/2025.05.06-14.52.45 | 6 | 357 | 0 | 4 | 361 |
| src/experiments/runs/hero_ohe_3.78c/2025.05.06-14.52.45 (Files) | 4 | 321 | 0 | 3 | 324 |
| src/experiments/runs/hero_ohe_3.78c/2025.05.06-14.52.45/models | 2 | 36 | 0 | 1 | 37 |
| src/experiments/runs/hero_ohe_3.78c/2025.05.10-16.04.47 | 6 | 462 | 0 | 6 | 468 |
| src/experiments/runs/hero_ohe_3.78c/2025.05.10-16.04.47 (Files) | 4 | 421 | 0 | 5 | 426 |
| src/experiments/runs/hero_ohe_3.78c/2025.05.10-16.04.47/models | 2 | 41 | 0 | 1 | 42 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats | 233 | 25,458 | 0 | 100 | 25,558 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.51.20_OHE[0.554] | 16 | 3,211 | 0 | 20 | 3,231 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.51.20_OHE[0.554] (Files) | 4 | 3,181 | 0 | 14 | 3,195 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.51.20_OHE[0.554]/window_1 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.51.20_OHE[0.554]/window_2 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.51.20_OHE[0.554]/window_3 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.51.20_OHE[0.554]/window_4 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.51.20_OHE[0.554]/window_5 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.51.20_OHE[0.554]/window_6 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.56.37_noOHE[0.565] | 16 | 1,264 | 0 | 8 | 1,272 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.56.37_noOHE[0.565] (Files) | 4 | 1,234 | 0 | 2 | 1,236 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.56.37_noOHE[0.565]/window_1 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.56.37_noOHE[0.565]/window_2 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.56.37_noOHE[0.565]/window_3 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.56.37_noOHE[0.565]/window_4 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.56.37_noOHE[0.565]/window_5 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-14.56.37_noOHE[0.565]/window_6 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561] | 28 | 5,995 | 0 | 20 | 6,015 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561] (Files) | 4 | 1,429 | 0 | 14 | 1,443 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_1 | 4 | 761 | 0 | 1 | 762 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_1 (Files) | 2 | 33 | 0 | 1 | 34 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_1/plots | 2 | 728 | 0 | 0 | 728 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_1/plots/CatBoost | 1 | 364 | 0 | 0 | 364 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_1/plots/LogisticRegression | 1 | 364 | 0 | 0 | 364 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_2 | 4 | 761 | 0 | 1 | 762 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_2 (Files) | 2 | 33 | 0 | 1 | 34 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_2/plots | 2 | 728 | 0 | 0 | 728 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_2/plots/CatBoost | 1 | 364 | 0 | 0 | 364 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_2/plots/LogisticRegression | 1 | 364 | 0 | 0 | 364 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_3 | 4 | 761 | 0 | 1 | 762 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_3 (Files) | 2 | 33 | 0 | 1 | 34 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_3/plots | 2 | 728 | 0 | 0 | 728 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_3/plots/CatBoost | 1 | 364 | 0 | 0 | 364 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_3/plots/LogisticRegression | 1 | 364 | 0 | 0 | 364 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_4 | 4 | 761 | 0 | 1 | 762 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_4 (Files) | 2 | 33 | 0 | 1 | 34 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_4/plots | 2 | 728 | 0 | 0 | 728 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_4/plots/CatBoost | 1 | 364 | 0 | 0 | 364 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_4/plots/LogisticRegression | 1 | 364 | 0 | 0 | 364 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_5 | 4 | 761 | 0 | 1 | 762 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_5 (Files) | 2 | 33 | 0 | 1 | 34 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_5/plots | 2 | 728 | 0 | 0 | 728 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_5/plots/CatBoost | 1 | 364 | 0 | 0 | 364 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_5/plots/LogisticRegression | 1 | 364 | 0 | 0 | 364 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_6 | 4 | 761 | 0 | 1 | 762 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_6 (Files) | 2 | 33 | 0 | 1 | 34 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_6/plots | 2 | 728 | 0 | 0 | 728 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_6/plots/CatBoost | 1 | 364 | 0 | 0 | 364 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.09.04_noOHE_extraFeat[0.561]/window_6/plots/LogisticRegression | 1 | 364 | 0 | 0 | 364 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560] | 34 | 3,001 | 0 | 8 | 3,009 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560] (Files) | 4 | 1,315 | 0 | 2 | 1,317 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_1 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_1 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_1/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_1/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_1/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_1/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_2 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_2 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_2/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_2/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_2/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_2/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_3 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_3 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_3/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_3/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_3/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_3/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_4 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_4 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_4/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_4/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_4/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_4/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_5 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_5 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_5/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_5/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_5/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_5/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_6 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_6 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_6/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_6/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_6/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.43.59_noOHE_extraFeat_standScale[0.560]/window_6/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560] | 34 | 3,012 | 0 | 8 | 3,020 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560] (Files) | 4 | 1,326 | 0 | 2 | 1,328 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_1 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_1 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_1/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_1/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_1/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_1/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_2 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_2 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_2/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_2/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_2/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_2/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_3 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_3 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_3/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_3/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_3/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_3/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_4 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_4 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_4/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_4/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_4/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_4/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_5 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_5 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_5/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_5/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_5/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_5/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_6 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_6 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_6/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_6/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_6/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-15.47.09_noOHE_standScale[0.560]/window_6/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565] | 34 | 3,010 | 0 | 8 | 3,018 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565] (Files) | 4 | 1,324 | 0 | 2 | 1,326 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_1 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_1 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_1/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_1/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_1/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_1/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_2 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_2 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_2/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_2/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_2/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_2/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_3 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_3 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_3/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_3/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_3/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_3/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_4 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_4 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_4/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_4/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_4/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_4/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_5 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_5 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_5/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_5/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_5/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_5/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_6 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_6 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_6/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_6/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_6/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.00.15_noOHE[0.565]/window_6/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547] | 34 | 2,564 | 0 | 8 | 2,572 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547] (Files) | 4 | 1,310 | 0 | 2 | 1,312 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_1 | 5 | 209 | 0 | 1 | 210 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_1 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_1/plots | 3 | 204 | 0 | 0 | 204 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_1/plots/CatBoost | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_1/plots/LogisticRegression | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_1/plots/RandomForest | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_2 | 5 | 209 | 0 | 1 | 210 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_2 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_2/plots | 3 | 204 | 0 | 0 | 204 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_2/plots/CatBoost | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_2/plots/LogisticRegression | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_2/plots/RandomForest | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_3 | 5 | 209 | 0 | 1 | 210 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_3 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_3/plots | 3 | 204 | 0 | 0 | 204 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_3/plots/CatBoost | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_3/plots/LogisticRegression | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_3/plots/RandomForest | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_4 | 5 | 209 | 0 | 1 | 210 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_4 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_4/plots | 3 | 204 | 0 | 0 | 204 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_4/plots/CatBoost | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_4/plots/LogisticRegression | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_4/plots/RandomForest | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_5 | 5 | 209 | 0 | 1 | 210 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_5 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_5/plots | 3 | 204 | 0 | 0 | 204 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_5/plots/CatBoost | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_5/plots/LogisticRegression | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_5/plots/RandomForest | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_6 | 5 | 209 | 0 | 1 | 210 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_6 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_6/plots | 3 | 204 | 0 | 0 | 204 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_6/plots/CatBoost | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_6/plots/LogisticRegression | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-16.09.43_noOHE_exclPickrates[0.547]/window_6/plots/RandomForest | 1 | 68 | 0 | 0 | 68 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-17.13.36_noOHE_lr-0.3_dpth-5[0.570] | 6 | 509 | 0 | 4 | 513 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-17.13.36_noOHE_lr-0.3_dpth-5[0.570] (Files) | 4 | 468 | 0 | 3 | 471 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.11-17.13.36_noOHE_lr-0.3_dpth-5[0.570]/models | 2 | 41 | 0 | 1 | 42 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568] | 20 | 1,763 | 0 | 5 | 1,768 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568] (Files) | 4 | 904 | 0 | 2 | 906 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/pipeline | 1 | 16 | 0 | 0 | 16 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_1 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_1 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_1/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_1/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_1/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_1/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_2 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_2 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_2/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_2/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_2/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_2/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_3 | 5 | 281 | 0 | 1 | 282 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_3 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_3/plots | 3 | 276 | 0 | 0 | 276 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_3/plots/CatBoost | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_3/plots/LogisticRegression | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-19.54.04_noOHE_rank20-40[0.568]/window_3/plots/RandomForest | 1 | 92 | 0 | 0 | 92 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-20.05.41_OHE_rank20-40[0.575] | 11 | 1,129 | 0 | 11 | 1,140 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-20.05.41_OHE_rank20-40[0.575] (Files) | 4 | 1,092 | 0 | 8 | 1,100 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-20.05.41_OHE_rank20-40[0.575]/pipeline | 1 | 22 | 0 | 0 | 22 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-20.05.41_OHE_rank20-40[0.575]/window_1 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-20.05.41_OHE_rank20-40[0.575]/window_2 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats/2025.05.12-20.05.41_OHE_rank20-40[0.575]/window_3 | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats | 74 | 21,688 | 0 | 48 | 21,736 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581] | 34 | 5,932 | 0 | 20 | 5,952 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581] (Files) | 4 | 1,690 | 0 | 14 | 1,704 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_1 | 5 | 737 | 0 | 1 | 738 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_1 (Files) | 2 | 41 | 0 | 1 | 42 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_1/plots | 3 | 696 | 0 | 0 | 696 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_1/plots/CatBoost | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_1/plots/LogisticRegression | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_1/plots/RandomForest | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_2 | 5 | 701 | 0 | 1 | 702 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_2 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_2/plots | 3 | 696 | 0 | 0 | 696 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_2/plots/CatBoost | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_2/plots/LogisticRegression | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_2/plots/RandomForest | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_3 | 5 | 701 | 0 | 1 | 702 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_3 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_3/plots | 3 | 696 | 0 | 0 | 696 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_3/plots/CatBoost | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_3/plots/LogisticRegression | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_3/plots/RandomForest | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_4 | 5 | 701 | 0 | 1 | 702 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_4 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_4/plots | 3 | 696 | 0 | 0 | 696 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_4/plots/CatBoost | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_4/plots/LogisticRegression | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_4/plots/RandomForest | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_5 | 5 | 701 | 0 | 1 | 702 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_5 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_5/plots | 3 | 696 | 0 | 0 | 696 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_5/plots/CatBoost | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_5/plots/LogisticRegression | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_5/plots/RandomForest | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_6 | 5 | 701 | 0 | 1 | 702 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_6 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_6/plots | 3 | 696 | 0 | 0 | 696 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_6/plots/CatBoost | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_6/plots/LogisticRegression | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.11-17.16.26_noOHE[0.581]/window_6/plots/RandomForest | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572] | 20 | 12,504 | 0 | 17 | 12,521 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572] (Files) | 4 | 1,301 | 0 | 14 | 1,315 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/pipeline | 1 | 28 | 0 | 0 | 28 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_1 | 5 | 3,725 | 0 | 1 | 3,726 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_1 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_1/plots | 3 | 3,720 | 0 | 0 | 3,720 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_1/plots/CatBoost | 1 | 1,240 | 0 | 0 | 1,240 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_1/plots/LogisticRegression | 1 | 1,240 | 0 | 0 | 1,240 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_1/plots/RandomForest | 1 | 1,240 | 0 | 0 | 1,240 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_2 | 5 | 3,725 | 0 | 1 | 3,726 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_2 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_2/plots | 3 | 3,720 | 0 | 0 | 3,720 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_2/plots/CatBoost | 1 | 1,240 | 0 | 0 | 1,240 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_2/plots/LogisticRegression | 1 | 1,240 | 0 | 0 | 1,240 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_2/plots/RandomForest | 1 | 1,240 | 0 | 0 | 1,240 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_3 | 5 | 3,725 | 0 | 1 | 3,726 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_3 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_3/plots | 3 | 3,720 | 0 | 0 | 3,720 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_3/plots/CatBoost | 1 | 1,240 | 0 | 0 | 1,240 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_3/plots/LogisticRegression | 1 | 1,240 | 0 | 0 | 1,240 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.12-20.10.54_OHE[0.572]/window_3/plots/RandomForest | 1 | 1,240 | 0 | 0 | 1,240 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21 | 20 | 3,252 | 0 | 11 | 3,263 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21 (Files) | 4 | 1,091 | 0 | 8 | 1,099 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/pipeline | 1 | 22 | 0 | 0 | 22 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_1 | 5 | 737 | 0 | 1 | 738 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_1 (Files) | 2 | 41 | 0 | 1 | 42 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_1/plots | 3 | 696 | 0 | 0 | 696 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_1/plots/CatBoost | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_1/plots/LogisticRegression | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_1/plots/RandomForest | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_2 | 5 | 701 | 0 | 1 | 702 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_2 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_2/plots | 3 | 696 | 0 | 0 | 696 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_2/plots/CatBoost | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_2/plots/LogisticRegression | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_2/plots/RandomForest | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_3 | 5 | 701 | 0 | 1 | 702 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_3 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_3/plots | 3 | 696 | 0 | 0 | 696 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_3/plots/CatBoost | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_3/plots/LogisticRegression | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats/2025.05.14-14.44.21/window_3/plots/RandomForest | 1 | 232 | 0 | 0 | 232 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats | 34 | 7,075 | 0 | 32 | 7,107 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581] | 34 | 7,075 | 0 | 32 | 7,107 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581] (Files) | 4 | 1,969 | 0 | 26 | 1,995 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_1 | 5 | 881 | 0 | 1 | 882 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_1 (Files) | 2 | 41 | 0 | 1 | 42 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_1/plots | 3 | 840 | 0 | 0 | 840 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_1/plots/CatBoost | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_1/plots/LogisticRegression | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_1/plots/RandomForest | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_2 | 5 | 845 | 0 | 1 | 846 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_2 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_2/plots | 3 | 840 | 0 | 0 | 840 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_2/plots/CatBoost | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_2/plots/LogisticRegression | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_2/plots/RandomForest | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_3 | 5 | 845 | 0 | 1 | 846 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_3 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_3/plots | 3 | 840 | 0 | 0 | 840 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_3/plots/CatBoost | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_3/plots/LogisticRegression | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_3/plots/RandomForest | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_4 | 5 | 845 | 0 | 1 | 846 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_4 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_4/plots | 3 | 840 | 0 | 0 | 840 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_4/plots/CatBoost | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_4/plots/LogisticRegression | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_4/plots/RandomForest | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_5 | 5 | 845 | 0 | 1 | 846 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_5 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_5/plots | 3 | 840 | 0 | 0 | 840 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_5/plots/CatBoost | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_5/plots/LogisticRegression | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_5/plots/RandomForest | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_6 | 5 | 845 | 0 | 1 | 846 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_6 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_6/plots | 3 | 840 | 0 | 0 | 840 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_6/plots/CatBoost | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_6/plots/LogisticRegression | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats/2025.05.11-18.43.18_noOHE[0.581]/window_6/plots/RandomForest | 1 | 280 | 0 | 0 | 280 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings | 19 | 3,960 | 0 | 17 | 3,977 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528] | 19 | 3,960 | 0 | 17 | 3,977 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528] (Files) | 4 | 1,209 | 0 | 14 | 1,223 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_1 | 5 | 917 | 0 | 1 | 918 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_1 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_1/plots | 3 | 912 | 0 | 0 | 912 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_1/plots/CatBoost | 1 | 304 | 0 | 0 | 304 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_1/plots/LogisticRegression | 1 | 304 | 0 | 0 | 304 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_1/plots/RandomForest | 1 | 304 | 0 | 0 | 304 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_2 | 5 | 917 | 0 | 1 | 918 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_2 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_2/plots | 3 | 912 | 0 | 0 | 912 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_2/plots/CatBoost | 1 | 304 | 0 | 0 | 304 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_2/plots/LogisticRegression | 1 | 304 | 0 | 0 | 304 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_2/plots/RandomForest | 1 | 304 | 0 | 0 | 304 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_3 | 5 | 917 | 0 | 1 | 918 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_3 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_3/plots | 3 | 912 | 0 | 0 | 912 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_3/plots/CatBoost | 1 | 304 | 0 | 0 | 304 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_3/plots/LogisticRegression | 1 | 304 | 0 | 0 | 304 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_playerHero-stats_ratings/2025.05.11-19.40.51_noOHE[0.528]/window_3/plots/RandomForest | 1 | 304 | 0 | 0 | 304 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings | 34 | 6,678 | 0 | 20 | 6,698 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557] | 34 | 6,678 | 0 | 20 | 6,698 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557] (Files) | 4 | 1,932 | 0 | 14 | 1,946 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_1 | 5 | 809 | 0 | 1 | 810 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_1 (Files) | 2 | 41 | 0 | 1 | 42 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_1/plots | 3 | 768 | 0 | 0 | 768 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_1/plots/CatBoost | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_1/plots/LogisticRegression | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_1/plots/RandomForest | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_2 | 5 | 809 | 0 | 1 | 810 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_2 (Files) | 2 | 41 | 0 | 1 | 42 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_2/plots | 3 | 768 | 0 | 0 | 768 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_2/plots/CatBoost | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_2/plots/LogisticRegression | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_2/plots/RandomForest | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_3 | 5 | 809 | 0 | 1 | 810 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_3 (Files) | 2 | 41 | 0 | 1 | 42 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_3/plots | 3 | 768 | 0 | 0 | 768 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_3/plots/CatBoost | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_3/plots/LogisticRegression | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_3/plots/RandomForest | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_4 | 5 | 773 | 0 | 1 | 774 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_4 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_4/plots | 3 | 768 | 0 | 0 | 768 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_4/plots/CatBoost | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_4/plots/LogisticRegression | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_4/plots/RandomForest | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_5 | 5 | 773 | 0 | 1 | 774 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_5 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_5/plots | 3 | 768 | 0 | 0 | 768 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_5/plots/CatBoost | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_5/plots/LogisticRegression | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_5/plots/RandomForest | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_6 | 5 | 773 | 0 | 1 | 774 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_6 (Files) | 2 | 5 | 0 | 1 | 6 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_6/plots | 3 | 768 | 0 | 0 | 768 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_6/plots/CatBoost | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_6/plots/LogisticRegression | 1 | 256 | 0 | 0 | 256 |
| src/experiments/runs/hero_ohe_3.78c_hero-stats_player-stats_ratings/2025.05.11-18.19.41_noOHE[0.557]/window_6/plots/RandomForest | 1 | 256 | 0 | 0 | 256 |
| src/experiments/src | 42 | 15,861 | 461 | 977 | 17,299 |
| src/experiments/src (Files) | 4 | 556 | 82 | 175 | 813 |
| src/experiments/src/core | 4 | 687 | 72 | 168 | 927 |
| src/experiments/src/dl | 22 | 12,670 | 91 | 271 | 13,032 |
| src/experiments/src/dl (Files) | 6 | 399 | 18 | 100 | 517 |
| src/experiments/src/dl/data | 4 | 8,179 | 28 | 65 | 8,272 |
| src/experiments/src/dl/models | 1 | 69 | 6 | 16 | 91 |
| src/experiments/src/dl/training | 11 | 4,023 | 39 | 90 | 4,152 |
| src/experiments/src/dl/training (Files) | 2 | 337 | 39 | 77 | 453 |
| src/experiments/src/dl/training/output | 9 | 3,686 | 0 | 13 | 3,699 |
| src/experiments/src/dl/training/output/win_predictor_2025.05.13_22.35.13[xp_gold_objectives_teamfights] | 3 | 1,350 | 0 | 4 | 1,354 |
| src/experiments/src/dl/training/output/win_predictor_2025.05.13_23.16.27[xp_gold_objectives] | 3 | 1,202 | 0 | 8 | 1,210 |
| src/experiments/src/dl/training/output/win_predictor_2025.05.13_23.21.04[xp_gold] | 3 | 1,134 | 0 | 1 | 1,135 |
| src/experiments/src/transformers | 9 | 1,629 | 165 | 290 | 2,084 |
| src/experiments/src/utils | 3 | 319 | 51 | 73 | 443 |
| src/live | 5 | 395 | 37 | 124 | 556 |

Summary / [Details](details.md) / [Diff Summary](diff.md) / [Diff Details](diff-details.md)