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