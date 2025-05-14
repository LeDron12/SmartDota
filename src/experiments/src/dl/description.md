Key temporal features we can use:

1. **Gold Advantage** (`radiant_gold_adv`): Array showing gold difference between Radiant and Dire teams over time
2. **XP Advantage** (`radiant_xp_adv`): Array showing experience difference between teams over time
3. **Teamfight Data** (`teamfights`): Detailed information about team fights including:
   - Damage dealt by each player
   - Healing done
   - Gold/XP gained
   - Deaths
   - Ability/item usage

4. **Objectives** (`objectives`): Important game events like:
   - First blood
   - Tower kills
   - Building kills

Here's my proposed baseline model architecture:

### Model Architecture: Temporal Win Probability Predictor

1. **Input Features** (per minute):
   - Gold advantage (normalized)
   - XP advantage (normalized)
   - Teamfight metrics (aggregated per minute):
     - Total damage dealt by each team
     - Total healing done by each team
     - Number of deaths per team
   - Objective status:
     - Number of towers remaining for each team
     - First blood status (binary)
     - Roshan kills (if any)

2. **Model Architecture**:
```
Input Layer (per minute features)
    ↓
LSTM Layer (64 units) - Captures temporal dependencies
    ↓
Dense Layer (32 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Output Layer (1 unit, Sigmoid) - Win probability
```

3. **Training Approach**:
   - Input: Sequence of game states from 0 to N minutes
   - Output: Binary win probability for Radiant team
   - Loss: Binary Cross Entropy
   - Optimizer: Adam
   - Validation: Use last 20% of matches for validation

4. **Data Preprocessing**:
   - Normalize gold and XP advantages
   - Create fixed-length sequences (pad shorter games)
   - Aggregate teamfight data into per-minute statistics
   - One-hot encode objective status

Here's a simple example of how the input data would be structured:

```python
# Example input format (per minute)
{
    'gold_advantage': float,  # Normalized gold difference
    'xp_advantage': float,    # Normalized XP difference
    'teamfight_metrics': {
        'radiant_damage': float,
        'dire_damage': float,
        'radiant_healing': float,
        'dire_healing': float,
        'radiant_deaths': int,
        'dire_deaths': int
    },
    'objectives': {
        'radiant_towers': int,  # Number of towers remaining
        'dire_towers': int,
        'first_blood': int,     # 0: none, 1: radiant, 2: dire
        'roshan_kills': int     # Number of Roshan kills
    }
}
```

### Why This Approach?

1. **Simplicity**: Uses readily available features without complex feature engineering
2. **Temporal Awareness**: LSTM layer captures game progression and momentum
3. **Interpretable**: Features directly relate to game state
4. **Scalable**: Can easily add more features or increase model complexity

### Implementation Steps:

1. Create a data pipeline to:
   - Extract per-minute features from match data
   - Normalize numerical features
   - Create fixed-length sequences
   - Generate labels (1 for Radiant win, 0 for Dire win)

2. Build the model using PyTorch or TensorFlow:
   - Implement the LSTM-based architecture
   - Add proper regularization (dropout)
   - Use appropriate loss function and optimizer

3. Training:
   - Use early stopping to prevent overfitting
   - Implement learning rate scheduling
   - Use appropriate batch size (32-64)

4. Evaluation:
   - Use ROC-AUC and accuracy metrics
   - Analyze prediction confidence over time
   - Validate on different skill brackets

This baseline model should provide a solid foundation for win probability prediction. Once this is working, we can explore more complex features like:
- Hero-specific metrics
- Item timings
- Player skill levels
- Draft information

Would you like me to elaborate on any part of this proposal or help with implementing a specific component?
