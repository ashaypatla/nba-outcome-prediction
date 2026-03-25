# NBA Game Outcome Prediction Model

A machine learning project that predicts NBA game outcomes by leveraging sportsbook-generated signals. The model produces win/loss predictions alongside calibrated win probabilities, benchmarked directly against sportsbook implied odds.

> **Course:** INFO-629 | **Author:** Ashay Patla

---

## Overview

With the rapid growth of sports betting, sportsbooks generate a large volume of historical odds data that serves as a powerful predictive signal. This project uses that data, specifically point spreads and game totals, to train a Logistic Regression model that predicts whether the home team wins a given NBA game.

A key goal was to explore whether a machine learning model can meaningfully outperform sportsbook implied probabilities, and to identify what additional features could close that gap.

---

## Repository Structure

```
nba-outcome-prediction/
│
├── notebooks/
│   └── NBA_outcome_prediction.html   # Full Jupyter notebook (HTML export)
│
├── data/
│   └── README.md                     # Data source instructions
│
├── models/
│   └── README.md                     # Model artifacts (add trained model here)
│
├── src/
│   └── predict.py                    # Standalone prediction function
│
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Dataset

**Source:** [NBA Odds Data — Kaggle](https://www.kaggle.com/datasets/christophertreasure/nba-odds-data?resource=download)

The dataset contains historical NBA game odds data including point spreads, game totals, and moneyline odds for both home and away teams across multiple seasons.

> Download the dataset from Kaggle and place the CSV file in the `data/` directory before running the notebook.

---

## Methodology

### Preprocessing

1. **Normalize to Home Team Perspective** — All game records were standardized from the home team's point of view to simplify training and ensure consistent feature representation.

2. **Season Phase Feature** — The season was divided into three phases to capture how sportsbook efficiency and team performance evolve over the year. As the season progresses, favorites win at a higher rate, validating season phase as a meaningful signal.

3. **Filter Seasons** — Only the most recent, complete seasons were retained for training. Incomplete seasons (2020 COVID year, 2023 partial dataset) were excluded to avoid noise.

4. **One-Hot Encoding** — Season phase was one-hot encoded into binary features (`season_phase_2`, `season_phase_3`) for model input.

### Input Features

| Feature | Description |
|---|---|
| `home_spread` | Point spread from the home team's perspective (positive = underdog, negative = favorite) |
| `total` | Over/under total points line set by sportsbooks |
| `season_phase_2` | Binary flag — mid-season |
| `season_phase_3` | Binary flag — late season |

**Target:** `home_win` (1 = home team won, 0 = home team lost)

### Model

**Logistic Regression** was selected because it:
- Natively models binary outcomes (win/loss)
- Produces calibrated win probabilities
- Supports multi-feature input
- Allows direct comparison against sportsbook implied probabilities

### Validation

**Walk-Forward Validation** was used across 4 training seasons to test model stability year-over-year. Results showed a consistent log loss of **0.59–0.62**, confirming the model generalizes reliably across seasons.

---

## Results

The final model was evaluated against the 2023–24 season (held-out test set) and compared to sportsbook implied probabilities:

| Metric | Model | Sportsbook Baseline |
|---|---|---|
| Log Loss | ~0.634 | ~0.639 |
| AUC | ~0.672 | ~0.668 |
| Brier Score | ~0.223 | ~0.223 |

The model showed **marginal improvement** over sportsbook predictions — reflecting how efficiently sportsbooks price game lines.

---

## Implementation

```python
def predict_home_win(spread, total, season_phase):
    """
    Predict the probability of the home team winning.

    Args:
        spread (float): Home team point spread (negative = favorite)
        total (float): Over/under total for the game
        season_phase (int): 1 = early season, 2 = mid-season, 3 = late season

    Returns:
        float: Probability of home team winning (0.0 – 1.0)
    """
    game = pd.DataFrame({
        "home_spread": [spread],
        "total": [total],
        "season_phase_2": [1 if season_phase == 2 else 0],
        "season_phase_3": [1 if season_phase == 3 else 0]
    })
    prob = model.predict_proba(game)[0, 1]
    print(f"Home Win Probability: {prob:.2%}")
    return prob
```

**Example — Celtics @ Spurs:**
```python
predict_home_win(spread=-3.5, total=221.5, season_phase=3)
# Home Win Probability: 61.80%
```
*(The Spurs, as the home team, won this game.)*

---

## Future Improvements

- **Include recent seasons** in training data to improve recency weighting
- **Add back-to-back game flags** to capture fatigue effects
- **Incorporate team performance metrics** (e.g., offensive/defensive rating, pace)
- **Train on playoff data** and include a `is_playoffs` binary feature
- **Experiment with ensemble models** (e.g., Gradient Boosting, XGBoost) for non-linear relationships

---

## Setup & Installation

### Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Notebook

1. Clone this repository
2. Download the dataset from Kaggle and place it in `data/`
3. Open `notebooks/NBA_outcome_prediction.html` in a browser to view the full analysis, or convert the original `.ipynb` to run interactively

---

## 📝 Conclusion

The model's predictions closely mirror sportsbook implied probabilities, with evaluation metrics showing modest but consistent improvement. This confirms that sportsbook odds are already highly efficient, yet room remains for targeted gains through richer feature engineering — particularly around team performance and game context.
