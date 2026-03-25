"""
predict.py
----------
Standalone prediction utility for the NBA Game Outcome Prediction model.

Usage:
    from src.predict import predict_home_win
    prob = predict_home_win(spread=-3.5, total=221.5, season_phase=3)
"""

import pandas as pd
import pickle
import os


def load_model(model_path: str = "models/nba_lr_model.pkl"):
    """Load the trained Logistic Regression model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Please train the model from the notebook first and save it."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict_home_win(
    spread: float,
    total: float,
    season_phase: int,
    model=None,
    model_path: str = "models/nba_lr_model.pkl",
    verbose: bool = True,
) -> float:
    """
    Predict the probability of the home team winning an NBA game.

    Parameters
    ----------
    spread : float
        Home team point spread set by sportsbooks.
        Negative values indicate the home team is favored.
        Positive values indicate the home team is the underdog.
    total : float
        Over/under total points line for the game.
    season_phase : int
        Phase of the NBA season:
            1 = Early season  (first ~27 games)
            2 = Mid season    (games 28–55)
            3 = Late season   (final ~27 games)
    model : sklearn estimator, optional
        A pre-loaded model object. If None, loads from `model_path`.
    model_path : str
        Path to the saved model pickle file.
    verbose : bool
        Whether to print the predicted probability.

    Returns
    -------
    float
        Probability of the home team winning (between 0.0 and 1.0).

    Examples
    --------
    >>> prob = predict_home_win(spread=-3.5, total=221.5, season_phase=3)
    Home Win Probability: 61.80%
    """
    if model is None:
        model = load_model(model_path)

    if season_phase not in (1, 2, 3):
        raise ValueError("season_phase must be 1, 2, or 3.")

    game = pd.DataFrame({
        "home_spread": [spread],
        "total": [total],
        "season_phase_2": [1 if season_phase == 2 else 0],
        "season_phase_3": [1 if season_phase == 3 else 0],
    })

    prob = model.predict_proba(game)[0, 1]

    if verbose:
        print(f"Home Win Probability: {prob:.2%}")

    return prob


if __name__ == "__main__":
    # Example: Celtics @ Spurs — Spurs as home team won
    predict_home_win(spread=-3.5, total=221.5, season_phase=3)
