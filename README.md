# NIFTY_options
submission1

This project reconstructs missing implied volatility (IV) values in an options dataset using a LightGBM regression model.

Methodology-


##Data Preparation
Loads option chain data and converts timestamps.
Extracts strike price and option type (CE/PE) from column names.
Computes time to expiry (TTE).
Feature Engineering
Each option contract is transformed into a training sample using:
TTE, Spot price, Strike price, Moneyness (strike / spot), Option type


##Surface-level IV anchor (mean IV at each timestamp)
Model Training
Trains a LightGBM regressor on available IV values.
Uses high-capacity settings to capture volatility surface patterns and local irregularities.
IV Reconstruction
Predicts missing IV values.
Clips negative predictions to maintain valid IV values.


##Output & Validation
Reconstructs the original option chain with predicted IVs.
