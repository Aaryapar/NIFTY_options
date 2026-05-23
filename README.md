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
------------------------------------------------


submission2
# Local Ensemble Interpolation for IV Surface Reconstruction
 reconstructs missing implied volatility (IV) values using a **local ensemble interpolation approach** based on neighboring observations.

The IV surface is treated as a 2D matrix:

- **Rows:** timestamps  
- **Columns:** option strikes (CE/PE)  
- **Cells:** implied volatility values  

---

## Approach

Missing values are reconstructed using:

### 1. Temporal Continuity (Primary Signal)
Uses nearby timestamps (`t-1`, `t+1`, `t-2`, `t+2`) with higher weights for closer observations.

### 2. Strike Continuity
Uses neighboring strikes to maintain smoothness across the IV smile.

### 3. Local Ensemble Prediction
Predictions from temporal, strike, and diagonal neighbors are combined using weighted averaging.

### 4. Iterative Reconstruction
The process runs iteratively so newly predicted values help reconstruct remaining missing entries.

### 5. Fallback Handling
Remaining missing values are filled using:
- Linear interpolation
- Forward/backward fill
- Median fallback

---


## Key Idea

The method leverages **temporal smoothness and strike continuity** to reconstruct missing IV values in a stable, fast, and reproducible way.
