
import pandas as pd
import numpy as np

# ============================================================

DATA_PATH = "/kaggle/input/mydataset/new.csv"

df = pd.read_csv(DATA_PATH)

# ============================================================

option_cols = [
    c for c in df.columns
    if ("CE" in c or "PE" in c)
]

surface = df[option_cols].copy()

print("Surface shape:", surface.shape)

# ============================================================


def predict_missing(df_):

    pred_df = df_.copy()

    rows = pred_df.shape[0]
    cols = pred_df.shape[1]

 
    # --------------------------------------------------------

    max_iter = 10

    for iteration in range(max_iter):

        old_nan_count = pred_df.isna().sum().sum()

        temp_df = pred_df.copy()

       
        # ====================================================

        for r in range(rows):

            for c in range(cols):

                
                # ------------------------------------------------

                if not pd.isna(pred_df.iloc[r,c]):
                    continue

                preds = []
                weights = []

              
                # =================================================

                temporal_neighbors = []

                for dr,w in [
                    (-1,0.35),
                    (1,0.35),
                    (-2,0.15),
                    (2,0.15)
                ]:

                    rr = r + dr

                    if rr < 0 or rr >= rows:
                        continue

                    val = pred_df.iloc[rr,c]

                    if not pd.isna(val):

                        temporal_neighbors.append(val)

                        preds.append(val)
                        weights.append(w)

                

                strike_neighbors = []

                for dc,w in [
                    (-1,0.08),
                    (1,0.08),
                    (-2,0.02),
                    (2,0.02)
                ]:

                    cc = c + dc

                    if cc < 0 or cc >= cols:
                        continue

                    val = pred_df.iloc[r,cc]

                    if not pd.isna(val):

                        strike_neighbors.append(val)

                        preds.append(val)
                        weights.append(w)

                # =================================================
                

                for dr,dc,w in [
                    (-1,-1,0.01),
                    (-1,1,0.01),
                    (1,-1,0.01),
                    (1,1,0.01)
                ]:

                    rr = r + dr
                    cc = c + dc

                    if (
                        rr < 0
                        or rr >= rows
                        or cc < 0
                        or cc >= cols
                    ):
                        continue

                    val = pred_df.iloc[rr,cc]

                    if not pd.isna(val):

                        preds.append(val)
                        weights.append(w)

                # =================================================

                if len(preds) >= 2:

                    preds = np.array(preds)
                    weights = np.array(weights)

                    weights = (
                        weights / weights.sum()
                    )

                    pred = np.sum(
                        preds * weights
                    )

                    # --------------------------------------------
                  

                    if len(temporal_neighbors) >= 2:

                        temp_mean = np.mean(
                            temporal_neighbors
                        )

                        pred = (
                            0.7 * pred
                            + 0.3 * temp_mean
                        )

                    if len(strike_neighbors) >= 2:

                        strike_mean = np.mean(
                            strike_neighbors
                        )

                        pred = (
                            0.9 * pred
                            + 0.1 * strike_mean
                        )

                    temp_df.iloc[r,c] = pred

        pred_df = temp_df

        # ====================================================
       

        new_nan_count = pred_df.isna().sum().sum()

        print(
            f"Iteration {iteration+1} | "
            f"Remaining NaNs: {new_nan_count}"
        )

        if new_nan_count == old_nan_count:
            break

    # ========================================================


    pred_df = pred_df.interpolate(
        method="linear",
        axis=0,
        limit_direction="both"
    )

    pred_df = pred_df.interpolate(
        method="linear",
        axis=1,
        limit_direction="both"
    )

    pred_df = pred_df.ffill().bfill()

    pred_df = pred_df.fillna(
        pred_df.median()
    )

    return pred_df


# ============================================================

pred_surface = predict_missing(surface)


# ============================================================

submission_df = df.copy()

submission_df[option_cols] = pred_surface

# =======================

OUTPUT_PATH = "/kaggle/working/final_submission.csv"

submission_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n===================================")
print("SUBMISSION SAVED")
print("===================================")

print("Path:", OUTPUT_PATH)

print("\nNaNs remaining:")
print(
    submission_df[option_cols]
    .isna()
    .sum()
    .sum()
)

# ============================================================
# PREVIEW
# ============================================================

print("\nSubmission preview:")

display(
    submission_df.head()
)