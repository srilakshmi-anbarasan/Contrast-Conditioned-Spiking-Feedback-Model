# STA 141 — Neural Spiking & Behavioral Feedback Modeling (R)

## Overview
This project analyzes 18 experimental mouse sessions stored as `.rds` files. It builds a combined dataset of trial-level behavioral outcomes and neural spiking activity, explores summary statistics and visualizations, normalizes neural activity across sessions, and trains multiple predictive models to classify feedback outcomes based on stimulus contrasts and neural features.

## What the Script Does
`STA 141 Project.R` performs:

1. **Load session data**
   - Reads `session1.rds` through `session18.rds` into a list.
2. **Session-level summaries**
   - Mouse ID, experiment date, number of trials, number of neurons, success/failure counts.
   - Visualizes trials per mouse.
3. **Stimulus condition visualization**
   - Scatter plot of `contrast_left` vs `contrast_right`.
4. **Neural activity exploration**
   - Computes average firing rates and spike count distributions.
   - Plots mean firing rate over time for an example session.
5. **Data integration**
   - Builds a unified, trial-level table with:
     - `mouse`, `date`, `feedback`, `contrast_left`, `contrast_right`, and `spikes` (trial spike counts).
   - Computes mean spike counts by stimuli + feedback.
6. **Normalization and de-biasing**
   - Standardizes spike counts within each mouse/date (`norm_spikes`).
   - Fits a linear model to remove session/mouse/day effects and extracts `residual_spikes`.
7. **Predictive modeling**
   - Splits data into train/test.
   - Trains and evaluates:
     - Logistic Regression
     - Random Forest
     - SVM (radial)
   - Reports accuracy on the test set.
8. **Export artifacts**
   - Saves integrated data (`integrated_data.rds`)
   - Saves “best model” (intended)
   - Generates prediction CSV for a provided test set

## Requirements
R packages used:
- `ggplot2`
- `data.table`
- `dplyr`
- `cluster`
- `caret`
- `randomForest`
- `e1071`

Install missing packages:
```r
install.packages(c(
  "ggplot2","data.table","dplyr","cluster",
  "caret","randomForest","e1071"
))
