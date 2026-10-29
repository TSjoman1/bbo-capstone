# Model Card: Surrogate-Based Optimiser for the BBO Capstone

## Model description

**Input:** All the data observed so far for one black-box function: input vectors x ∈ [0, 1]^d (d = 2–8) and their scalar outputs y. That is the initial `.npy` data plus every weekly query and result, between 22 and 52 points per function by week 12.

**Output:** A recommended next query, x\* ∈ [0, 1]^d, for each function, formatted for submission as `0.xxxxxx-0.xxxxxx-…`. It also outputs diagnostic values: the predicted mean, the uncertainty, and which models agreed.

**Model architecture:** A surrogate-plus-acquisition loop with two components:

1. **Surrogate models** (scikit-learn):
   - **Gaussian process (GP):** constant × Matérn ν=2.5 kernel with a separate length scale per dimension, plus a white-noise kernel. Outputs are normalised.
   - **Week 9–12 ensemble:** SVR (RBF), random forest, extra trees, gradient boosting (Huber loss) and an MLP (32-16) as well as the GP. These were ranked by 3-fold cross-validated RMSE, and the top three were blended with inverse-squared-error weights.
   - **Week 13 function-specific models:**
     - a two-Gaussian-source model fitted to log|y| with differential evolution (F1)
     - a power-law edge model, y = c + a·x1^p (F5)
     - a ridge-regularised local quadratic (F3, F4, F7, F8)
2. **Acquisition and search:**
   - Sobol candidates (2^14 to 2^16) are drawn inside a trust region (a box) around the best point so far.
   - Week 9–12 scoring: 0.9 × ensemble mean + 0.1 × model disagreement.
   - Week 13 scoring:
     - posterior mean (F2)
     - probability of improvement including noise (F6)
     - agreement between the GP and the local quadratic (F7, F8)
   - **Region filter (F2, F3):** KMeans on the outputs plus an RBF-SVM classifier restricts candidates to the high-value cluster.

## Performance

**Measure:** The best output value found per function (the challenge objective is maximisation). Surrogate quality was monitored with 3-fold cross-validated RMSE on standardised outputs, and with the GP's predicted mean and standard deviation at candidate points.

**Data analysed:** The initial data plus 12 weekly queries per function.

![Best value found per function](images/best_so_far.png)

| Function | Initial best | Best by week 12 | Improvement | Assessment |
|---|---|---|---|---|
| F1 (2D) | 7.7e-16 | 7.7e-16 | none | The pipeline never found signal. Log-scale analysis in week 13 located two peaks. |
| F2 (2D) | 0.611 | 0.675 | +0.064 | Noisy (sd ≈ 0.05). The GP's estimated true value at the best point is ≈ 0.60. |
| F3 (3D) | −0.0348 | −0.00092 | +0.034 | Probably close to the maximum of 0. |
| F4 (4D) | −4.03 | 0.644 | +4.67 | Sharp peak; later queries near it scored slightly lower. |
| F5 (4D) | 1,089 | 4,442.6 | ×4.1 | Boundary optimum found in week 2. Still rising along x1. |
| F6 (5D) | −0.714 | −0.247 | +0.47 | Noisy (sd ≈ 0.06). |
| F7 (6D) | 1.365 | 2.606 | +1.24 | Improved in 8 of 12 weeks. |
| F8 (8D) | 9.598 | 9.990 | +0.39 | Improved in each of the last 5 weeks. |

**Week 13 results:** [fill in]

## Limitations

- **Very small data.** 22–52 points per function makes cross-validation rankings noisy and leaves high-dimensional spaces (F7, F8) mostly unexplored. Optima outside the trust region can't be found.
- **Tree models can't extrapolate.** They predict flat values beyond the data. This kept F5 stuck near x1 ≈ 0.10 for several weeks.
- **Output scale.** Without a log transform, F1's outputs (down to 1e-196) are numerically zero to every model.
- **Noise was underestimated until late.** The trust regions centred on lucky "best" points for F2 and F6.
- **Local search only.** Fixed small trust regions (0.012–0.02) guarantee small, safe steps but miss other modes. It is a local optimiser, not a global one.
- **The F1 and F5 week-13 models are hypotheses.** They fit the observed data well but rest on assumed functional forms (Gaussian peaks, a power law), and the sign of F1's peaks isn't explained by the model.
- **Synthetic setting.** Each function is a single anonymous benchmark. The model encodes no domain knowledge, and its behaviour on real, drifting or constrained problems is untested.

## Trade-offs

- **Exploration vs exploitation.** Weighting 90% on the mean gave reliable small gains on smooth functions (F3, F7, F8) at the cost of missing big opportunities elsewhere (F1's peaks, F5's extrapolation). The week-13 script deliberately takes more risk, which is only sensible if the best value ever found is what gets scored.
- **Generic ensemble vs specific models.** The six-model ensemble needs no assumptions but performs poorly where the function breaks its implicit assumptions. Specific models (power law, two-peak model, local quadratic) are more data-efficient but can fail badly if their assumed form is wrong.
- **Interpretability vs flexibility.** GPs and quadratics give length scales, gradients and uncertainty you can read directly. Trees and MLPs are less interpretable, and they're also unreliable at just 20–50 points.
- **Noise robustness vs speed.** Choosing points by posterior mean or probability of improvement protects against chasing noise, but on noisy functions it often recommends points whose expected value is below the best observed value.
- **Compute.** The full script runs in about 25 seconds on a laptop. GP restarts and differential evolution dominate the cost, and both grow with data size: GP fitting is O(n³).
