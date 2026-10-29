# Datasheet: BBO Capstone, Functions 1–8

This datasheet covers all eight functions. Questions that apply to every function are answered once. Function-specific answers are given in tables or per-function bullets.

> **Scenario column:** the real-world scenarios below follow the challenge brief. **[Check them against your course brief before submitting.]**

## Function overview

1. **Which function?** All eight; see the table.
2. **Real-world scenario:** see the table.
3. **Dimensionality:** 2D–8D; see the table.
4. **Initial data points:** see the table.
5. **What the output represents:** a scalar score to maximise. Where the real quantity should be minimised (side effects, error), it is supplied negated.

| Function | Scenario (per challenge brief) | Dimensions | Initial points | Output |
|---|---|---|---|---|
| F1 | Detecting contamination or radiation sources in a 2D area | 2 | 10 | Signal strength; almost zero away from a source |
| F2 | Tuning a noisy ML model | 2 | 10 | Log-likelihood-type score (noisy) |
| F3 | Drug discovery: combining three compounds | 3 | 15 | Negative adverse-reaction score (maximum ≈ 0) |
| F4 | Warehouse or product placement (four parameters) | 4 | 30 | Performance relative to a baseline |
| F5 | Chemical process yield | 4 | 20 | Yield (positive, heavy-tailed) |
| F6 | Recipe optimisation (five ingredients) | 5 | 20 | Negative quality loss |
| F7 | ML hyperparameter tuning (six hyperparameters) | 6 | 30 | Model performance score |
| F8 | High-dimensional hyperparameter tuning (eight hyperparameters) | 8 | 40 | Model performance score |

## Nature of the data

1. **Structure of the initial dataset.** For each function, an inputs array of shape (n, d) with values in [0, 1], and an outputs array of shape (n,):

   | Function | Inputs | Outputs |
   |---|---|---|
   | F1 | (10, 2) | (10,) |
   | F2 | (10, 2) | (10,) |
   | F3 | (15, 3) | (15,) |
   | F4 | (30, 4) | (30,) |
   | F5 | (20, 4) | (20,) |
   | F6 | (20, 5) | (20,) |
   | F7 | (30, 6) | (30,) |
   | F8 | (40, 8) | (40,) |

2. **How the dataset grows.** One point per function per week: 12 weekly points plus one final query, giving 22–52 points by week 12. The sampling pattern changed in three phases:
   - **Weeks 1–4:** boundary and corner probes, spread widely.
   - **Weeks 5–12:** tightly clustered in trust regions around the best point. Steps shrank to about 0.005–0.02 per coordinate.
   - **Week 13:** deliberate jumps based on function-specific models.
3. **Noise or randomness.** Near-repeat queries (within about 0.01 of each other) showed clear noise for two functions:
   - **F2:** outputs 0.54–0.675 at almost the same input; noise sd ≈ 0.045.
   - **F6:** outputs −0.44 to −0.25; noise sd ≈ 0.06.
   - **F3, F4, F5, F7, F8:** near-repeats differ only slightly and smoothly, consistent with low or no noise.

   Because of this, the final strategy for F2 and F6 chose points by GP posterior mean or probability of improvement, rather than chasing the best observed value.
4. **Landscape shape:**
   - **F1:** two very narrow peaks (σ ≈ 0.018) near (0.628, 0.629) and (0.421, 0.422). Elsewhere it is effectively zero (values down to 1e-278). The peaks were inferred from log|y|, which a Gaussian fits to within about 1 nat.
   - **F2:** noisy ridge along x1 ≈ 0.705, nearly independent of x2 (GP length scale ≈ 100).
   - **F3:** smooth single peak near (0.499, 0.506, 0.474); most sensitive to x3, nearly insensitive to x1.
   - **F4:** a sharp single peak; outputs fall from 0.644 to 0.52 within about 0.015.
   - **F5:** monotone, with the optimum on the boundary at x2 = x3 = x4 = 1 and still rising with x1. Heavy-tailed (1,089 initially → 4,442 on the edge).
   - **F6:** moderately smooth but noisy; possibly more than one peak.
   - **F7:** smooth, locally single-peaked; very sensitive to x6 (length scale 0.15), insensitive to x3.
   - **F8:** very smooth and nearly linear locally (GP length scales about 7–20).

## Your optimisation strategy

1. **Methods:**
   - **Weeks 1–4:** manual reasoning plus probing of boundaries and corners.
   - **Weeks 5–12:** surrogate-based Bayesian-style optimisation. A cross-validated ensemble (GP, SVR, random forest, extra trees, gradient boosting, MLP) was searched within a trust region, with 32,768 Sobol candidates scored as 0.9 × mean + 0.1 × model disagreement. For F2 and F3, KMeans plus an SVM filtered the candidates.
   - **Week 13:** a model for each function (see the next answer).
2. **Why these methods, per function:**
   - **F1:** after generic models failed, I switched to a two-Gaussian-source model on log|y|, because the outputs span about 600 orders of magnitude.
   - **F2 and F6:** a noise-aware GP, because repeat queries showed noise.
   - **F3, F4, F7, F8:** a GP plus local quadratic in a trust region, because these are smooth with a local optimum; a move was taken only if both models agreed.
   - **F5:** a 1D search along the boundary edge, then a power-law extrapolation, because the optimum is on the boundary and tree models can't extrapolate.
3. **Balancing exploration and exploitation.** Early weeks explored. From week 5 the search exploited heavily: small trust regions (radius 0.012–0.02) with a 90:10 weighting of mean to disagreement. In week 13, assuming the best value ever found is what's scored, I took calculated risks where models predicted a large upside (F1, F4, F5, F6) and exploited where models agreed (F7, F8).
4. **How the strategy changed:**
   - In week 2, finding the F5 boundary optimum led to edge-restricted searches.
   - In weeks 5–8, the switch to trust regions produced the steady gains on F3, F6 and F7.
   - In weeks 9–12, the ensemble and clustering pipeline was introduced.
   - In week 13, a critical review exposed blind spots: log scale for F1, noise for F2 and F6, extrapolation for F5, and trust-region size for F8.

## Data handling and preprocessing

1. **Input scaling.** The inputs are already in [0, 1], so no rescaling was needed for the GP. StandardScaler was applied inside the SVR and MLP pipelines, which are scale-sensitive. Candidates were clipped to [1e-6, 0.999999].
2. **Surrogate models.** A GP (Matérn 5/2 with a length scale per dimension, plus white noise), SVR, random forest, extra trees, gradient boosting and an MLP. Week 13 added a ridge-regularised local quadratic, a power-law model (F5) and a two-Gaussian log-magnitude model (F1).
3. **Preprocessing for the surrogates:**
   - Outputs were standardised (z-scored).
   - GP hyperparameters were fitted by maximum marginal likelihood, with 10 optimiser restarts in the final script (none in the week-10 pipeline).
   - Noise bounds were set from the observed repeat spread.
   - The F1 model was fitted on log|y| instead of y, and zero outputs were excluded.
4. **Outliers and unusual points:**
   - No points were removed.
   - Very negative probe results (F3: −0.17; F4: −1.85 and −32.6) were kept because they mark the edges of the landscape.
   - The F1 far-field values (below 1e-150) were down-weighted in its model, because they carry less information and may be affected by floating-point precision.
   - For F2 and F6, the "best" points were treated as possibly lucky draws rather than as outliers.

## Weekly iteration and learning

1. **How new data changed my view of the landscape.** Boundary probes showed that F5 is monotone towards a corner and that F2 depends on x1 alone. Repeat queries showed noise in F2 and F6. Steady gains showed that F7 and F8 are smooth and nearly linear around the incumbent. F1's structure only appeared once the data was viewed on a log scale.
2. **Local optima.**
   - **F4:** every move after week 7 scored 0.52–0.63, below 0.644, which suggests a sharp local (or global) peak.
   - **F3:** gains of about 0.001 suggest it is near the maximum.
   - **F5:** the 1D edge search was stuck, but this was a model limitation rather than a true local optimum.
   - Detection was by stalled best-so-far values, falling outputs around the incumbent, and models disagreeing about the direction to move.
3. **Most informative queries:**
   - F5, week 2 (the corner at x1 = 0.14): the largest gain in the project.
   - The F3 and F4 boundary probes in weeks 1–4: poor scores, but they ruled out regions.
   - The F2 and F6 near-repeats: they revealed the noise.
   - F1, week 5 (0.426, 0.493) → −2.7e-4: the only weekly F1 query close to a peak, and key to fitting the two-peak model.
4. **If I restarted:**
   - Plot every function and check output scale and noise in week 1, using log transforms and one deliberate repeat query each.
   - Use GPs with hyperparameter restarts as the main surrogate, avoiding tree models for extrapolation.
   - Use adaptive trust regions that expand after successes and shrink after failures (TuRBO-style).
   - Explore for longer on the low-dimensional functions.

## Performance and results

1. **Best values** and 2. **inputs that produced them** (to week 12):

| Function | Best value | Input vector | When |
|---|---|---|---|
| F1 | 7.71e-16 | [0.731024, 0.733000] | initial data |
| F2 | 0.674801 | [0.705102, 0.008500] | week 5 |
| F3 | −0.000919 | [0.498794, 0.507162, 0.474515] | week 10 |
| F4 | 0.644021 | [0.414343, 0.423853, 0.379653, 0.420414] | week 7 |
| F5 | 4442.56 | [0.140270, 0.999999, 0.999999, 0.999999] | week 2 |
| F6 | −0.246570 | [0.448781, 0.258846, 0.529013, 0.801203, 0.150451] | week 7 |
| F7 | 2.605808 | [0.000712, 0.095712, 0.390355, 0.295142, 0.324047, 0.740029] | week 12 |
| F8 | 9.989937 | [0.122105, 0.147461, 0.135558, 0.099923, 0.887853, 0.450509, 0.195506, 0.563905] | week 12 |

**Week 13 results:** **[fill in]**

3. **Confidence that these are near the global maximum:**
   - **High: F3.** Values are close to the likely ceiling of 0, and the models agree.
   - **Moderate: F4, F7, F8.** These are locally converged, but most of the 6D and 8D spaces are unexplored, and F7 and F8 were still improving.
   - **Low: F1.** The model predicts peak values around 0.6, far above 1e-16.
   - **Low: F5.** The edge trend is still rising.
   - **Low: F2 and F6.** The best values are inflated by noise, and the GP's estimated true value at the F2 best point is ≈ 0.60.
4. **Did results match expectations?** Mostly yes:
   - F1 behaved as a narrow-source detection problem would.
   - F5 behaved like a yield surface with a corner optimum.
   - F2 and F6 were noisy, as a noisy ML score or a recipe would be.
   - F7 and F8 were smooth, as hyperparameter response surfaces typically are.
   - The surprise was how much structure F1 hid on a linear scale.

## Ethical, practical and general considerations

1. **Real-world relevance.** The same setting appears in hyperparameter tuning, drug and materials discovery, chemical process optimisation, A/B testing and sensor placement: expensive evaluations, a limited budget and no gradients.
2. **Limitations of synthetic functions:**
   - There are no constraints, costs or safety limits.
   - The noise is stationary and the functions don't drift.
   - The inputs carry no meaning, so domain knowledge can't help.
   - There is one query per week regardless of difficulty, whereas real budgets are usually shared and adaptive.
3. **Would the strategy scale?**
   - The ideas would: surrogate models, trust regions and checking noise and scale.
   - The implementation would not scale as it is. GPs cost O(n³) and lose effectiveness above about 20 dimensions.
   - Hand-tuned radii and per-function judgement don't automate well. Larger problems would need TuRBO or BoTorch, batch acquisition and principled noise models.
4. **Risks and pitfalls for future users:**
   - Treating a lucky noisy observation as the optimum (F2, F6).
   - Using tree models where extrapolation is needed (F5).
   - Modelling raw outputs that span many orders of magnitude (F1).
   - Over-trusting cross-validation rankings on fewer than 50 points.
   - Fixed trust regions that are too small (F8).
   - Taking risky final-query choices without knowing the scoring rule.
