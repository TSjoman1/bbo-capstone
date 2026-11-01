# Black-Box Optimisation Capstone: Maximising Eight Unknown Functions

## Non-technical explanation of the project

Imagine eight locked machines. Each one takes a few dial settings and gives back a score, but you can't see inside, and you're only allowed to try one setting per machine each week. My goal was to find the settings that give the highest scores. I used the results I'd seen so far to build simple "guess models" of each machine, then used those models to decide where to try next. That meant balancing safe bets near known good settings against riskier experiments elsewhere. Over 13 weeks, I improved the score on seven of the eight machines, sometimes dramatically, such as from 1,089 to 4,443.

## Repository contents

| File | What it is |
|---|---|
| `README.md` | This overview |
| `data_sheet.md` | Datasheet: the data, how it grew, and per-function notes |
| `model_card.md` | Model card: the surrogate models, performance and limitations |
| `bbo_week10_run.py` | Pipeline used in weeks 9–12 (ensemble surrogates and trust regions) |
| `bbo_week13_final.py` | Final-query script: function-specific models; all 12 weeks of data embedded |
| `data/` | Initial `.npy` input and output files for the eight functions (small, under 10 KB each) |
| `images/best_so_far.png` | Progress chart used below |

To reproduce the final recommendations:

```bash
pip install numpy scipy scikit-learn pandas
python bbo_week13_final.py data/
```

## Data

The data comes from the BBO capstone challenge in **[course name, e.g. Imperial College Business School / Emeritus Professional Certificate in Machine Learning and AI]**. Each of the eight functions is a hidden synthetic function on the unit hypercube [0, 1]^d, with d ranging from 2 to 8. The task is to maximise every function.

- **Initial data:** 10–40 points per function, supplied as `.npy` arrays (see the table below).
- **Weekly queries:** one new input per function per week, submitted through the course portal, with the output returned a few days later. That gave 12 weekly points plus one final query per function.
- **Size:** fewer than 60 points per function in total, so everything fits in this repository.

| Function | Dimensions | Initial points | Points by week 12 |
|---|---|---|---|
| F1 | 2 | 10 | 22 |
| F2 | 2 | 10 | 22 |
| F3 | 3 | 15 | 27 |
| F4 | 4 | 30 | 42 |
| F5 | 4 | 20 | 32 |
| F6 | 5 | 20 | 32 |
| F7 | 6 | 30 | 42 |
| F8 | 8 | 40 | 52 |

See `data_sheet.md` for more detail.

## Model

The approach is surrogate-based Bayesian-style optimisation, built from scratch with scikit-learn and SciPy. The method changed over three phases:

1. **Weeks 1–4: exploration.** I probed boundaries and corners to learn the structure of each function. This found the boundary optimum of F5 in week 2.
2. **Weeks 5–12: ensemble surrogates and trust regions.** For each function I compared six model types with 3-fold cross-validation: a Gaussian process (GP) with a Matérn 5/2 kernel, SVR, random forest, extra trees, gradient boosting and an MLP. I blended the top three, weighted by inverse error. I then searched 32,768 Sobol candidates in a small box around the best point, scoring them as 0.9 × predicted mean + 0.1 × disagreement between the models. For F2 and F3, KMeans clustering of the outputs plus an SVM classifier filtered the candidates down to the promising region.
3. **Week 13: a model for each function.** I fitted each function with a model that matched its observed behaviour:
   - **F1:** a two-Gaussian-peak model fitted to log|y|.
   - **F5:** a power-law extrapolation along the boundary.
   - **F2 and F6:** noise-aware GPs, choosing points by posterior mean or by probability of improvement.
   - **F3, F4, F7 and F8:** a GP and a local quadratic, accepting a move only where both models agreed.

**Why this approach?** With very few, expensive evaluations, a surrogate model is essential, and GPs give calibrated uncertainty when data is small. Writing it from scratch meant I could adapt each function's strategy as I learned about it. See `model_card.md`.

## Hyperparameter optimisation

| Hyperparameter | How it was chosen |
|---|---|
| GP kernel (length scale per dimension, amplitude, noise) | Maximum marginal likelihood. The final script restarts the optimiser 10 times; the week-10 pipeline used no restarts, which was a weakness. |
| GP noise bounds | Set from the spread of near-repeat queries: noise sd ≈ 0.045 for F2 and ≈ 0.06 for F6. |
| Surrogate choice and ensemble weights | 3-fold cross-validated RMSE on standardised outputs, keeping the top three models. |
| Tree, SVR and MLP settings | Fixed sensible defaults (e.g. 220 trees, SVR with C=10 and ε=0.03), kept small to avoid overfitting about 20–50 points. |
| Trust-region radius | Hand-tuned per function (0.012–0.02), widened to 0.06 for F8 in the final week after it improved every week. |
| Exploration weight | 0.9 × mean + 0.1 × model disagreement late in the challenge; risk taken deliberately in the final query. |

## Results

![Best value found per function](images/best_so_far.png)

| Function | Initial best | Best by week 12 | Found in | Week 13 result |
|---|---|---|---|---|
| F1 | 7.7e-16 | 7.7e-16 | initial data | **[ ]** |
| F2 | 0.611 | 0.675 | week 5 | **[ ]** |
| F3 | −0.0348 | −0.00092 | week 10 | **[ ]** |
| F4 | −4.03 | 0.644 | week 7 | **[ ]** |
| F5 | 1,089 | 4,442.6 | week 2 | **[ ]** |
| F6 | −0.714 | −0.247 | week 7 | **[ ]** |
| F7 | 1.365 | 2.606 | week 12 | **[ ]** |
| F8 | 9.598 | 9.990 | week 12 | **[ ]** |

**What I learned:**

- **Diagnose before optimising.** The generic pipeline worked well on smooth, low-noise functions (F3, F7 and F8 improved steadily) but stalled on functions that broke its assumptions:
  - **Extreme scale (F1).** Outputs as small as 1e-200 looked like pure zeros, but on a log scale they revealed two narrow, unexplored peaks.
  - **Noise (F2 and F6).** Near-repeat queries differed by about 0.05, so the "best" values were partly luck.
  - **A boundary optimum (F5).** Tree-based surrogates cannot extrapolate, so the search stayed stuck near x1 ≈ 0.10 while the trend kept rising.
- **Simple models built on a hypothesis about the function beat generic ensembles when data is scarce.** A power law, a two-peak model and a local quadratic were more useful than six cross-validated model families.
- **Exploration pays off early.** Boundary probes in weeks 1–4 scored badly but produced the largest single gain of the project (F5).

## Contact

**[Your name]** · **[GitHub / LinkedIn / email]**
