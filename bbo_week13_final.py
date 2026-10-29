"""
BBO Capstone - Week 13 (final query) recommendation script
===========================================================

Builds on bbo_week10_run.py. Main changes vs. that script:

  * F1  : explicit two-Gaussian-source model fitted on log|y| (instead of
          pure space-filling).  The magnitudes of all non-zero observations
          are explained to within ~1 nat by two narrow peaks on the diagonal.
  * F2/F6: noise-aware GP (WhiteKernel with realistic bounds).  Near-repeat
          submissions show noise sd ~0.045 (F2) / ~0.06 (F6), so the
          "best observed" point is partly luck; we pick by posterior mean
          (F2) or by probability of beating the incumbent (F6).
  * F5  : parametric extrapolation along the x2=x3=x4=1 edge
          (y = c + a*x1^p).  Tree models cannot extrapolate, which kept the
          old pipeline pinned at x1~0.10.
  * F3/F4/F7/F8: GP (with hyper-parameter restarts) + local quadratic;
          a candidate is only accepted where BOTH models beat the incumbent
          (F7, F8).  F3/F4 have no agreed improvement -> fallbacks.
  * F8  : larger trust region (consistent improvement every week -> expand,
          TuRBO-style).

Assumption: the challenge scores the BEST value ever found, so the final
query is allowed to be risky.  Set SCORE_IS_BEST_EVER = False for the
conservative variant.

Usage:  python bbo_week13_final.py [data_dir]
        data_dir must contain initial_inputsK*.npy / initial_outputsK*.npy
"""
import sys, glob, os, warnings
import numpy as np
from scipy.optimize import least_squares, differential_evolution, curve_fit
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel as W
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
np.set_printoptions(precision=6, suppress=True)
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data"
SCORE_IS_BEST_EVER = True
LO, HI = 1e-6, 0.999999

# ----------------------------------------------------------------------------
# Submitted queries and results, weeks 1-12
# ----------------------------------------------------------------------------
week_X = {
1: [[0.004675,0.206855],[0.794266,0.736888],[0.000001,0.999999],[0.992948,0.112704],[0.426354,0.492733],[0.001156,0.683876],[0.999999,0.882817],[0.679936,0.005556],[0.248984,0.999840],[0.596129,0.311340],[0.100491,0.000648],[0.221543,0.568580]],
2: [[0.704368,0.000001],[0.699305,0.323774],[0.639382,0.926552],[0.728910,0.999999],[0.705102,0.008500],[0.710766,0.008566],[0.705115,0.019472],[0.785822,0.999194],[0.705088,0.005787],[0.705103,0.008149],[0.704803,0.008497],[0.705248,0.008694]],
3: [[0.999999,0.999999,0.396121],[0.985471,0.700565,0.343019],[0.548814,0.715189,0.602763],[0.969986,0.041360,0.008370],[0.505078,0.613544,0.339546],[0.494084,0.608958,0.388328],[0.499180,0.507055,0.473245],[0.507659,0.510529,0.503245],[0.497876,0.506954,0.465989],[0.498794,0.507162,0.474515],[0.499026,0.507445,0.470977],[0.498747,0.503111,0.474570]],
4: [[0.440431,0.425455,0.378362,0.397090],[0.417630,0.419189,0.353154,0.430091],[0.440000,0.440000,0.380000,0.450000],[0.401347,0.377003,0.254333,0.397563],[0.417178,0.420463,0.374467,0.429092],[0.421021,0.425731,0.374563,0.425807],[0.414343,0.423853,0.379653,0.420414],[0.421138,0.428902,0.396446,0.438262],[0.414323,0.422578,0.392505,0.421134],[0.412539,0.419988,0.390808,0.418968],[0.412165,0.423834,0.391304,0.418244],[0.402365,0.412618,0.379570,0.412647]],
5: [[0.208085,0.956226,0.999999,0.961039],[0.140270,0.999999,0.999999,0.999999],[0.173370,0.999092,0.946636,0.224417],[0.000001,0.999999,0.999999,0.999999],[0.027197,0.998748,0.997196,0.999718],[0.062192,0.998867,0.999159,0.999132],[0.047258,0.999999,0.999999,0.999999],[0.045736,0.999999,0.999999,0.999999],[0.117640,0.999999,0.999999,0.999999],[0.083372,0.999999,0.999999,0.999999],[0.101342,0.999999,0.999999,0.999999],[0.101612,0.999999,0.999999,0.999999]],
6: [[0.421438,0.141251,0.803533,0.999999,0.000001],[0.536196,0.223866,0.572390,0.784908,0.000001],[0.425028,0.141785,0.737334,0.940849,0.011376],[0.381276,0.178455,0.509914,0.779381,0.137596],[0.365177,0.180035,0.512483,0.781082,0.130009],[0.435715,0.262074,0.530873,0.789319,0.160199],[0.448781,0.258846,0.529013,0.801203,0.150451],[0.422142,0.258924,0.487695,0.801303,0.155280],[0.448741,0.255252,0.532323,0.803371,0.156084],[0.447954,0.258854,0.529640,0.799688,0.150316],[0.447803,0.259742,0.517827,0.801181,0.153989],[0.453253,0.258479,0.526762,0.801094,0.152355]],
7: [[0.000001,0.289680,0.379057,0.207442,0.377853,0.749214],[0.000001,0.255092,0.376720,0.129619,0.362952,0.800080],[0.120739,0.235472,0.272940,0.102002,0.421342,0.801353],[0.000001,0.253085,0.350739,0.196710,0.346325,0.743091],[0.001416,0.230384,0.373684,0.197251,0.340429,0.748148],[0.001641,0.182613,0.390803,0.216426,0.342594,0.752385],[0.000001,0.121336,0.398237,0.226646,0.327315,0.740116],[0.000001,0.088081,0.398235,0.283630,0.329243,0.743592],[0.000837,0.081593,0.391262,0.271336,0.328217,0.746693],[0.000222,0.088733,0.390440,0.289189,0.328725,0.739184],[0.000731,0.081186,0.397753,0.299505,0.329328,0.740001],[0.000712,0.095712,0.390355,0.295142,0.324047,0.740029]],
8: [[0.107521,0.000001,0.135540,0.000001,0.999999,0.472313,0.154741,0.399472],[0.112528,0.114528,0.154266,0.000001,0.999999,0.491108,0.249710,0.471215],[0.339689,0.263857,0.038844,0.154532,0.991473,0.542394,0.152953,0.543523],[0.087420,0.116012,0.169763,0.000077,0.811536,0.403175,0.069227,0.744485],[0.119009,0.102372,0.137573,0.000245,0.957373,0.440706,0.198938,0.554364],[0.058027,0.161187,0.069257,0.065179,0.987013,0.428646,0.137172,0.626297],[0.122131,0.146371,0.157152,0.000001,0.999999,0.468642,0.216491,0.628673],[0.089123,0.107755,0.117229,0.010956,0.960781,0.455072,0.195218,0.553461],[0.089950,0.143581,0.097454,0.045280,0.944889,0.482884,0.214280,0.581363],[0.100155,0.163020,0.107280,0.064267,0.925274,0.464208,0.217046,0.565674],[0.119488,0.152169,0.122106,0.080816,0.906155,0.469469,0.202247,0.554161],[0.122105,0.147461,0.135558,0.099923,0.887853,0.450509,0.195506,0.563905]]}
week_y = {
1: [7.447341643458188e-154,-3.9815247433350936e-28,0,1.215351412887282e-278,-0.0002670268408892753,5.36775804210243e-172,1.5090671292380316e-141,8.766390066198821e-168,1.4191785946244262e-196,-5.4584429273294365e-31,-9.605064139485034e-196,-2.0218902661924853e-44],
2: [0.6172900751980995,0.5015986538537118,0.33654561095427205,0.6031643899593944,0.6748009846303914,0.5779249408893591,0.5141012036690533,0.14862607096797514,0.6227582290178335,0.5435093441814842,0.6367504198505801,0.6717911720520203],
3: [-0.05207167658519098,-0.036023459020365835,-0.06396500086888668,-0.17318514952306402,-0.040369855563413326,-0.010617970059944836,-0.0022541540582801754,-0.026966821284351662,-0.005940851105715514,-0.0009193517836112129,-0.0029165986468995533,-0.0010389783516082827],
4: [0.2595037973219054,0.622786841822617,-0.2204255338978487,-1.8483746365405618,0.6243228121681388,0.6213069833451219,0.6440206402279682,0.3948137719254743,0.6319077553755013,0.6318264280035,0.6169458500211893,0.521831241483977],
5: [3434.2749503065615,4442.5587689331,1231.5653431156597,4440.480873839292,4380.751187601097,4401.390418904232,4440.558208158286,4440.5522348394325,4441.619946911586,4440.85636071105,4441.176931571624,4441.1829695492925],
6: [-0.7751811524938192,-0.49682453472547217,-0.6086872195135707,-0.4063144184731558,-0.41875596392333775,-0.3422713756820223,-0.24657027587933755,-0.2762220750399286,-0.44321122782708317,-0.36534836289533923,-0.33251575158078533,-0.3994650990856157],
7: [2.152118715860644,1.7719677362044728,1.469733681482609,2.2780141516105292,2.3494843306163675,2.459272711382689,2.56419466102695,2.588532047136528,2.56725917862628,2.5967521184657003,2.5858722712844884,2.605808249740993],
8: [9.925911122306,9.947482025892,9.8223121504056,9.9255757417665,9.9583008493039,9.9475454545956,9.9526866303046,9.9629500993934,9.9743808474976,9.9811048745994,9.9874755562034,9.989936800548]}


def load(i):
    fx = sorted(glob.glob(os.path.join(DATA_DIR, f"*initial_inputs{i}*.npy")))[0]
    fy = sorted(glob.glob(os.path.join(DATA_DIR, f"*initial_outputs{i}*.npy")))[0]
    X = np.vstack([np.load(fx), np.asarray(week_X[i], float)])
    y = np.concatenate([np.load(fy).reshape(-1), np.asarray(week_y[i], float)])
    return X, y


def fit_gp(X, y, noise=(1e-6, 1e-1), noise0=1e-3, restarts=10):
    k = C(1.0) * Matern(np.full(X.shape[1], 0.3), (1e-3, 1e3), nu=2.5) + W(noise0, noise)
    return GPR(k, normalize_y=True, n_restarts_optimizer=restarts, random_state=0).fit(X, y)


def fit_local_quad(X, y, center, n):
    idx = np.argsort(np.linalg.norm(X - center, axis=1))[:n]
    P = PolynomialFeatures(2)
    m = RidgeCV(alphas=np.logspace(-6, 2, 40)).fit(P.fit_transform(X[idx] - center), y[idx])
    return lambda Z: m.predict(P.transform(Z - center))


def box(center, r, n_log2=14, seed=5):
    U = qmc.Sobol(len(center), scramble=True, seed=seed).random_base2(n_log2)
    return np.clip(center - r + 2 * r * U, LO, HI)


results = {}

# ----------------------------------------------------------------------------
# F1 : two-source Gaussian model on log|y|
#   ln|y| = ln(A1 g1 - A2 g2), g = exp(-|x-c|^2 / 2s^2), fitted with DE.
#   Near-field points are up-weighted (far tail values ~1e-200 carry less
#   information and may be affected by numerics).
# ----------------------------------------------------------------------------
X, y = load(1)
nz = y != 0
L, S, Xn = np.log(np.abs(y[nz])), np.sign(y[nz]), X[nz]
w = np.exp(L / 60)

def f1_model(p, Z):
    a1, x1, y1, s1, a2, x2, y2, s2 = p
    l1 = a1 - ((Z[:, 0] - x1) ** 2 + (Z[:, 1] - y1) ** 2) / (2 * s1 ** 2)
    l2 = a2 - ((Z[:, 0] - x2) ** 2 + (Z[:, 1] - y2) ** 2) / (2 * s2 ** 2)
    hi, lo = np.maximum(l1, l2), np.minimum(l1, l2)
    return np.where(l1 >= l2, 1, -1), hi + np.log1p(-np.exp(lo - hi) + 1e-300)

def f1_loss(p):
    s, m = f1_model(p, Xn)
    return np.sum(w * (m - L) ** 2) + 400 * np.sum(w * (s != S))

bounds = [(-5, 15), (0, 1), (0, 1), (.005, .15)] * 2
best = min((differential_evolution(f1_loss, bounds, seed=s, maxiter=3000, popsize=40,
                                   tol=1e-10, polish=True) for s in range(3)),
           key=lambda o: o.fun)
p = best.x
_, mag = f1_model(p, Xn)
print(f"F1 two-source fit: peak A c=({p[1]:.4f},{p[2]:.4f}) sigma={p[3]:.4f} lnA={p[0]:.2f} | "
      f"peak B c=({p[5]:.4f},{p[6]:.4f}) sigma={p[7]:.4f} | log-mag RMS (top 9) "
      f"{np.sqrt(np.mean(np.sort((mag - L)**2)[:9])):.2f}")
# Peak A: ~8/10 of its near-field observations are positive
results[1] = (np.array([p[1], p[2]]), f"peak A centre, predicted |y|~{np.exp(p[0]):.2f}")
if not SCORE_IS_BEST_EVER:
    results[1] = (X[np.argmax(y)], "safe: best observed")

# ----------------------------------------------------------------------------
# F2 : noisy plateau -> maximise GP posterior MEAN (noise-aware kernel)
# ----------------------------------------------------------------------------
X, y = load(2)
g = fit_gp(X, y, noise=(1e-3, 1.0), noise0=0.05)
noise_sd = np.sqrt(g.kernel_.k2.noise_level) * y.std()
Cc = box(X[np.argmax(y)], 0.01, seed=2)
m = g.predict(Cc)
j = np.argmax(m)
x2 = Cc[j].copy(); x2[1] = 0.0085           # x2 is irrelevant (length-scale ~100)
results[2] = (x2, f"posterior mean {m[j]:.4f}; noise sd~{noise_sd:.3f}; best-observed was a lucky draw")

# ----------------------------------------------------------------------------
# F3 / F4 : GP + local quadratic; accept only if both agree, else fallback
# F7 / F8 : same, with bigger trust region for F8 (improved every week)
# ----------------------------------------------------------------------------
cfg = {3: (.02, 20), 4: (.02, 25), 7: (.04, 30), 8: (.06, 40)}
for i, (r, nq) in cfg.items():
    X, y = load(i)
    b = X[np.argmax(y)]
    g = fit_gp(X, y)
    q = fit_local_quad(X, y, b, nq)
    Cc = box(b, r)
    mg, sg = g.predict(Cc, return_std=True)
    mq = q(Cc)
    inc = y.max()
    ok = (mg > inc) & (mq > inc)
    if ok.any():
        rank = np.argsort(np.argsort(mg)) + np.argsort(np.argsort(mq))
        j = np.argmax(np.where(ok, rank, -1))
        results[i] = (Cc[j], f"GP {mg[j]:.5f}±{sg[j]:.4f}, quad {mq[j]:.5f} (incumbent {inc:.5f}); "
                             f"{ok.sum()} agreeing candidates")
    elif i == 3:
        top2 = X[np.argsort(-y)[:2]]
        results[3] = (top2.mean(0), "no agreed improvement -> midpoint of two best points")
    elif i == 4:
        if SCORE_IS_BEST_EVER:
            Cc = box(b, 0.03, seed=5)
            mg, sg = g.predict(Cc, return_std=True)
            j = np.argmax(mg)
            results[4] = (Cc[j], f"upside play: GP {mg[j]:.4f}±{sg[j]:.3f} (quadratic disagrees)")
        else:
            results[4] = (b, "safe: best observed (week 7)")

# ----------------------------------------------------------------------------
# F5 : extrapolate along the x2=x3=x4=1 edge with y = c + a*x1^p
# ----------------------------------------------------------------------------
X, y = load(5)
edge = (X[:, 1:] > 0.9999).all(1)
f = lambda x, c, a, pw: c + a * x ** pw
pp, _ = curve_fit(f, X[edge, 0], y[edge], p0=[4440, 100, 3], maxfev=20000)
x1 = 0.25
results[5] = (np.array([x1, HI, HI, HI]),
              f"power-law edge fit predicts {f(x1, *pp):.1f} (best {y.max():.1f}); exponent {pp[2]:.2f}")

# ----------------------------------------------------------------------------
# F6 : noisy -> pick point maximising P(y_new > incumbent), incl. noise
# ----------------------------------------------------------------------------
X, y = load(6)
g = fit_gp(X, y, noise=(1e-3, 1.0), noise0=0.05)
nv = g.kernel_.k2.noise_level * y.std() ** 2
U = qmc.Sobol(5, seed=2).random(2 ** 15)
m, s = g.predict(U, return_std=True)
pi = norm.sf((y.max() - m) / np.sqrt(s ** 2 + nv))
j = np.argmax(pi)
b = X[np.argmax(y)]
mb, sb = g.predict(b[None], return_std=True)
pib = norm.sf((y.max() - mb[0]) / np.sqrt(sb[0] ** 2 + nv))
results[6] = (U[j], f"P(beat {y.max():.3f}) = {pi[j]:.2f} vs {pib:.2f} for resubmitting near best")
if not SCORE_IS_BEST_EVER:
    results[6] = (b, "safe: best observed")

# ----------------------------------------------------------------------------
print("\n=== WEEK 13 (FINAL) RECOMMENDATIONS ===")
for i in range(1, 9):
    x, note = results[i]
    x = np.clip(x, LO, HI)
    print(f"F{i}: " + "-".join(f"{v:.6f}" for v in x))
    print(f"     {note}")
