import numpy as np, pandas as pd, warnings, json, os
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.svm import SVR, SVC
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import qmc

warnings.filterwarnings('ignore')
SEED=20261008
rng=np.random.default_rng(SEED)

paths={
1:('/mnt/data/initial_inputs1(1).npy','/mnt/data/initial_outputs1(1).npy'),
2:('/mnt/data/initial_inputs2(1).npy','/mnt/data/initial_outputs2(1).npy'),
3:('/mnt/data/initial_inputs3(1).npy','/mnt/data/initial_outputs3(1).npy'),
4:('/mnt/data/initial_inputs4(1).npy','/mnt/data/initial_outputs4(1).npy'),
5:('/mnt/data/initial_inputs5(1).npy','/mnt/data/initial_outputs5(1).npy'),
6:('/mnt/data/initial_inputs6(2).npy','/mnt/data/initial_outputs6(2).npy'),
7:('/mnt/data/initial_inputs7(2).npy','/mnt/data/initial_outputs7(2).npy'),
8:('/mnt/data/initial_inputs8(2).npy','/mnt/data/initial_outputs8(2).npy')}

week_X={
1:[[0.004675,0.206855],[0.794266,0.736888],[0.000001,0.999999],[0.992948,0.112704],[0.426354,0.492733],[0.001156,0.683876],[0.999999,0.882817],[0.679936,0.005556],[0.248984,0.999840]],
2:[[0.704368,0.000001],[0.699305,0.323774],[0.639382,0.926552],[0.728910,0.999999],[0.705102,0.008500],[0.710766,0.008566],[0.705115,0.019472],[0.785822,0.999194],[0.705088,0.005787]],
3:[[0.999999,0.999999,0.396121],[0.985471,0.700565,0.343019],[0.548814,0.715189,0.602763],[0.969986,0.041360,0.008370],[0.505078,0.613544,0.339546],[0.494084,0.608958,0.388328],[0.499180,0.507055,0.473245],[0.507659,0.510529,0.503245],[0.497876,0.506954,0.465989]],
4:[[0.440431,0.425455,0.378362,0.397090],[0.417630,0.419189,0.353154,0.430091],[0.440000,0.440000,0.380000,0.450000],[0.401347,0.377003,0.254333,0.397563],[0.417178,0.420463,0.374467,0.429092],[0.421021,0.425731,0.374563,0.425807],[0.414343,0.423853,0.379653,0.420414],[0.421138,0.428902,0.396446,0.438262],[0.414323,0.422578,0.392505,0.421134]],
5:[[0.208085,0.956226,0.999999,0.961039],[0.140270,0.999999,0.999999,0.999999],[0.173370,0.999092,0.946636,0.224417],[0.000001,0.999999,0.999999,0.999999],[0.027197,0.998748,0.997196,0.999718],[0.062192,0.998867,0.999159,0.999132],[0.047258,0.999999,0.999999,0.999999],[0.045736,0.999999,0.999999,0.999999],[0.117640,0.999999,0.999999,0.999999]],
6:[[0.421438,0.141251,0.803533,0.999999,0.000001],[0.536196,0.223866,0.572390,0.784908,0.000001],[0.425028,0.141785,0.737334,0.940849,0.011376],[0.381276,0.178455,0.509914,0.779381,0.137596],[0.365177,0.180035,0.512483,0.781082,0.130009],[0.435715,0.262074,0.530873,0.789319,0.160199],[0.448781,0.258846,0.529013,0.801203,0.150451],[0.422142,0.258924,0.487695,0.801303,0.155280],[0.448741,0.255252,0.532323,0.803371,0.156084]],
7:[[0.000001,0.289680,0.379057,0.207442,0.377853,0.749214],[0.000001,0.255092,0.376720,0.129619,0.362952,0.800080],[0.120739,0.235472,0.272940,0.102002,0.421342,0.801353],[0.000001,0.253085,0.350739,0.196710,0.346325,0.743091],[0.001416,0.230384,0.373684,0.197251,0.340429,0.748148],[0.001641,0.182613,0.390803,0.216426,0.342594,0.752385],[0.000001,0.121336,0.398237,0.226646,0.327315,0.740116],[0.000001,0.088081,0.398235,0.283630,0.329243,0.743592],[0.000837,0.081593,0.391262,0.271336,0.328217,0.746693]],
8:[[0.107521,0.000001,0.135540,0.000001,0.999999,0.472313,0.154741,0.399472],[0.112528,0.114528,0.154266,0.000001,0.999999,0.491108,0.249710,0.471215],[0.339689,0.263857,0.038844,0.154532,0.991473,0.542394,0.152953,0.543523],[0.087420,0.116012,0.169763,0.000077,0.811536,0.403175,0.069227,0.744485],[0.119009,0.102372,0.137573,0.000245,0.957373,0.440706,0.198938,0.554364],[0.058027,0.161187,0.069257,0.065179,0.987013,0.428646,0.137172,0.626297],[0.122131,0.146371,0.157152,0.000001,0.999999,0.468642,0.216491,0.628673],[0.089123,0.107755,0.117229,0.010956,0.960781,0.455072,0.195218,0.553461],[0.089950,0.143581,0.097454,0.045280,0.944889,0.482884,0.214280,0.581363]]}
week_y={
1:[7.447341643458188e-154,-3.9815247433350936e-28,0,1.215351412887282e-278,-0.0002670268408892753,5.36775804210243e-172,1.5090671292380316e-141,8.766390066198821e-168,1.4191785946244262e-196],
2:[0.6172900751980995,0.5015986538537118,0.33654561095427205,0.6031643899593944,0.6748009846303914,0.5779249408893591,0.5141012036690533,0.14862607096797514,0.6227582290178335],
3:[-0.05207167658519098,-0.036023459020365835,-0.06396500086888668,-0.17318514952306402,-0.040369855563413326,-0.010617970059944836,-0.0022541540582801754,-0.026966821284351662,-0.005940851105715514],
4:[0.2595037973219054,0.622786841822617,-0.2204255338978487,-1.8483746365405618,0.6243228121681388,0.6213069833451219,0.6440206402279682,0.3948137719254743,0.6319077553755013],
5:[3434.2749503065615,4442.5587689331,1231.5653431156597,4440.480873839292,4380.751187601097,4401.390418904232,4440.558208158286,4440.5522348394325,4441.619946911586],
6:[-0.7751811524938192,-0.49682453472547217,-0.6086872195135707,-0.4063144184731558,-0.41875596392333775,-0.3422713756820223,-0.24657027587933755,-0.2762220750399286,-0.44321122782708317],
7:[2.152118715860644,1.7719677362044728,1.469733681482609,2.2780141516105292,2.3494843306163675,2.459272711382689,2.56419466102695,2.588532047136528,2.56725917862628],
8:[9.925911122306,9.947482025892,9.8223121504056,9.9255757417665,9.9583008493039,9.9475454545956,9.9526866303046,9.9629500993934,9.9743808474976]}

# Load + append data
data={}
for i in range(1,9):
    X0=np.load(paths[i][0]); y0=np.load(paths[i][1]).reshape(-1)
    X=np.vstack([X0,np.asarray(week_X[i],float)])
    y=np.concatenate([y0,np.asarray(week_y[i],float)])
    data[i]=(X,y)

# Models kept deliberately compact so this is reproducible on a laptop.
def models(d):
    kernel=ConstantKernel(1.0,(1e-3,1e3))*Matern(np.ones(d),(1e-3,1e3),nu=2.5)+WhiteKernel(1e-5,(1e-8,1e-1))
    return {
      'GP':GaussianProcessRegressor(kernel=kernel,normalize_y=False,n_restarts_optimizer=0,random_state=SEED),
      'SVR':make_pipeline(StandardScaler(),SVR(C=10,gamma='scale',epsilon=0.03)),
      'ExtraTrees':ExtraTreesRegressor(n_estimators=220,max_features=1.0,random_state=SEED,n_jobs=1),
      'RandomForest':RandomForestRegressor(n_estimators=220,max_features=1.0,random_state=SEED,n_jobs=1),
      'GradientBoosting':GradientBoostingRegressor(n_estimators=180,learning_rate=0.04,max_depth=2,loss='huber',random_state=SEED),
      'MLP':make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(32,16),alpha=0.01,max_iter=1800,early_stopping=True,random_state=SEED))
    }

# 3-fold CV once: enough to rank surrogates without making the run prohibitively slow.
cv_summary={}; fitted={}
for i,(X,y) in data.items():
    ym=float(y.mean()); ys=float(y.std()) or 1.0; yz=(y-ym)/ys
    cv=KFold(n_splits=3,shuffle=True,random_state=SEED+i)
    rows=[]
    for name,m in models(X.shape[1]).items():
        rmse=-cross_val_score(m,X,yz,cv=cv,scoring='neg_root_mean_squared_error',n_jobs=1)
        rows.append((name,float(rmse.mean()),float(rmse.std())))
    rows=sorted(rows,key=lambda z:z[1]); cv_summary[i]=rows
    top=rows[:3]
    fitlist=[]
    for name,err,std in top:
        m=models(X.shape[1])[name]; m.fit(X,yz); fitlist.append((name,m,err))
    fitted[i]=(fitlist,ym,ys)

# Dynamic partitions for F2/F3 based on objective-value clusters, following the paper's spirit.
partitions={}
for i in (2,3):
    X,y=data[i]; best=None
    for k in (2,3,4):
        if len(y)<=k: continue
        km=KMeans(n_clusters=k,n_init=30,random_state=SEED+i+k).fit(y.reshape(-1,1))
        sc=silhouette_score(y.reshape(-1,1),km.labels_)
        if best is None or sc>best[0]: best=(sc,k,km)
    sc,k,km=best
    means={lab:float(y[km.labels_==lab].mean()) for lab in range(k)}
    good=max(means,key=means.get)
    svc=SVC(kernel='rbf',C=10,gamma='scale').fit(X,(km.labels_==good).astype(int))
    partitions[i]=(k,sc,good,svc,means)

# Function-specific trust region radii around best-ever point (elite archive).
radius={2:0.015,3:0.015,4:0.012,6:0.012,7:0.015,8:0.020}

def local_candidates(center,r,n=32768,seed=1):
    d=len(center); sob=qmc.Sobol(d,scramble=True,seed=seed); u=sob.random_base2(int(np.log2(n)))
    lo=np.maximum(1e-6,center-r); hi=np.minimum(0.999999,center+r)
    return lo+(hi-lo)*u

def min_dist(C,X):
    # chunked min Euclidean distance
    out=np.full(len(C),np.inf)
    for j in range(0,len(C),4096):
        A=C[j:j+4096]
        D=((A[:,None,:]-X[None,:,:])**2).sum(axis=2)
        out[j:j+4096]=np.sqrt(D.min(axis=1))
    return out

def ensemble_score(i,C):
    fitlist,ym,ys=fitted[i]
    preds=np.column_stack([m.predict(C) for _,m,_ in fitlist])
    errs=np.array([e for _,_,e in fitlist])
    w=1/(errs**2+1e-9); w=w/w.sum()
    mu=preds@w
    disagree=preds.std(axis=1)
    # late-stage exploitation heavy
    return 0.90*mu+0.10*disagree, mu, disagree, w

results={}
# F1: model signal is pathological; maximize distance from observed points.
X,y=data[1]; sob=qmc.Sobol(2,scramble=True,seed=SEED+1); C=1e-6+(0.999998)*sob.random_base2(16)
d=min_dist(C,X); idx=int(np.argmax(d)); results[1]=(C[idx],None,None,None)

# F2/F3: local elite region + learned promising-region filter
for i in (2,3):
    X,y=data[i]; center=X[int(np.argmax(y))]; C=local_candidates(center,radius[i],32768,SEED+i)
    svc=partitions[i][3]; mask=svc.predict(C)==1
    if mask.sum()>=100: C=C[mask]
    # avoid essentially duplicate submissions
    C=C[min_dist(C,X)>2e-4]
    s,mu,sd,w=ensemble_score(i,C); idx=int(np.argmax(s)); results[i]=(C[idx],mu[idx],sd[idx],w)

# F4/F6/F7/F8: elite-centered trust regions
for i in (4,6,7,8):
    X,y=data[i]; center=X[int(np.argmax(y))]; C=local_candidates(center,radius[i],32768,SEED+i)
    C=C[min_dist(C,X)>2e-4]
    s,mu,sd,w=ensemble_score(i,C); idx=int(np.argmax(s)); results[i]=(C[idx],mu[idx],sd[idx],w)

# F5: one-dimensional refinement with x2=x3=x4 at upper boundary.
i=5; X,y=data[i]
x1=np.linspace(0.08,0.20,30001); C=np.column_stack([x1,np.full((len(x1),3),0.999999)])
C=C[min_dist(C,X)>1e-5]
s,mu,sd,w=ensemble_score(i,C); idx=int(np.argmax(s)); results[i]=(C[idx],mu[idx],sd[idx],w)

# output log
rows=[]
for i in range(1,9):
    X,y=data[i]; x,mu,sd,w=results[i]
    top=cv_summary[i][:3]
    pred=None if mu is None else float(mu*fitted[i][2]+fitted[i][1])
    rows.append({'function':i,'n_obs':len(y),'best_y':float(y.max()),'best_x':X[y.argmax()].tolist(),
                 'top_models':[(a,round(b,6)) for a,b,_ in top],
                 'candidate':x.tolist(),'predicted_y':pred})

print('=== CV + WEEK 10 RECOMMENDATIONS ===')
for row in rows:
    print(f"F{row['function']}: n={row['n_obs']} best={row['best_y']:.12g}")
    print('  top:',row['top_models'])
    print('  candidate:', '-'.join(f'{v:.6f}' for v in row['candidate']))
    print('  predicted_y:',row['predicted_y'])
for i in (2,3):
    k,sc,good,svc,means=partitions[i]
    print(f'F{i} dynamic KMeans: K={k}, silhouette={sc:.4f}, cluster means={means}, good={good}')

pd.DataFrame(rows).to_json('/mnt/data/bbo_week10_results.json',orient='records',indent=2)
