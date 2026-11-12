import numpy as np
from scipy.stats import f
from scipy.cluster.hierarchy import linkage

def preprocess(x, autoscale=True, weights=None):
    x=np.asarray(x,float)
    if x.ndim!=2 or len(x)<2 or x.shape[1]<1 or not np.isfinite(x).all(): raise ValueError('Matriz inválida ou com valores ausentes.')
    mu=x.mean(0); sd=x.std(0,ddof=1)
    if np.any(sd==0): raise ValueError('Remova variáveis constantes.')
    scale=sd if autoscale else np.ones_like(sd)
    w=np.ones_like(sd) if weights is None else np.asarray(weights,float)
    if w.shape!=sd.shape or not np.isfinite(w).all() or np.any(w<=0): raise ValueError('Pesos devem ser positivos e finitos.')
    return (x-mu)/scale*w,mu,scale,w

def pca(z):
    u,s,vt=np.linalg.svd(z,full_matrices=False)
    r=int(np.sum(s>np.finfo(float).eps*max(z.shape)*s[0]))
    if not r: raise ValueError('Matriz sem variância.')
    p=vt[:r].T; t=u[:,:r]*s[:r]; l=s[:r]**2/(len(z)-1)
    return t,p,l,100*l/l.sum()

def diagnostics(z,t,p,l,a,confidence=.95):
    n=len(z); T2=np.sum(t[:,:a]**2/l[:a],axis=1)
    Q=np.sum((z-t[:,:a]@p[:,:a].T)**2,axis=1)
    # In-sample empirical quantiles, descriptive only.
    return T2,Q,np.quantile(T2,confidence),np.quantile(Q,confidence)

def correlations(x,t):
    x=np.asarray(x,float); t=np.asarray(t,float)
    x=x-x.mean(0); t=t-t.mean(0)
    den=np.linalg.norm(x,axis=0)[:,None]*np.linalg.norm(t,axis=0)[None,:]
    return np.divide(x.T@t,den,out=np.full(den.shape,np.nan),where=den>0)

def hca(z,method,metric):
    if method=='ward' and metric!='euclidean': raise ValueError('Ward exige distância euclidiana.')
    return linkage(z,method=method,metric=metric)
