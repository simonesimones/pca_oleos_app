import numpy as np
from core import *
rng=np.random.default_rng(12)
for shape in [(12,3),(5,20),(10,1)]:
 x=rng.normal(size=shape); z,*_=preprocess(x); t,p,l,e=pca(z)
 assert np.allclose(z,t@p.T); assert np.allclose(e.sum(),100); assert np.allclose(l,np.linalg.svd(z,compute_uv=False)[:len(l)]**2/(len(z)-1)); T,Q,*_=diagnostics(z,t,p,l,len(l)); assert np.max(Q)<1e-20; assert np.allclose(T,np.sum(t*t/l,axis=1)); assert hca(z,'ward','euclidean').shape==(shape[0]-1,4)
print('Testes numéricos OK')
