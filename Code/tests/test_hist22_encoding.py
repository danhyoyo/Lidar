"""Independent Hist22 contract and population-covariance checks."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'detector'), str(ROOT / 'detector/core/datasets')]
from core.bev_encoding import resolve_bev_encoding
from utils_1.preprocess import encode_bev

G = dict(x_min=0., x_max=3.2, x_res=.1, y_min=-1.6, y_max=1.6, y_res=.1,
         z_min=-2.5, z_max=1., z_res=.1)


def encode(points, backend='numpy', geometry=G, **options):
    from utils_1.hist22_backend import encode_hist22_chw
    return encode_hist22_chw(points, resolve_bev_encoding(dict(name='hist22', backend=backend, **options), geometry))


def brute_force(points, geometry=G):
    p = np.asarray(points[:, :4], np.float64)
    keep = np.isfinite(p).all(axis=1)
    for k, axis in enumerate('xyz'):
        keep &= (p[:, k] > geometry[axis+'_min']+.001) & (p[:, k] < geometry[axis+'_max']-.001)
    p = p[keep]
    cell = np.floor((p[:, :2]-[geometry['x_min'], geometry['y_min']])/.1).astype(int)
    result = np.zeros((8, round((geometry['y_max']-geometry['y_min'])/.1), round((geometry['x_max']-geometry['x_min'])/.1)), np.float32)
    for x, y in np.unique(cell, axis=0):
        for s, radius in enumerate((5, 10)):
            q = p[(np.abs(cell[:, 0]-x) <= radius) & (np.abs(cell[:, 1]-y) <= radius), :3]
            n = len(q)
            result[s*2, y, x] = np.log1p(n)
            centered = q-q.mean(axis=0)
            eig = np.maximum(np.linalg.eigvalsh(centered.T@centered/n), 0)[::-1]
            trace = eig.sum()
            if n >= 6 and trace > 1e-12:
                result[s*2+1, y, x] = 1
                result[4+s*2, y, x] = np.clip((eig[0]-eig[1])/trace, 0, 1)
                result[5+s*2, y, x] = np.clip(2*(eig[1]-eig[2])/trace, 0, 1)
    return result


@pytest.mark.parametrize('options', [dict(out_channels=True), dict(out_channels=22.), dict(out_channels=14), dict(version=True), dict(version=2), dict(backend='auto'), dict(windows=[11,21]), dict(min_neighbors=6)])
def test_strict_schema(options):
    with pytest.raises(ValueError):
        resolve_bev_encoding(dict(name='hist22', **options), G)


def test_schema_metadata_backend_independent_and_fixed_resolution():
    a = resolve_bev_encoding(dict(name='hist22'), G)
    b = resolve_bev_encoding(dict(name='hist22', backend='numba'), G)
    assert a.channels == 22 and a.backend == 'numpy'
    assert a.semantic_hash == b.semantic_hash
    metadata = a.semantic_metadata()
    assert metadata['base_hist14'] == resolve_bev_encoding(dict(name='hist14'), G).semantic_metadata()
    assert metadata['geometry_features']['window_cells'] == [11, 21]
    assert 'backend' not in metadata
    for axis in ('x', 'y'):
        with pytest.raises(ValueError, match='0.1'):
            resolve_bev_encoding(dict(name='hist22'), dict(G, **{axis+'_res': .2}))


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
@pytest.mark.parametrize('kind', ['random', 'line', 'plane', 'volume', 'duplicate', 'sparse', 'empty', 'nonfinite', 'strided'])
def test_independent_geometry_and_first14(backend, kind):
    from utils_1.bev_backend import encode_hist14_chw
    rng = np.random.default_rng(54)
    p = rng.uniform([.01,-1.59,-2.49,0], [3.19,1.59,.99,1], (400,4))
    if kind == 'line':
        t = np.linspace(-.3, .3, 40); p = np.column_stack([1.5+t, t, -.7+t, np.ones(40)])
    elif kind == 'plane':
        x,y = np.meshgrid(np.linspace(1.2,1.8,8),np.linspace(-.3,.3,8)); p=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,-.7),np.ones(x.size)])
    elif kind == 'volume':
        x,y,z=np.meshgrid(np.linspace(1.2,1.8,4),np.linspace(-.3,.3,4),np.linspace(-1,-.4,4));p=np.column_stack([x.ravel(),y.ravel(),z.ravel(),np.ones(x.size)])
    elif kind == 'duplicate': p=np.tile([1.25,-.65,-.7,.4],(1001,1))
    elif kind == 'sparse': p=p[:5]
    elif kind == 'empty': p=p[:0]
    elif kind == 'nonfinite': p[:4,np.arange(4)]=np.nan
    elif kind == 'strided': p=p[::2]
    original=p.copy()
    actual=encode(p,backend)
    hist=encode_hist14_chw(p,resolve_bev_encoding(dict(name='hist14'),G))
    if backend == 'numpy': np.testing.assert_array_equal(actual[:14],hist)
    else: np.testing.assert_allclose(actual[:14],hist,atol=1e-6,rtol=1e-5)
    np.testing.assert_allclose(actual[14:],brute_force(p),atol=2e-6,rtol=1e-5)
    assert actual.shape==(22,32,32) and actual.dtype==np.float32
    assert actual.flags.c_contiguous
    # Numba owns allocations through a MemInfo base; no shared ndarray buffer.
    assert actual.flags.owndata or not isinstance(actual.base, np.ndarray)
    assert np.isfinite(actual).all()
    assert not actual[:,actual[11]==0].any()
    np.testing.assert_array_equal(p,original)
    public=encode_bev(p,G,dict(name='hist22',backend=backend))
    assert public.flags.c_contiguous
    np.testing.assert_array_equal(public,actual.transpose(1,2,0))
    later=encode(np.empty((0,4)),backend)
    assert not np.shares_memory(actual,later)


@pytest.mark.parametrize('backend', ['numpy','numba'])
def test_window_edges_count_raw_normalization_and_finite_roi(backend):
    points=np.array([[.025,-1.575,-.7,.2],[.525,-1.575,-.6,.3],[.625,-1.575,-.5,.4],[1.025,-1.575,-.4,.5],[1.125,-1.575,-.3,.6]])
    actual=encode(points,backend,density_norm=2,intensity_scale=2)
    np.testing.assert_allclose(actual[14:],brute_force(points),atol=2e-6,rtol=1e-5)
    assert actual[14,0,0]==np.float32(np.log1p(2))
    assert actual[16,0,0]==np.float32(np.log1p(4))
    assert actual[14:].max()>1
    for axis,k in zip('xyz',range(3)):
        for side,sign in [('min',1),('max',-1)]:
            boundary=G[axis+'_'+side]+sign*.001
            for value in [np.nextafter(boundary,-np.inf),boundary,np.nextafter(boundary,np.inf)]:
                p=points[:1].copy();p[0,k]=value
                np.testing.assert_allclose(encode(p,backend)[14:],brute_force(p),atol=2e-6)
    for edge in [.1,.2,1.2]:
        for value in [np.nextafter(edge,-np.inf),edge,np.nextafter(edge,np.inf)]:
            p=points[:1].copy();p[0,0]=value
            np.testing.assert_allclose(encode(p,backend)[14:],brute_force(p),atol=2e-6)


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_invalid_points(backend):
    for p in [np.zeros((2,3)),np.zeros(4),np.ones((2,4),complex),np.array([['x']*4])]:
        with pytest.raises(ValueError,match='points'): encode(p,backend)


def test_missing_numba_fails_explicitly_even_empty_in_fresh_process():
    script=f"""
import builtins,sys,numpy as np
sys.path[:0] = {[str(ROOT/'detector'),str(ROOT/'detector/core/datasets')]!r}
original=builtins.__import__
def blocked(name,*a,**kw):
    if name=='numba' or name.startswith('numba.'): raise ImportError('blocked')
    return original(name,*a,**kw)
builtins.__import__=blocked
from utils_1.preprocess import encode_bev
for p in [np.empty((0,4)),np.array([[1.,0.,-.7,.2]])]:
    encode_bev(p,{G!r},dict(name='hist22',backend='numpy'))
    try: encode_bev(p,{G!r},dict(name='hist22',backend='numba'))
    except ImportError as e: assert 'Numba' in str(e) and 'backend=numpy' in str(e)
    else: raise AssertionError('missing explicit dependency')
"""
    subprocess.run([sys.executable,'-c',script],check=True,cwd=ROOT)


@pytest.mark.parametrize('backend',['numpy','numba'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_far_roi_duplicates_and_near_degenerate_trace_threshold(backend,dtype):
    geometry=dict(G,x_max=70.4,y_min=-40.,y_max=40.)
    rng=np.random.default_rng(19)
    for position in [[.025,-39.975,-2.475],[69.975,39.975,.975],[35.125,-.625,-.7]]:
        for spread in [0.,1e-7,1e-6,1e-4,.001]:
            p=np.tile([*position,.5],(10001,1))
            if spread:p[:,:3]+=rng.uniform(-spread,spread,(len(p),3))
            p=p.astype(dtype)
            actual=encode(p,backend,geometry)
            expected=brute_force(p,geometry)
            np.testing.assert_allclose(actual[14:],expected,atol=2e-6,rtol=1e-5)


def test_analytic_eigenvalues_match_eigvalsh_repeated_and_extreme_ratios():
    from utils_1.hist22_backend import _eigen_geometry
    rng=np.random.default_rng(123)
    for eig in [[1,1,1],[1,1,0],[1,0,0],[1,1e-8,1e-12],[1,1-1e-12,1e-12],[1e-10,1e-10,0],[0,0,0]]:
        for _ in range(20):
            q,_=np.linalg.qr(rng.normal(size=(3,3)))
            c=(q*np.array(eig))@q.T
            values=np.maximum(np.linalg.eigvalsh(c),0)
            trace=values.sum()
            expected=[0,0,0] if trace<=1e-12 else [1,(values[2]-values[1])/trace,2*(values[1]-values[0])/trace]
            actual=_eigen_geometry(c[0,0],c[0,1],c[0,2],c[1,1],c[1,2],c[2,2])
            np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=1e-5)


@pytest.mark.parametrize('backend',['numpy','numba'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int64])
def test_extra_features_noncontiguous_and_integer_inputs(backend,dtype):
    p=np.zeros((60,7),dtype)
    rng=np.random.default_rng(18)
    p[:,:4]=rng.uniform([.1,-1.5,-2.4,0],[3.1,1.5,.9,1],(len(p),4))
    if dtype != np.int64: p[:,4:]=np.nan
    p=p[::2]
    p.flags.writeable=False
    original=p.copy()
    actual=encode(p,backend)
    expected=encode(p[:,:4].astype(np.float64),'numpy')
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=1e-5)
    np.testing.assert_array_equal(p,original)


def test_numba_has_no_numpy_fallback(monkeypatch):
    import utils_1.hist22_backend as module
    p=np.tile([1.25,-.65,-.7,.4],(12,1))
    expected=encode(p,'numpy')
    def forbidden(*a,**kw): raise AssertionError('NumPy fallback invoked')
    monkeypatch.setattr(module,'_encode_numpy',forbidden)
    np.testing.assert_allclose(encode(p,'numba'),expected,atol=2e-6,rtol=1e-5)


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_independent_outputs_concurrent_calls(backend):
    from concurrent.futures import ThreadPoolExecutor
    clouds=[np.tile([.125+i*.3,-.65,-.7,.4],(i+1,1)) for i in range(8)]
    expected=[encode(p,'numpy') for p in clouds]
    encode(clouds[0],backend)
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual=list(pool.map(lambda p:encode(p,backend),clouds))
    for value,reference in zip(actual,expected):np.testing.assert_allclose(value,reference,atol=2e-6,rtol=1e-5)
    for i,a in enumerate(actual):
        for b in actual[i+1:]:assert not np.shares_memory(a,b)


def test_nonfinite_covariance_raises_clearly():
    from utils_1.hist22_backend import _eigen_geometry
    for bad in [np.nan,np.inf,-np.inf]:
        for i in range(6):
            values=[1.,0.,0.,1.,0.,1.];values[i]=bad
            with pytest.raises(ValueError,match='not finite'):_eigen_geometry(*values)


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_runtime_matches_independent_analysis_prototype(backend):
    import importlib.util
    path=ROOT/'tools/analysis/bev_geometry_maps.py'
    spec=importlib.util.spec_from_file_location('hist22_prototype',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=np.random.default_rng(79).uniform([.002,-1.598,-2.498,0],[3.198,1.598,.998,1],(1500,4)).astype(np.float32)
    expected=module.encode_maps(p,G)['hist22'].transpose(2,0,1)
    actual=encode(p,backend)
    np.testing.assert_array_equal(actual[[11,15,17]],expected[[11,15,17]])
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=1e-5)


def test_numba_cold_first_use_is_thread_safe_in_fresh_process():
    """All workers enter initialization before any compiled dispatcher exists."""
    import os
    script = f"""
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import numpy as np
sys.path[:0] = {[str(ROOT / 'detector'), str(ROOT / 'detector/core/datasets')]!r}
from core.bev_encoding import resolve_bev_encoding
import utils_1.hist22_backend as module
geometry = {G!r}
numpy_schema = resolve_bev_encoding(dict(name='hist22', backend='numpy'), geometry)
numba_schema = resolve_bev_encoding(dict(name='hist22', backend='numba'), geometry)
t = np.linspace(-.025, .025, 32)
clouds = [np.column_stack([.15 + i*.3 + t, -.65+t, -.7+2*t, np.full(len(t), .4)]) for i in range(8)]
expected = [module.encode_hist22_chw(p, numpy_schema) for p in clouds]
assert module._numba_kernel is None
assert 'numba' not in sys.modules
barrier = Barrier(8)
def encode(i):
    barrier.wait(timeout=30)
    return module.encode_hist22_chw(clouds[i], numba_schema)
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(encode, i) for i in range(8)]
    errors = [repr(f.exception()) for f in futures if f.exception() is not None]
    assert not errors, errors
    actual = [f.result() for f in futures]
for value, reference in zip(actual, expected):
    assert value.dtype == np.float32 and value.flags.c_contiguous
    np.testing.assert_allclose(value, reference, atol=2e-6, rtol=1e-5)
for i, a in enumerate(actual):
    for b in actual[i+1:]:
        assert not np.shares_memory(a, b)
"""
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                            env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_numba_initialization_failure_is_retryable_in_fresh_process():
    script = f"""
import builtins, sys
import numpy as np
sys.path[:0] = {[str(ROOT / 'detector'), str(ROOT / 'detector/core/datasets')]!r}
from core.bev_encoding import resolve_bev_encoding
import utils_1.hist22_backend as module
names = ('_eigen_geometry', '_stable_covariance', '_add_moment_vector',
         '_zero_moment_vector', '_prefix_moment_vector', '_difference_moment_vector')
original_helpers = {{name: getattr(module, name) for name in names}}
original_import = builtins.__import__
def blocked(name, *a, **kw):
    if name == 'numba' or name.startswith('numba.'):
        raise ImportError('intentional unavailable dependency')
    return original_import(name, *a, **kw)
builtins.__import__ = blocked
try:
    module._get_numba_kernel()
except ImportError as exc:
    assert 'Numba' in str(exc) and 'backend=numpy' in str(exc)
else:
    raise AssertionError('explicit dependency request must fail')
finally:
    builtins.__import__ = original_import
assert module._numba_kernel is None
assert all(getattr(module, name) is function for name, function in original_helpers.items())
import numba
original_njit = numba.njit
calls = 0
def fail_during_construction(*a, **kw):
    global calls
    calls += 1
    if calls == 3:
        raise RuntimeError('intentional dispatcher construction failure')
    return original_njit(*a, **kw)
numba.njit = fail_during_construction
try:
    module._get_numba_kernel()
except RuntimeError as exc:
    assert 'intentional dispatcher construction failure' in str(exc)
else:
    raise AssertionError('injected construction failure did not happen')
finally:
    numba.njit = original_njit
assert module._numba_kernel is None
assert all(getattr(module, name) is function for name, function in original_helpers.items())
assert not hasattr(module, '_literal_unroll')
schema = resolve_bev_encoding(dict(name='hist22', backend='numba'), {G!r})
points = np.array([[.15, -.65, -.7, .4]])
actual = module.encode_hist22_chw(points, schema)
assert actual[11].sum() == 1 and np.isfinite(actual).all()
kernel = module._get_numba_kernel()
assert module._get_numba_kernel() is kernel
class ForbiddenLock:
    def __enter__(self):
        raise AssertionError('hot encoder must not acquire creation lock')
    def __exit__(self, *a):
        return False
module._numba_initialization_lock = ForbiddenLock()
np.testing.assert_array_equal(module.encode_hist22_chw(points, schema), actual)
"""
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
