"""CANUPO-inspired Hist22 v1, with explicit NumPy and compiled CPU backends.

Both paths pool ten float64 XYZ moments about the ROI midpoint. A two-pass
local covariance calculation handles near-zero variance without cancellation.
Scratch arrays belong to one invocation, including under DataLoader workers.
"""
from __future__ import annotations

import math
from threading import Lock
import numpy as np

from .bev_backend import _encode_hist14_numpy

HIST22_IMPLEMENTATION_VERSION = 1
_MOMENT_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
_numba_kernel = None
_numba_initialization_lock = Lock()


def _base_schema(schema):
    try:
        from core.bev_encoding import resolve_bev_encoding
    except ImportError:
        from detector.core.bev_encoding import resolve_bev_encoding
    return resolve_bev_encoding(dict(name='hist14', backend=schema.backend,
                                    density_norm=schema.density_norm,
                                    intensity_scale=schema.intensity_scale), dict(schema.geometry))


def _validated_points(points, schema):
    if schema.name != 'hist22' or schema.version != 1 or schema.channels != 22:
        raise ValueError('hist22 rasterization requires a hist22 v1 schema')
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] < 4 or p.dtype.kind not in 'fiu':
        raise ValueError('hist22 expects real numeric points shaped (N, >=4)')
    if p.dtype not in (np.dtype('float32'), np.dtype('float64')):
        p = np.ascontiguousarray(p[:, :4], dtype=np.float64)
    return p[:, :4]


def encode_hist22(points, schema):
    """Compatibility API returning independently owned contiguous [Y,X,22]."""
    return encode_hist22_chw(points, schema).transpose(1, 2, 0).copy()


def encode_hist22_chw(points, schema):
    """Return independently owned contiguous float32 [22,Y,X] model input."""
    p = _validated_points(points, schema)
    if schema.backend == 'numpy':
        return _encode_numpy(p, schema)
    if schema.backend == 'numba':
        kernel = _get_numba_kernel()  # Dependency checked even for empty clouds.
        g = dict(schema.geometry)
        # NumPy calloc avoids Numba eagerly touching every page of this 47MiB
        # output. The compiled kernel writes occupied cells directly into CHW.
        out = np.zeros((22, schema.grid_shape[1], schema.grid_shape[0]), np.float32)
        return kernel(p, out, schema.grid_shape[0], schema.grid_shape[1],
                      g['x_min'], g['x_max'], g['x_res'], g['y_min'], g['y_max'], g['y_res'],
                      g['z_min'], g['z_max'], schema.density_norm, schema.intensity_scale)
    raise ValueError(f'unsupported hist22 backend: {schema.backend!r}')


def encode_hist22_numpy(points, schema):
    """NumPy reference in CHW layout, independent of schema backend provenance."""
    return _encode_numpy(_validated_points(points, schema), schema)


def _encode_numpy(points, schema):
    w, h, _ = schema.grid_shape
    out = np.zeros((22, h, w), np.float32)
    # Keep exact reference arithmetic and first-14 rounding, independently of
    # the grouping/pooling used for the eight additional channels.
    out[:14] = _encode_hist14_numpy(points, _base_schema(schema), channels_first=True)
    occupied_y, occupied_x = np.nonzero(out[11])
    if not len(occupied_x):
        return out
    p = np.asarray(points, np.float64)
    g = dict(schema.geometry)
    keep = np.isfinite(p).all(axis=1)
    for k, axis in enumerate('xyz'):
        keep &= (p[:, k] > g[axis+'_min']+.001) & (p[:, k] < g[axis+'_max']-.001)
    p = p[keep, :3]
    ix = np.floor((p[:, 0]-g['x_min'])/g['x_res']).astype(np.int64)
    iy = np.floor((p[:, 1]-g['y_min'])/g['y_res']).astype(np.int64)
    flat = iy*w+ix
    q = p-np.array([(g[a+'_min']+g[a+'_max'])/2 for a in 'xyz'])
    x,y,z = q.T
    values = (np.ones(len(p)), x,y,z,x*x,x*y,x*z,y*y,y*z,z*z)
    tables = np.zeros((h+1,w+1,10), np.float64)
    for k,value in enumerate(values):
        tables[1:,1:,k] = np.bincount(flat, weights=value, minlength=h*w).reshape(h,w)
    np.cumsum(tables, axis=1, out=tables)
    np.cumsum(tables, axis=0, out=tables)
    for scale,radius in enumerate((5,10)):
        lo_y,hi_y=np.maximum(0,occupied_y-radius),np.minimum(h,occupied_y+radius+1)
        lo_x,hi_x=np.maximum(0,occupied_x-radius),np.minimum(w,occupied_x+radius+1)
        m=tables[hi_y,hi_x]-tables[lo_y,hi_x]-tables[hi_y,lo_x]+tables[lo_y,lo_x]
        n=m[:,0]
        out[14+2*scale,occupied_y,occupied_x]=np.log1p(n)
        valid=n>=6
        if not valid.any():
            continue
        slots=np.flatnonzero(valid)
        means=m[slots,1:4]/n[slots,None]
        cov=np.empty((len(slots),3,3),np.float64)
        for row,col,k in ((0,0,4),(0,1,5),(0,2,6),(1,1,7),(1,2,8),(2,2,9)):
            cov[:,row,col]=cov[:,col,row]=m[slots,k]/n[slots]-means[:,row]*means[:,col]
        # Moment subtraction is ill-conditioned for almost coincident points.
        # Re-evaluate only those neighborhoods about their actual centroid.
        # 1e-4 chooses a stable arithmetic path; validity still uses 1e-12.
        near=np.flatnonzero(np.abs(np.trace(cov,axis1=1,axis2=2))<1e-4)
        for j in near:
            slot=slots[j]
            mask=(np.abs(ix-occupied_x[slot])<=radius)&(np.abs(iy-occupied_y[slot])<=radius)
            local=p[mask]
            centered=local-local[0]
            centered-=centered.mean(axis=0)
            cov[j]=centered.T@centered/len(local)
        eig=np.maximum(np.linalg.eigvalsh(cov),0)
        if not np.isfinite(eig).all():
            raise ValueError('hist22 covariance eigenvalues are not finite')
        trace=eig.sum(axis=1)
        accepted=trace>1e-12
        slots=slots[accepted]
        e=eig[accepted];t=trace[accepted]
        ys,xs=occupied_y[slots],occupied_x[slots]
        out[15+2*scale,ys,xs]=1
        out[18+2*scale,ys,xs]=np.clip((e[:,2]-e[:,1])/t,0,1)
        out[19+2*scale,ys,xs]=np.clip(2*(e[:,1]-e[:,0])/t,0,1)
    return out


def _eigen_geometry(xx,xy,xz,yy,yz,zz):
    """Stable closed form for a real symmetric 3x3, including repeated roots."""
    if not (math.isfinite(xx) and math.isfinite(xy) and math.isfinite(xz)
            and math.isfinite(yy) and math.isfinite(yz) and math.isfinite(zz)):
        raise ValueError("hist22 covariance eigenvalues are not finite")
    q=(xx+yy+zz)/3.
    aa,bb,cc=xx-q,yy-q,zz-q
    p2=aa*aa+bb*bb+cc*cc+2*(xy*xy+xz*xz+yz*yz)
    if p2 == 0.:
        l1=l2=l3=max(0.,q)
    else:
        p=math.sqrt(p2/6.)
        a,b,c=aa/p,bb/p,cc/p
        d,e,f=xy/p,xz/p,yz/p
        r=(a*(b*c-f*f)-d*(d*c-f*e)+e*(d*f-b*e))/2.
        phi=math.acos(min(1.,max(-1.,r)))/3.
        cosine=math.cos(phi)
        sine=math.sqrt(max(0.,1.-cosine*cosine))
        l1=q+2*p*cosine
        l3=q-p*(cosine+math.sqrt(3.)*sine)
        l2=q-p*(cosine-math.sqrt(3.)*sine)
        l1=max(0.,l1);l2=max(0.,l2);l3=max(0.,l3)
    if not (math.isfinite(l1) and math.isfinite(l2) and math.isfinite(l3)):
        raise ValueError('hist22 covariance eigenvalues are not finite')
    trace=l1+l2+l3
    if trace<=1e-12:
        return 0.,0.,0.
    return 1.,min(1.,max(0.,(l1-l2)/trace)),min(1.,max(0.,2*(l2-l3)/trace))


def _stable_covariance(points, head, next_point, ix, iy, radius, w, h):
    """Two-pass covariance for near-degenerate neighborhoods, without scans of unrelated points."""
    mean=np.zeros(3,np.float64)
    anchor=np.zeros(3,np.float64)
    count=0
    for y in range(max(0,iy-radius),min(h,iy+radius+1)):
        for x in range(max(0,ix-radius),min(w,ix+radius+1)):
            j=head[y*w+x]
            while j>=0:
                if count==0:
                    for k in range(3): anchor[k]=np.float64(points[j,k])
                for k in range(3): mean[k]+=np.float64(points[j,k])-anchor[k]
                count+=1
                j=next_point[j]
    mean/=count
    m=np.zeros(6,np.float64)
    for y in range(max(0,iy-radius),min(h,iy+radius+1)):
        for x in range(max(0,ix-radius),min(w,ix+radius+1)):
            j=head[y*w+x]
            while j>=0:
                a=np.float64(points[j,0])-anchor[0]-mean[0]
                b=np.float64(points[j,1])-anchor[1]-mean[1]
                c=np.float64(points[j,2])-anchor[2]-mean[2]
                m[0]+=a*a;m[1]+=a*b;m[2]+=a*c;m[3]+=b*b;m[4]+=b*c;m[5]+=c*c
                j=next_point[j]
    return m/count



def _add_moment_vector(destination, source, sign):
    for k in _literal_unroll(_MOMENT_INDICES):
        destination[k] += sign * source[k]


def _zero_moment_vector(destination):
    for k in _literal_unroll(_MOMENT_INDICES):
        destination[k] = 0.


def _prefix_moment_vector(destination, previous, source):
    for k in _literal_unroll(_MOMENT_INDICES):
        destination[k] = previous[k] + source[k]


def _difference_moment_vector(destination, upper, lower):
    for k in _literal_unroll(_MOMENT_INDICES):
        destination[k] = upper[k] - lower[k]


def _accumulate_hist22(
    points, output, x_size, y_size, x_min, x_max, x_res, y_min, y_max, y_res,
    z_min, z_max, density_norm, intensity_scale,
):
    """One point pass and one occupied-cell pass, with float64 arithmetic.

    A zero-initialized grid maps each cell to a compact accumulator row plus one.
    Empty rows are never touched. Workspace and output are local to each call.
    """
    cell_count = x_size * y_size
    capacity = min(points.shape[0], cell_count)
    lookup = np.zeros(cell_count, dtype=np.int32)
    cells = np.empty(capacity, dtype=np.int64)
    # No accumulator for the constant occupancy feature; offsets use cols 11/12.
    accum = np.empty((capacity, 23), dtype=np.float64)
    occupied_count = 0
    flat_output=output.reshape(22,cell_count)
    head=np.full(cell_count,-1,np.int64)
    next_point=np.empty(points.shape[0],np.int64)
    origin_x=(x_min+x_max)/2
    origin_y=(y_min+y_max)/2
    origin_z=(z_min+z_max)/2
    z_span = z_max - z_min
    density_log = math.log1p(density_norm)
    # Integer counts recur across cells/bins. Evaluate log1p once per small count
    # without changing the original double-precision expression or its rounding.
    table_size = min(points.shape[0] + 1, 256)
    density_table = np.empty(table_size, dtype=np.float64)
    for count in range(table_size):
        density_table[count] = min(1.0, math.log1p(count) / density_log)
    for index in range(points.shape[0]):
        x = np.float64(points[index, 0])
        y = np.float64(points[index, 1])
        z = np.float64(points[index, 2])
        intensity = np.float64(points[index, 3])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and math.isfinite(intensity)):
            continue
        if not (x > x_min + .001 and x < x_max - .001 and
                y > y_min + .001 and y < y_max - .001 and
                z > z_min + .001 and z < z_max - .001):
            continue
        ix = int(math.floor((x - x_min) / x_res))
        iy = int(math.floor((y - y_min) / y_res))
        if ix < 0 or ix >= x_size or iy < 0 or iy >= y_size:
            raise ValueError("hist14 point lies outside the resolved grid; check geometry divisibility")
        flat = iy * x_size + ix
        next_point[index]=head[flat]
        head[flat]=index
        row = lookup[flat] - 1
        if row < 0:
            row = occupied_count
            occupied_count += 1
            lookup[flat] = row + 1
            cells[row] = flat
            for column in range(23):
                accum[row, column] = 0.0
        a,b,c=x-origin_x,y-origin_y,z-origin_z
        accum[row,13]+=1
        accum[row,14]+=a
        accum[row,15]+=b
        accum[row,16]+=c
        accum[row,17]+=a*a
        accum[row,18]+=a*b
        accum[row,19]+=a*c
        accum[row,20]+=b*b
        accum[row,21]+=b*c
        accum[row,22]+=c*c
        height = (z - z_min) / z_span
        band = min(int(math.floor(4 * height)), 3)
        value = min(1.0, max(0.0, intensity * intensity_scale))
        if accum[row, 10] == 0:
            accum[row, 6] = height
        accum[row, band] += 1
        accum[row, 4] = max(accum[row, 4], height)
        accum[row, 5] += height
        accum[row, 6] = min(accum[row, 6], height)
        accum[row, 7] += height * height
        accum[row, 8] = max(accum[row, 8], value)
        accum[row, 9] += value
        accum[row, 10] += 1
        accum[row, 11] += (x - (x_min + (ix + .5) * x_res)) / x_res
        accum[row, 12] += (y - (y_min + (iy + .5) * y_res)) / y_res

    for row in range(occupied_count):
        flat = cells[row]
        count = accum[row, 10]
        for band in range(4):
            bin_count = int(accum[row, band])
            flat_output[band, flat] = (density_table[bin_count] if bin_count < table_size
                                         else min(1.0, math.log1p(bin_count) / density_log))
        mean_height = accum[row, 5] / count
        variance = max(0.0, accum[row, 7] / count - mean_height * mean_height)
        flat_output[4, flat] = accum[row, 4]
        flat_output[5, flat] = mean_height
        flat_output[6, flat] = accum[row, 4] - accum[row, 6]
        flat_output[7, flat] = math.sqrt(variance) if count > 1 else 0.0
        flat_output[8, flat] = accum[row, 8]
        flat_output[9, flat] = accum[row, 9] / count
        total_count = int(count)
        flat_output[10, flat] = (density_table[total_count] if total_count < table_size
                                   else min(1.0, math.log1p(total_count) / density_log))
        flat_output[11, flat] = 1
        flat_output[12, flat] = min(.5, max(-.5, accum[row, 11] / count))
        flat_output[13, flat] = min(.5, max(-.5, accum[row, 12] / count))
    if occupied_count == 0:
        return output
    # Sweep two vertical windows, then prefix-sum occupied strip columns only.
    # No dense float64 moment grid is allocated; both strips fit in L2 cache.
    starts=np.empty(y_size+1,np.int64)
    sorted_rows=np.empty(occupied_count,np.int64)
    current=0
    for y in range(y_size):
        starts[y]=current
        for x in range(x_size):
            flat=y*x_size+x
            row=lookup[flat]-1
            if row>=0:
                sorted_rows[current]=row
                cells[current]=flat
                current+=1
    starts[y_size]=current
    strip=np.zeros((2,x_size,10),np.float64)
    prefix=np.empty((2,x_size+1,10),np.float64)
    prefix_lookup=np.empty((2,x_size),np.int64)
    for scale in range(2):
        radius=5 if scale==0 else 10
        for y in range(min(y_size,radius+1)):
            for j in range(starts[y],starts[y+1]):
                row=sorted_rows[j];x=cells[j]%x_size
                _add_moment_vector(strip[scale,x], accum[row,13:23], 1.)
    m=np.empty(10,np.float64)
    for iy in range(y_size):
        if starts[iy]<starts[iy+1]:
            for scale in range(2):
                _zero_moment_vector(prefix[scale,0])
                current=0
                for x in range(x_size):
                    if strip[scale,x,0]>0:
                        _prefix_moment_vector(prefix[scale,current+1], prefix[scale,current], strip[scale,x])
                        current+=1
                    prefix_lookup[scale,x]=current
            for j in range(starts[iy],starts[iy+1]):
                flat=cells[j];ix=flat%x_size
                for scale in range(2):
                    radius=5 if scale==0 else 10
                    xl=max(0,ix-radius);xh=min(x_size,ix+radius+1)
                    lo=prefix_lookup[scale,xl-1] if xl>0 else 0
                    hi=prefix_lookup[scale,xh-1]
                    _difference_moment_vector(m, prefix[scale,hi], prefix[scale,lo])
                    n=m[0]
                    flat_output[14+2*scale,flat]=math.log1p(n)
                    if n<6:
                        continue
                    a,b,c=m[1]/n,m[2]/n,m[3]/n
                    xx=m[4]/n-a*a;xy=m[5]/n-a*b;xz=m[6]/n-a*c
                    yy=m[7]/n-b*b;yz=m[8]/n-b*c;zz=m[9]/n-c*c
                    if abs(xx+yy+zz)<1e-4:
                        stable=_stable_covariance(points,head,next_point,ix,iy,radius,x_size,y_size)
                        xx,xy,xz,yy,yz,zz=stable
                    valid,a,b=_eigen_geometry(xx,xy,xz,yy,yz,zz)
                    flat_output[15+2*scale,flat]=valid
                    flat_output[18+2*scale,flat]=a
                    flat_output[19+2*scale,flat]=b
        for scale in range(2):
            radius=5 if scale==0 else 10
            entering=iy+radius+1
            leaving=iy-radius
            if entering<y_size:
                for j in range(starts[entering],starts[entering+1]):
                    row=sorted_rows[j];x=cells[j]%x_size
                    _add_moment_vector(strip[scale,x], accum[row,13:23], 1.)
            if leaving>=0:
                for j in range(starts[leaving],starts[leaving+1]):
                    row=sorted_rows[j];x=cells[j]%x_size
                    _add_moment_vector(strip[scale,x], accum[row,13:23], -1.)
    return output


# Preserve the Python definitions for retries. Building dispatchers must never
# redecorate partially published CPUDispatcher objects after an earlier error.
_NUMBA_PYTHON_HELPERS = (
    ("_eigen_geometry", _eigen_geometry, False),
    ("_stable_covariance", _stable_covariance, False),
    ("_add_moment_vector", _add_moment_vector, True),
    ("_zero_moment_vector", _zero_moment_vector, True),
    ("_prefix_moment_vector", _prefix_moment_vector, True),
    ("_difference_moment_vector", _difference_moment_vector, True),
)


def _get_numba_kernel():
    global _numba_kernel
    # Encoding after initialization takes no lock, including across workers.
    if _numba_kernel is not None:
        return _numba_kernel
    with _numba_initialization_lock:
        if _numba_kernel is not None:
            return _numba_kernel
        try:
            from numba import njit, literal_unroll
        except ImportError as exc:
            raise ImportError(
                'hist22 backend=numba requires Numba. Install a compatible Numba '
                'version or select backend=numpy.'
            ) from exc
        compiled_helpers = {}
        for name, python_function, inline in _NUMBA_PYTHON_HELPERS:
            options = dict(cache=True, fastmath=False, nogil=True)
            if inline:
                options['inline'] = 'always'
            compiled_helpers[name] = njit(**options)(python_function)
        kernel = njit(cache=True, fastmath=False, nogil=True)(_accumulate_hist22)
        # Publish dependencies only after every decorator succeeds, and publish
        # the kernel last so no encoder can observe incomplete initialization.
        compiled_helpers['_literal_unroll'] = literal_unroll
        globals().update(compiled_helpers)
        _numba_kernel = kernel
        return kernel
