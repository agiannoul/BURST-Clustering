import numpy as np
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft
import pycatch22


def Catch22Transformation(X):
    Xcatch22=[]
    for timeseries in X:
        Xcatch22.append(pycatch22.catch22_all(timeseries)['values'])
    Xcatch22=np.array(Xcatch22)
    return Xcatch22
def DefaultTransformation(timeseries):
    return timeseries

def get_transformation(distance_measure,transformation):
    if distance_measure == 'euclidean' and transformation == 'catch22':
        return Catch22Transformation
    return DefaultTransformation

def get_distance_measure(distance_measure):
    if distance_measure == 'euclidean':
        return euclidean
    elif distance_measure == 'zscore_euclidean':
        return zscore_euclidean
    elif distance_measure == 'SBD':
        return sbd
    raise ValueError(f'Not found distance measure: {distance_measure}')




def euclidean(a,b):
    return np.linalg.norm(a - b)
def zscore_euclidean(a,b):
    return np.linalg.norm(_zscore(a, ddof=1) - _zscore(b, ddof=1))
def sbd(a,b):
    return _sbd(_zscore(a, ddof=1),_zscore(b, ddof=1))



def _sbd(x, y):
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    return dist

def _ncc_c(x, y):
    den = np.array(norm(x) * norm(y))
    try:
        den[den == 0] = np.inf
    except Exception as e:
        print(e)
        den[den == 0] = np.Inf
    x_len = len(x)
    fft_size = 1 << (2 * x_len - 1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[-(x_len - 1):], cc[:x_len]))
    return np.real(cc) / den

def _zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd
    return np.nan_to_num(res)