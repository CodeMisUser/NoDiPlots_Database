__all__ = ['hilbert', 'clean_scipy_fftpack_cache']

from scipy import fftpack, linalg
from numpy import (allclose, angle, arange, argsort, array, asarray,
                   atleast_1d, atleast_2d, cast, dot, exp, expand_dims,
                   iscomplexobj, mean, ndarray, newaxis, ones, pi,
                   poly, polyadd, polyder, polydiv, polymul, polysub, polyval,
                   prod, product, r_, ravel, real_if_close, reshape,
                   roots, sort, sum, take, transpose, unique, where, zeros,
                   zeros_like)
# import scipy.fftpack._fftpack as sff
import numba


# def clean_scipy_fftpack_cache():
    # sff.destroy_zfft_cache()
    # sff.destroy_zfftnd_cache()
    # sff.destroy_drfft_cache()
    # sff.destroy_cfft_cache()
    # sff.destroy_cfftnd_cache()
    # sff.destroy_rfft_cache()
    # sff.destroy_ddct2_cache()
    # sff.destroy_ddct1_cache()
    # sff.destroy_dct2_cache()
    # sff.destroy_dct1_cache()
    # sff.destroy_ddst2_cache()
    # sff.destroy_ddst1_cache()
    # sff.destroy_dst2_cache()
    # sff.destroy_dst1_cache()


@numba.njit
def _apply_step_function(a):
    N = a.shape[0]

    if N % 2 == 0:
        #h[0] = h[N // 2] = 1
        # No need to modify these coefficients
        #Xf[1 : N//2] = Xf[1 : N//2] * 2
        for i in range(1, N//2):
            a[i] = 2 * a[i]
        # Zero other elements
        for i in range(N//2 + 1, N):
            a[i] = 0
    else:
        #h[0] = 1
        # No need to modify these coefficients
        #Xf[1 : (N+1)//2] = Xf[1 : (N+1)//2] * 2
        for i in range(1, (N+1)//2):
            a[i] = 2 * a[i]
        # Zero other elements
        #Xf[(N+1)//2 : ] = 0
        for i in range((N+1)//2, N):
            a[i] = 0


def hilbert(x, N=None, overwrite_x=False):
    """
    Compute the analytic signal, using the Hilbert transform.
    The transformation is done along the last axis by default.
    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`
    Notes
    -----
    The analytic signal ``x_a(t)`` of signal ``x(t)`` is:
    .. math:: x_a = F^{-1}(F(x) 2U) = x + i y
    where `F` is the Fourier transform, `U` the unit step function,
    and `y` the Hilbert transform of `x`. [1]_
    In other words, the negative half of the frequency spectrum is zeroed
    out, turning the real-valued signal into a complex signal.  The Hilbert
    transformed signal can be obtained from ``np.imag(hilbert(x))``, and the
    original signal from ``np.real(hilbert(x))``.
    """
    
    # The Hilbert transform is extremely slow on some datasets (when length of array is far away from a power of two)
    # To solve this problem, I have added a zero-padding using the fftpack.helper.next_fast_len function to find the best size
    # Before returning x, I truncate back to the original dimension
    
    x = asarray(x)
    dim=x.shape[0]                                             # original dimension of x
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        #N = x.shape[0]
        N=fftpack.helper.next_fast_len(x.shape[0])             # N = best size to perform fast fourier transform
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = fftpack.fft(x, N, overwrite_x=overwrite_x)            # zero padding to the size of N, then performing the fft
    # clean_scipy_fftpack_cache()
    _apply_step_function(Xf)

    x = fftpack.ifft(Xf, overwrite_x=True)                      
    # clean_scipy_fftpack_cache()
    x=x[:dim]                                                  # truncating back to original size dim
    return x


if __name__ == '__main__':
    def test_hilbert_performance(n):
        from time import time
        import numpy as np
        a = np.random.random(n).astype(np.float64)
        start = time()
        res = hilbert(a, overwrite_x=True)
        end = time()
        print("Elapsed time for n={:.3g} {} elements (result as {}): {:.3g} s"
              "".format(n, a.dtype, res.dtype, end - start))
        return end - start

    def test_hilbert():
        import scipy.signal as signal
        import numpy as np
        for _ in range(10):
            # Need to test both even and odd number of elements
            a = np.random.random(1023)
            b = np.random.random(1024)
            #print(np.max(np.abs(hilbert(a) - signal.hilbert(a))))
            assert allclose(hilbert(a), signal.hilbert(a), 1e-15)
            assert allclose(hilbert(b), signal.hilbert(b), 1e-15)

    test_hilbert()
    print(min([test_hilbert_performance(n) for n in ([int(1e6)] * 3)]))
    print(min([test_hilbert_performance(n) for n in ([int(1e8)] * 5)]))
