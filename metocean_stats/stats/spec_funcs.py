import numpy as np
import pandas as pd

def jonswap(f,hs,tp,gamma='fit', sigma_low=.07, sigma_high=.09):
    """
    Purpose:
        To determine spectral density based on the JONSWAP spectrum 

    Input:
        hs  - Significant wave height
        tp  - Spectral peak period
        f   - array of Wave frequency

    Output:
        sf  - Spectral density
    """
    g = 9.82
    fp = 1/tp
    if gamma == 'fit':
       gamma = min(np.exp(3.484*(1-0.1975*(0.036-0.0056*tp/np.sqrt(hs))*(tp**4)/hs**2)),5) # see MET-report_03-2021.pdf, max value should not exceed 5
    else:
       pass
    
    print('gamma-JONSWAP is',gamma)
    alpha  = 5.061*(hs**2/tp**4)*(1-0.287*np.log(gamma)) # see MET-report_03-2021.pdf
    E_pm = alpha*(g**2)*((2*np.pi)**-4)*f**-5*np.exp((-5/4)*((fp/f)**4))
    sigma = np.ones(f.shape)*sigma_low
    sigma[f > 1./tp] = sigma_high
    E_js = E_pm*gamma**np.exp(-0.5*(((f/fp)-1)/sigma)**2)   # see MET-report_03-2021.pdf
    sf = np.nan_to_num(E_js)
    return sf

def velocity_spectrum(f, S_w, depth, ref_depth):
    g = 9.82
    h = depth 
    z = ref_depth
    #k = (1/g)*(2*np.pi/f)**2
    k, ik = waveno(t=1/f,h =depth) 
    k = np.nan_to_num(k)
    G = 2*np.pi *f* np.cosh(k*(depth-ref_depth))/np.sinh(k*ref_depth)
    G = 2*np.pi*f*np.exp(-k*z)*(1+np.exp(-2*(k*h-k*z)))/(1.-np.exp(-2*k*h))
    G = np.nan_to_num(G)
    S_uu = S_w*G**2
    return S_uu



def torsethaugen(f, hs, tp):
    """
    Purpose:
        To determine spectral density based on the Torsethaugen double peaked spectrum 

    Input:
        hs  - Significant wave height
        tp  - Spectral peak period
        f   - array of Wave frequency

    Output:
        sf  - Spectral density
    """
    # Constants
    pi = np.pi
    g = 9.81
    af, ae, au = 6.6, 2.0, 25.0
    a10, a1, rkg = 0.7, 0.5, 35.0
    b1, a20, a2, a3 = 2.0, 0.6, 0.3, 6.0
    g0 = 3.26

    tpf = af * hs ** (1.0 / 3.0)
    tl = ae * hs ** (1.0 / 2.0)
    el = (tpf - tp) / (tpf - tl)

    if tp <= tpf:
        rw = (1.0 - a10) * np.exp(-(el / a1) ** 2) + a10
        hw1 = rw * hs
        tpw1 = tp
        sp = (2.0 * pi / g) * hw1 / tpw1 ** 2
        gam1 = max(1.0, rkg * sp ** (6.0 / 7.0))

        hw2 = np.sqrt(1.0 - rw ** 2) * hs
        tpw2 = tpf + b1
        h1, tp1 = hw1, tpw1
        h2, tp2 = hw2, tpw2
    else:
        tu = au
        eu = (tp - tpf) / (tu - tpf)
        rs = (1.0 - a20) * np.exp(-(eu / a2) ** 2) + a20
        hs1 = rs * hs
        tps1 = tp
        sf = (2.0 * pi / g) * hs / tpf ** 2
        gam1 = max(1.0, rkg * sf ** (6.0 / 7.0) * (1.0 + a3 * eu))
        hs2 = np.sqrt(1.0 - rs ** 2) * hs
        tps2 = af * hs2 ** (1.0 / 3.0)
        h1, tp1 = hs1, tps1
        h2, tp2 = hs2, tps2

    e1 = (1.0 / 16.0) * (h1 ** 2) * tp1
    e2 = (1.0 / 16.0) * (h2 ** 2) * tp2
    ag = (1.0 + 1.1 * np.log(gam1) ** 1.19) / gam1

    f1n, f2n = f * tp1, f * tp2
    
    sigma1 = np.where(f1n > 1.0, 0.09, 0.07)
    
    fnc1 = f1n ** (-4) * np.exp(-f1n ** (-4))
    fnc2 = gam1 ** (np.exp(-1.0 / (2.0 * sigma1 ** 2) * (f1n - 1.0) ** 2))
    s1 = g0 * ag * fnc1 * fnc2

    fnc3 = f2n ** (-4) * np.exp(-f2n ** (-4))
    s2 = g0 * fnc3

    sf = e1 * s1 + e2 * s2
    return np.nan_to_num(sf)


def waveno(t, h):
    """
    Purpose:
        To compute wave number

    Input:
        t  - Wave period
        h  - Water depth (can be an array or a single value)

    Output:
        k - Wave number 
        nier - Negative depth values: nier = 1
    """
    # Set value of constants
    g = 9.82
    A = [0.66667, 0.35550, 0.16084, 0.06320, 0.02174, 0.00654, 0.00171, 0.00039, 0.00011]

    # Initialize output
    nier = 0
    sigma = 2.0 * np.pi / t

    # Function to compute wave number for a single depth
    def compute_k(depth):
        nonlocal nier
        b = g * depth
        if b < 0:
            nier = 1
            return 0.0
        y = sigma * sigma * depth / g
        x = A[4] + y * (A[5] + y * (A[6] + y * (A[7] + y * A[8])))
        x = 1.0 + y * (A[0] + y * (A[1] + y * (A[2] + y * (A[3] + y * x))))
        c = np.sqrt(b / (y + 1.0 / x))
        return sigma / c

    # Check if h is a single value or a list
    if isinstance(h, (int, float)):
        k = compute_k(h)
    else:
        k = [compute_k(depth) for depth in h]

    return k, nier
