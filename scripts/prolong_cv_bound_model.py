"""Why a coarse-CV-range bound cannot limit the p >= 1 AMR prolongation.

Models one coarse element and its two children on the PRODUCTION node sets
(solution_points -> Chebyshev-Gauss x_sp, flux_points -> Gauss x_fp), and asks
how many of the 2(p+1) fine control-volume averages fall outside the range of
the coarse element's own p+1 control-volume averages -- the bound proposed as
the escalation after limit_prolongation_dmp's group-mean bound over-clipped.

The answer is 2 of 8 for ANY data with a gradient, including pure linear data,
because the outermost fine CV averages over a narrower interval nearer the
element edge than the outermost coarse CV does. The count is set by the node
geometry, not by smoothness, so the bound is not exact on linear data and would
cost the scheme its order of accuracy. See the comment on
limit_prolongation_dmp in src/amr.cpp.

Bounds are widened to admit the child's own mean, exactly as that kernel does.
"""

import numpy as np

p = 3; n = p+1
# solution_points(): Chebyshev-Gauss on [0,1]
x_sp = np.array([0.5*(1.0-np.cos((2*i+1)/(2.0*(p+1))*np.pi)) for i in range(n)])
# flux_points(): [0, gauss_legendre(0,1,p) nodes, 1]
gl, _ = np.polynomial.legendre.leggauss(p)
gl = 0.5*(gl+1.0)
x_fp = np.concatenate(([0.0], gl, [1.0]))

def phi(l, xi):                      # coarse Lagrange basis on x_sp
    r = np.ones_like(np.asarray(xi, dtype=float))
    for j in range(n):
        if j != l: r *= (xi - x_sp[j])/(x_sp[l]-x_sp[j])
    return r

def u_of(c, xi):                     # coarse element polynomial
    return sum(c[l]*phi(l, xi) for l in range(n))

def avg(c, a, b):                    # exact average of u over [a,b] (GL, deg p exact)
    q, w = np.polynomial.legendre.leggauss(p+2)
    xi = 0.5*(b-a)*q + 0.5*(a+b)
    return float(np.sum(w*u_of(c, xi))*0.5)

def coarse_cvs(c):
    return np.array([avg(c, x_fp[k], x_fp[k+1]) for k in range(n)])

def fine_cvs(c, s):                  # child s occupies xi in [s/2,(s+1)/2]
    return np.array([avg(c, (s+x_fp[k])/2.0, (s+x_fp[k+1])/2.0) for k in range(n)])

def report(name, f):
    c = f(x_sp)                      # coarse SP values = sample f at coarse SPs
    ccv = coarse_cvs(c); lo, hi = ccv.min(), ccv.max()
    out = 0; tot = 0; worst = 0.0
    for s in (0, 1):
        fcv = fine_cvs(c, s); mean = avg(c, s/2.0, (s+1)/2.0)
        # bounds widened to admit the child's own mean, as limit_prolongation_dmp does
        blo, bhi = min(lo, mean), max(hi, mean)
        for v in fcv:
            tot += 1
            d = max(blo - v, v - bhi)
            if d > 1e-14: out += 1; worst = max(worst, d)
    print(f"{name:34s} fine CVs outside coarse CV range: {out}/{tot}   worst excursion {worst:.3e}")

report("u = xi   (pure linear)",        lambda xi: xi)
report("u = xi^2",                      lambda xi: xi**2)
report("sine, 16 coarse elems/wavelen", lambda xi: np.sin(2*np.pi*xi/16.0))
report("sine, 64 coarse elems/wavelen", lambda xi: np.sin(2*np.pi*xi/64.0))
report("u = const",                     lambda xi: np.ones_like(xi))

print()
print("x_sp =", np.round(x_sp, 6))
print("x_fp =", np.round(x_fp, 6))
c = x_sp.copy()   # u = xi
print("u=xi: coarse CVs =", np.round(coarse_cvs(c), 6))
print("u=xi: child-0 fine CVs =", np.round(fine_cvs(c, 0), 6), " child-0 mean =", round(avg(c,0,0.5),6))
