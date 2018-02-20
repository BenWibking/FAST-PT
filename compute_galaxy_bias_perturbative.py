''' 
Compute perturbative galaxy bias model.
Based on 'fastpt_example_plot.py' included in FAST-PT (commit 19472bf)
Ben Wibking, Feb. 2018
'''

import numpy as np 
import scipy.integrate as integrate
from scipy.interpolate import interp1d 
import FASTPT 
import HT

def j0(x):
    return ( np.sin(x) / x )

def j1(x):
    return ( (np.sin(x)/x**2) - (np.cos(x)/x) )

def bin_avg_spherical_j0(k,rminus,rplus):
    """compute the bin-averaged spherical Bessel function j0."""
    integral = lambda r: r**2 * j1(k*r) / k
    return (3.0 / (rplus**3 - rminus**3)) * (integral(rplus) - integral(rminus))

def xi_fftlog(k,pk):
    alpha_k=1.5
    beta_r=-1.5
    mu=.5 
    pf=(2*np.pi)**(-1.5)
    r, this_xi = HT.k_to_r(k, pk, alpha_k, beta_r, mu, pf)
    return r, this_xi

def xi_simple_binaverage(k_in,pk_in,rmin=0.1,rmax=130.0,nbins=200):
    """compute the integral of bessel function
    over the galaxy power spectrum to obtain the 3d real-space correlation function."""
    bins = np.logspace(np.log10(rmin),np.log10(rmax),nbins+1)
    binmin = bins[:-1]
    binmax = bins[1:]
    bins = zip(binmin, binmax)
    r = 0.5*(binmin+binmax)
    xi = np.empty(binmin.shape[0])

    pk_interp = interp1d(k_in,pk_in)
    super_fac = 16
    k  = np.logspace(np.log10(k_in[0]),np.log10(k_in[-1]),k_in.shape[0]*super_fac)
    pk = pk_interp(k)

    for i, (rminus, rplus) in enumerate(bins):
        # compute signal in bin i on the interval [rminus, rplus)
        y = k**2 / (2.0*np.pi**2) * bin_avg_spherical_j0(k,rminus,rplus) * pk
        result = integrate.simps(y*k, x=np.log(k)) # do integral in d(ln k)
        xi[i] = result

    return r,xi

# load the input power spectrum data 
# (TODO: call pycamb directly)
d=np.loadtxt('Pk_test.dat')
kin=d[:,0]
Pin=d[:,1]

#k = kin
#P = Pin

#from P_extend import k_extend
#extrap = k_extend(kin, high=2.0) # log10
#k = extrap.extrap_k()
#P = extrap.extrap_P_high(Pin)

npoints = 6000
power=interp1d(kin,Pin)
k=np.logspace(np.log10(kin[0]),np.log10(kin[-1]),npoints)
P=power(k)

print('k-points: {}'.format(k.shape[0]))
print('kmin = {}; kmax = {}'.format(np.min(k),np.max(k)))
print('dk/k = {}'.format(np.max(np.diff(k)/k[:-1])))

P_window=np.array([.2,.2])  
C_window=.65	
nu=-2; n_pad=1000
# initialize the FASTPT class	

log_kmin = -5.0
log_kmax = 3.0 # extrapolating too far increases noise in P_{1loop}, thus in P_gg
fastpt=FASTPT.FASTPT(k,nu,low_extrap=log_kmin,high_extrap=log_kmax,n_pad=n_pad,verbose=True) 

P_lin, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4 = fastpt.P_bias(P,C_window=C_window)
# **DO NOT** subtract asymptotic terms according to the user manual
#Pd2d2 -= 2.0*sig4
#Pd2s2 -= (4./3.)*sig4
#Ps2s2 -= (8./9.)*sig4

# now add P_spt to P_lin *after* computing bias terms
P_spt=fastpt.one_loop(P,C_window=C_window) 

def galaxy_power(b1,b2,bs):
    # P+P_spt below should be the full nonlinear power spectrum, in principle
    # (might want to replace it with HALOFIT power spectrum?)
    P_g = (b1**2)*(P+P_spt) + (b1*b2)*Pd1d2 + (1./4.)*(b2**2)*Pd2d2 + (b1*bs)*Pd1s2 + (1./2.)*(b2*bs)*Pd2s2 + (1./4.)*(bs**2)*Ps2s2
    return P_g



import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

fig=plt.figure()

x1=10**(-2.5)
x2=1e2
ax1=fig.add_subplot(111)
ax1.set_ylim(1e-2,1e5)
ax1.set_xlim(x1,x2)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel(r'$P(k)$ [Mpc/$h$]$^3$')
ax1.set_xlabel(r'$k$ [$h$/Mpc]')
ax1.xaxis.set_major_formatter(FormatStrFormatter('%2.2f'))

ax1.plot(k,P+P_spt, label=r'$P_{lin} + P_{SPT}$', color='black')

def plot_galaxy_power(b1,b2,bs):
    P_g = galaxy_power(b1,b2,bs)
    mylabel = r'$P_g(b_1={}, b_2={}, b_s={})$'.format(b1,b2,bs)
    color = next(ax1._get_lines.prop_cycler)['color']
    ax1.plot(k,P_g, label=mylabel, color=color)
    ax1.plot(k,-P_g, '--', label=None, alpha=.5, color=color)

plot_galaxy_power(1.5, 0., -0.1)
plot_galaxy_power(1.5, 0., 0.1)
plot_galaxy_power(1.5, 0.1, 0.)
plot_galaxy_power(1.5, -0.1, 0.)

plt.grid()
plt.legend(loc='best')
plt.tight_layout()
fig.savefig('galaxy_bias.pdf')


## plot correlation functions

def plot_galaxy_correlation_fftlog(b1,b2,bs):
    P_g = galaxy_power(b1,b2,bs)
    r, xi_gg = xi_fftlog(k,P_g)
    mylabel = r'$\xi_g(b_1={}, b_2={}, b_s={})$'.format(b1,b2,bs)
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(r, xi_gg, label=mylabel, color=color)
    ax.plot(r, -xi_gg, '--', label=None, color=color, alpha=.5)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$r$ [Mpc/$h$]')
ax.set_ylabel(r'$\xi(r)$')

r, xi_mm = xi_fftlog(k,P+P_spt)
ax.plot(r, xi_mm, label=r'matter (SPT)', color='black')

plot_galaxy_correlation_fftlog(1.5, 0., -0.1)
plot_galaxy_correlation_fftlog(1.5, 0., 0.1)
plot_galaxy_correlation_fftlog(1.5, 0.1, 0.)
plot_galaxy_correlation_fftlog(1.5, -0.1, 0.)

plt.legend(loc='best')
plt.grid()
plt.tight_layout()
fig.savefig('xi_gg_fftlog.pdf')


## plot (bin-averaged) correlation functions

P_g_linear = galaxy_power(1.5, 0., 0.)
r, xi_gg_linear = xi_simple_binaverage(k,P_g_linear)

def plot_galaxy_correlation_binavg(b1,b2,bs):
    P_g = galaxy_power(b1,b2,bs)
    r, xi_gg = xi_simple_binaverage(k,P_g)
    mylabel = r'$\xi_g(b_1={}, b_2={}, b_s={})$'.format(b1,b2,bs)
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(r, xi_gg/xi_gg_linear, label=mylabel, color=color)
#    ax.plot(r, xi_gg, label=mylabel, color=color)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xlabel(r'$r$ [Mpc/$h$]')
ax.set_ylabel(r'$\xi(r)$')

#r, xi_mm = xi_simple_binaverage(k,P+P_spt)
#ax.plot(r, xi_mm, label=r'matter (SPT)', color='black')

#plot_galaxy_correlation_binavg(1.5, 0., -0.2)
#plot_galaxy_correlation_binavg(1.5, 0., -0.1)
#plot_galaxy_correlation_binavg(1.5, 0., 0.1)
#plot_galaxy_correlation_binavg(1.5, 0., 0.2)
plot_galaxy_correlation_binavg(1.5, 0.3, 0.)
plot_galaxy_correlation_binavg(1.5, 0.2, 0.)
plot_galaxy_correlation_binavg(1.5, 0.1, 0.)
plot_galaxy_correlation_binavg(1.5, -0.1, 0.)
plot_galaxy_correlation_binavg(1.5, -0.2, 0.)
plot_galaxy_correlation_binavg(1.5, -0.3, 0.)

plt.legend(loc='best')
plt.grid()
plt.tight_layout()
fig.savefig('xi_gg_binavg.pdf')
