from os import system as sys
import fileinput as fi
from numpy.core.multiarray import dtype
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from numpy import pi, exp, log, sin, cos, tan, log10, abs, gradient, sinh, cosh, arctan, argmax, argmin
from scipy.special import hyp2f1, erf, erfc, erfcinv, gammainc, erfi, spence, expit, gamma
from scipy.optimize import fsolve
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse import linalg as las
from math import ceil, floor
import time
import csv
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as ticker
import re
import warnings

import os
from multiprocessing import Pool
from sklearn.linear_model import LinearRegression

def multiple_workers(foo, inputs):
    pool = Pool(processes=32)
    outputs = pool.map(foo, inputs)
    return outputs

# constants
kB = 8.617e-5 #eV/K
q = 1.6e-19 #C
mo = 9.11e-31 #kg
hred = 1.054571817e-34
h = hred*2*pi
epso = 8.85e-12

# plotting section
SMALL_SIZE = 22
MEDIUM_SIZE = 26
BIGGER_SIZE = 32

plt.rc('font', size=SMALL_SIZE, weight='bold')          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["figure.figsize"] = (12, 10)
plt.rcParams['axes.linewidth'] = 2.5
# plt.rc('text', usetex=True)
# plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def logy_lin_plot_dual(x1, y1, x2, y2, c1, c2, s1, s2, a1=[1], a2=[1], mask=[True], m1=[None], m2=[None], l=[None], cmap=False, xlim=0, ylim=0, lw=2.0, xlabel='', ylabel='', title='', figname='temp1.png', ms=12):
    fig, ax = plt.subplots()
    ax_twin = ax.twinx()
    for i in range(len(x1)):
        if mask[i%len(mask)]:
            ax.semilogy(x1[i], y1[i], linewidth=lw, marker=m1[i%len(m1)], color=c1[i%len(c1)], linestyle=s1[i%len(s1)], markersize=ms, alpha=a1[i%len(a1)])
    
    for i in range(len(x2)):
        if mask[i%len(mask)]:
            ax_twin.plot(x2[i], y2[i], linewidth=lw, marker=m2[i%len(m2)], color=c2[i%len(c2)], linestyle=s2[i%len(s2)], markersize=ms, alpha=a2[i%len(a2)])

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax_twin.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.xlabel(xlabel=xlabel, fontweight='bold')
    plt.ylabel(ylabel=ylabel, fontweight='bold')
    plt.title(label=title, fontweight='bold')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    labels = ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels()
    [label.set_fontweight('bold') for label in labels]
    ax.minorticks_on()
    labels = ax_twin.xaxis.get_ticklabels() + ax_twin.yaxis.get_ticklabels()
    [label.set_fontweight('bold') for label in labels]
    ax_twin.minorticks_on()
    # ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=5))
    plt.tick_params(axis='y', which='minor')
    ax.tick_params(direction='in', length=6, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='major')
    ax.tick_params(direction='in', length=3, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='minor')
    ax_twin.tick_params(direction='in', length=6, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='major')
    ax_twin.tick_params(direction='in', length=3, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='minor')
    if not l == [None]:
        ax.legend(l)
    if not xlim == 0:
        ax.set(xlim=xlim)
    if not ylim == 0:
        ax.set(ylim=ylim)
    # plt.figure(figsize=(1,1))
    plt.grid()
    # plt.show(block=False)
    plt.savefig(fname=figname)

def lin_plot(x, y, c, s, mask=[True], m=[None], l=[None], a=[1], xlim=0, ylim=0, lw=2.0, xlabel='', ylabel='', title='', figname='temp.png', ms=12):
    fig, ax = plt.subplots()
    for i in range(len(x)):
        if mask[i%len(mask)]:
            ax.plot(x[i], y[i], linewidth=lw, marker=m[i%len(m)], color=c[i%len(c)], linestyle=s[i%len(s)], markersize=ms, alpha=a[i%len(a)])
    # ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.xlabel(xlabel=xlabel, fontweight='bold')
    plt.ylabel(ylabel=ylabel, fontweight='bold')
    plt.title(label=title, fontweight='bold')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    labels = ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels()
    [label.set_fontweight('bold') for label in labels]
    ax.minorticks_on()
    ax.tick_params(direction='in', length=6, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='major')
    ax.tick_params(direction='in', length=3, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='minor')
    if not l == [None]:
        ax.legend(l)
    if not xlim == 0:
        ax.set(xlim=xlim)
    if not ylim == 0:
        ax.set(ylim=ylim)
    # plt.figure(figsize=(1,1))
    plt.grid()
    # plt.show(block=False)
    plt.savefig(fname=figname)

def logx_plot(x, y, c, s, mask=[True], m=[None], l=[None], a=[1], cmap=False, xlim=0, ylim=0, lw=2.0, xlabel='', ylabel='', title='', figname='temp.png', ms=12):
    fig, ax = plt.subplots()
    for i in range(len(x)):
        if mask[i%len(mask)]:
            ax.semilogx(x[i], y[i], linewidth=lw, marker=m[i%len(m)], color=c[i%len(c)], linestyle=s[i%len(s)], markersize=ms, alpha=a[i%len(a)])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    # ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.xlabel(xlabel=xlabel, fontweight='bold')
    plt.ylabel(ylabel=ylabel, fontweight='bold')
    plt.title(label=title, fontweight='bold')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    labels = ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels()
    [label.set_fontweight('bold') for label in labels]
    ax.minorticks_on()
    ax.tick_params(direction='in', length=6, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='major')
    ax.tick_params(direction='in', length=3, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='minor')
    if not l == [None]:
        ax.legend(l)
    if not xlim == 0:
        ax.set(xlim=xlim)
    if not ylim == 0:
        ax.set(ylim=ylim)
    # plt.figure(figsize=(1, 1))
    plt.grid()
    # plt.show(block=False)
    plt.savefig(fname=figname)

def logy_plot(x, y, c, s, mask=[True], m=[None], l=[None], a=[1], cmap=False, xlim=0, ylim=0, lw=2.0, xlabel='', ylabel='', title='', figname='temp.png', ms=12, savemode=True):
    fig, ax = plt.subplots()
    for i in range(len(x)):
        if mask[i%len(mask)]:
            ax.semilogy(x[i], y[i], linewidth=lw, marker=m[i%len(m)], color=c[i%len(c)], linestyle=s[i%len(s)], markersize=ms, alpha=a[i%len(a)])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    # ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.xlabel(xlabel=xlabel, fontweight='bold')
    plt.ylabel(ylabel=ylabel, fontweight='bold')
    plt.title(label=title, fontweight='bold')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    labels = ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels()
    [label.set_fontweight('bold') for label in labels]
    ax.minorticks_on()
    ax.tick_params(direction='in', length=6, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='major')
    ax.tick_params(direction='in', length=3, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='minor')
    if not l == [None]:
        ax.legend(l)
    if not xlim == 0:
        ax.set(xlim=xlim)
    if not ylim == 0:
        ax.set(ylim=ylim)
    # plt.figure(figsize=(1, 1))
    plt.grid()
    if savemode:
        plt.savefig(fname=figname)
    else:
        plt.show(block=False)

def logy_lin_plot(x, y, c, s, mask=[True], m=[None], l=[None], cmap=False, a=[1], xlim=0, ylim=0, lw=2.0, xlabel='', ylabel='', title='', figname='temp.png', ms=12):
    fig, ax = plt.subplots()
    ax_twin = ax.twinx()
    for i in range(len(x)):
        if mask[i%len(mask)]:
            ax.semilogy(x[i], abs(y[i]), linewidth=lw, marker=m[i%len(m)], color=c[i%len(c)], linestyle=s[i%len(s)], markersize=ms, alpha=a[i%len(a)])
            ax_twin.plot(x[i], y[i], linewidth=lw, marker=m[i%len(m)], color=c[i%len(c)], linestyle=s[i%len(s)], markersize=ms, alpha=a[i%len(a)])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax_twin.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.xlabel(xlabel=xlabel, fontweight='bold')
    plt.ylabel(ylabel=ylabel, fontweight='bold')
    plt.title(label=title, fontweight='bold')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    labels = ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels()
    [label.set_fontweight('bold') for label in labels]
    ax.minorticks_on()
    labels = ax_twin.xaxis.get_ticklabels() + ax_twin.yaxis.get_ticklabels()
    [label.set_fontweight('bold') for label in labels]
    ax_twin.minorticks_on()
    # ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=5))
    plt.tick_params(axis='y', which='minor')
    ax.tick_params(direction='in', length=6, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='major')
    ax.tick_params(direction='in', length=3, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='minor')
    ax_twin.tick_params(direction='in', length=6, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='major')
    ax_twin.tick_params(direction='in', length=3, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='minor')
    if not l == [None]:
        ax.legend(l)
    if not xlim == 0:
        ax.set(xlim=xlim)
    if not ylim == 0:
        ax.set(ylim=ylim)
    # plt.figure(figsize=(1,1))
    plt.grid()
    # plt.show(block=False)
    plt.savefig(fname=figname)

def lin_lin_plot(x1, y1, x2, y2, c1, c2, s1, s2, a1=[1], a2=[1], mask=[True], m1=[None], m2=[None], l=[None], cmap=False, xlim=0, ylim=0, lw=2.0, xlabel='', ylabel='', title='', figname='temp1.png', ms=12):
    fig, ax = plt.subplots()
    ax_twin = ax.twinx()
    for i in range(len(x1)):
        if mask[i%len(mask)]:
            ax.plot(x1[i], y1[i], linewidth=lw, marker=m1[i%len(m1)], color=c1[i%len(c1)], linestyle=s1[i%len(s1)], markersize=ms, alpha=a1[i%len(a1)])
    
    for i in range(len(x2)):
        if mask[i%len(mask)]:
            ax_twin.plot(x2[i], y2[i], linewidth=lw, marker=m2[i%len(m2)], color=c2[i%len(c2)], linestyle=s2[i%len(s2)], markersize=ms, alpha=a2[i%len(a2)])

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax_twin.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.xlabel(xlabel=xlabel, fontweight='bold')
    plt.ylabel(ylabel=ylabel, fontweight='bold')
    plt.title(label=title, fontweight='bold')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    labels = ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels()
    [label.set_fontweight('bold') for label in labels]
    ax.minorticks_on()
    labels = ax_twin.xaxis.get_ticklabels() + ax_twin.yaxis.get_ticklabels()
    [label.set_fontweight('bold') for label in labels]
    ax_twin.minorticks_on()
    # ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=5))
    plt.tick_params(axis='y', which='minor')
    ax.tick_params(direction='in', length=6, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='major')
    ax.tick_params(direction='in', length=3, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='minor')
    ax_twin.tick_params(direction='in', length=6, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='major')
    ax_twin.tick_params(direction='in', length=3, width=2, colors='k',
               grid_color='k', grid_alpha=0.5, which='minor')
    if not l == [None]:
        ax.legend(l)
    if not xlim == 0:
        ax.set(xlim=xlim)
    if not ylim == 0:
        ax.set(ylim=ylim)
    # plt.figure(figsize=(1,1))
    plt.grid()
    # plt.show(block=False)
    plt.savefig(fname=figname)

# helper functions
def sf(y, wl=31, order=3):  # smoothing linear curves
    yhat = []
    for i in range(len(y)):
        yhat.append(savgol_filter(y[i], window_length=wl, polyorder=order))
    return yhat

def sf_log10(y, wl=31, order=3):  # smoothing logy curves
    yhat = []
    for i in range(len(y)):
        yhat.append(savgol_filter(np.log10(np.abs(y[i])), window_length=wl, polyorder=order))
    return [10 ** yy for yy in yhat]

def FD_int_3D(eta): # Fermi-Dirac integral for parabolic 3D DOS
    x = eta
    mu = x**4 + 50 + 33.6*x*( 1 - 0.68 * exp( -0.17 * ( x + 1 )**2 ) )
    xi = 3*pi**0.5 / (4*mu**(3/8))
    y = ( exp( - x ) + xi ) ** -1
    return y

def FD_int_2D(eta): # Fermi-Dirac integral for linear 2D DOS
    x = eta
    y = log(1+exp(x))
    return y

def gaussian_states(Nt, Tt, Ep, Ef, T, nature):
    zeta = (Ef - Ep)/(kB*T)
    s =  Tt/(T*2**0.5)
    H = (2**0.5/s)*erfcinv(exp(-s**2/2))
    K = 2*(1-(H/s)*(2/pi)**0.5*exp(s**2*(1-H**2)/2))
    if zeta < -s**2:
        G = exp(s**2/2 + zeta)/(1 + exp(K*(zeta + s**2)))
    else:
        G = 0.5*erfc(-zeta*H/(s*2**0.5))
    return Nt*G if nature=='Acceptor' else Nt*(G-1)

# def exponential_states(Nt, Tt, Ep, Ef, T, nature):
#     b = T/Tt
#     xp = -exp((Ep-Ef)/(kB*T))
#     return Nt*hyp2f1(1, b, 1+b, xp) if nature=='Acceptor' else Nt*(hyp2f1(1, b, 1+b, xp)-1)
#     # if (abs((Ep-Ef)/(kB*T)) < 100):
#     #     xp = -exp((Ep-Ef)/(kB*T))
#     #     return Nt*hyp2f1(1, b, 1+b, xp) if nature=='Acceptor' else Nt*(hyp2f1(1, b, 1+b, xp)-1)
#     # elif ( (Ep-Ef)/(kB*T) > 100 ):
#     #     return 0 if nature=='Acceptor' else -Nt
#     # else:
#     #     return Nt if nature=='Acceptor' else 0

def exponential_states(Nt, Tt, Ep, Ef, T, nature):
    """
    This function implements the closed-form solution for the exponential FD integral from -inf to +inf.
    The assumed DoS is (Nt/Wt)*exp((E-Em)/Wt)*(E<=Em)
    :param Nt:
    :param Tt:
    :param Ep:
    :param Ef: Fermi energy in eV.
    :param T: Ambient temperature in K.
    :param nature:
    :return: carrier concentration occupying the exponential band tail DoS.
    """
    phit = kB*T
    Wt = kB*Tt
    alpha = Wt / phit
    a, b, c = 1, 1/alpha, 1 + 1/alpha

    def check_exp_overflow(Ep, Ef, phit):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = exp((Ep - Ef) / phit)
            if len(w) > 0 and issubclass(w[-1].category, RuntimeWarning):
                return True, inf
            return False, result

    overflowed, z = check_exp_overflow(Ep, Ef, phit)

    if not overflowed:
        nt = Nt * hyp2f1( a, b, c, -z )
    else:
        """
        This alternate expression is adopted from:
        A. L. M. Beckers, “Cryogenic MOSFET Modeling for Large-Scale Quantum
        Computing,” Ph.D. dissertation, EPFL, Lausanne, 2021. [Online]. Available:
        https://infoscience.epfl.ch/handle/20.500.14299/178114
        """
        # print("OVERFLOW")
        zeta = expit((Ef-Ep)/phit)
        zetab = exp((Ef - Ep)/Wt) #zeta**b
        t1 = (zeta**a)*((gamma(c)*gamma(b-a))/(gamma(b)*gamma(c-a)))*hyp2f1(a, c-b, a-b+1, zeta)
        t2 = (zetab)*((gamma(c)*gamma(a-b))/(gamma(a)*gamma(c-b)))*hyp2f1(b, c-a, b-a+1, zeta)
        nt = Nt * (t1+t2)

    return nt if nature=='Acceptor' else nt-Nt

def lower_energy_levels(H, num_levels):
    ## Uses Lanczos Method
    E, V = las.eigsh(H, k=num_levels, which='SM')
    return E, V

def agilent_csv_cleaner(fname):
    full_data_set = []
    with open(fname, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            # print(line_count, row)
            if row[0] in ['Dimension1', 'Dimension2', 'DataName', 'DataValue']:
                full_data_set.append(row)
            line_count += 1
    print(f'Processed {line_count} lines')

    line_count = 0
    data_set = []
    N1 = 0
    N2 = 0
    while line_count < len(full_data_set):
        row = full_data_set[line_count]
        if row[0] == 'Dimension1':
            N1 = int(row[1])
            print('Dimension1:', N1)
            line_count += 1
            row = full_data_set[line_count]
            N2 = int(row[1])
            print('Dimension2:', N2)
            line_count += 1
            temp_data = [[[str(X) for X in full_data_set[line_count][1:9]]] for ii in range(N2)]
            line_count += 1
            for ii in range(N2):
                for elem in range(N1):
                    # print(full_data_set[line_count])
                    temp_data[ii].append([float(X) for X in full_data_set[line_count][1:9]])
                    line_count += 1
            data_set += temp_data
    print(f'Number of datasets: {len(data_set)}')

    pd_data_set = [pd.DataFrame(X[1:], columns=X[0]) for X in data_set]

    return pd_data_set, N1


def vtcc_calc(Vg, Id, vt_i):
    for i in range(len(Vg)):
        if Id[i] >= vt_i:
            break
    idxl = i

    return Vg[idxl - 1] + (Vg[idxl] - Vg[idxl - 1]) * (log10(vt_i) - log10(Id[idxl - 1])) / (
                log10(Id[idxl]) - log10(Id[idxl - 1]))

def vtle_calc(Vg, Id, Vd=100e-3, pmode=False):
    # Find the point where the derivative is maximum
    x = Vg
    y = Id
    f_prime = gradient(Id) / gradient(Vg) 
    
    # Find the point where the derivative is maximum
    x_max_index = np.argmax(f_prime)
    x_max = x[x_max_index]
    y_max = y[x_max_index]
    
    # Select 5 points around x_max
    indices = np.arange(max(0, x_max_index - 2), min(len(x), x_max_index + 3))
    print(indices)
    x_points = x[indices]
    y_points = y[indices]
    
    # Perform linear regression
    X = x_points.reshape(-1, 1)
    reg = LinearRegression().fit(X, y_points)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    # Equation of the best-fit line
    def best_fit_line(x):
        return slope * x + intercept
    
    # Find the x-intercept of the best-fit line
    x_intercept = -intercept / slope
    
    if not pmode:
        return x_intercept - Vd / 2
    else:
        xfit = [x_intercept, max(x)]
        yfit = [0, max(y)]
        return x_intercept - Vd / 2, xfit, yfit
    
def logle_calc(Vg, Id, lower_i, higher_i):
    for i in range(len(Vg)):
        if Id[::-1][i] <= lower_i: #lower limit
            break
    idxl = len(Vg) - i - 1
    print(f'lower_i = {Id[idxl]}')
    for i in range(len(Vg)):
        if Id[i] >= higher_i: # higher limit for SS extraction
            break
    idxu = i
    print(f'higher_i = {Id[idxu]}')
    lxx = Vg[idxl:idxu+1]
    lyy = log10(Id[idxl:idxu+1])
    
    # Perform linear regression
    lX = lxx.reshape(-1, 1)
    lreg = LinearRegression().fit(lX, lyy)
    lslope = lreg.coef_[0]
    lintercept = lreg.intercept_
    return lslope, lintercept
    

def keithley_reader_xls(fname):
    # sname = lambda i: 'Data' if i == 0 else f'Append{i}'
    sname = lambda i: 'Data' if i == 0 else f'Cycle{i+1}'
    df_sheet_all = pd.read_excel(fname, sheet_name=None)
    i = 0
    pd_data_set = []
    while sname(i) in df_sheet_all:
        pd_data_set.append(df_sheet_all[sname(i)])
        i += 1
    return pd_data_set

def keithley_reader_xls_bs(fname):
    df_sheet_all = pd.read_excel(fname, sheet_name='Data')
    np_df = df_sheet_all.values
    n1, n2 = np.shape(np_df)
    np_data_set = []
    tnow = 0
    tprev = 0
    start = 0
    for ii in range(n1):
        tnow = int(np_df[ii,5])
        if tnow != tprev:
            tprev = tnow
            np_data_set.append(np_df[start:ii,:])
            start = ii
    np_data_set.append(np_df[start:,:])
    return np_data_set

def SS_calc(Vg, Id, lower_i, higher_i):
    for i in range(int(len(Vg)/2)):
        if Id[i] >= lower_i: #lower limit
            break
    idxl = i
    for i in range(int(len(Vg)/2)):
        if Id[i] >= higher_i: # higher limit for SS extraction
            break
    idxu = i
    print(Id[idxl], Id[idxu])
    return (Vg[idxu] - Vg[idxl])*1e3/(log10(Id[idxu]) - log10(Id[idxl]))

def read_lis(fname):
    lines = list(fi.input(files = fname))

    line_num = 0
    first_word = lines[line_num].split()
    first_word = None if len(first_word) == 0 else first_word[0]
    acq_start = 0
    acq_list = []
    while first_word != 'y':
        if acq_start == 1:
            acq_list.append(lines[line_num].split())
        if first_word == 'time':
            line_num += 2
            acq_start = 1
        else:
            line_num += 1
        first_word = lines[line_num].split()
        first_word = None if len(first_word) == 0 else first_word[0]

    time_acq_list = np.array([ float(x[0]) for x in acq_list ])
    vsn_acq_list = np.array([ float(x[1]) for x in acq_list ])
    vwbl_acq_list = np.array([ float(x[2]) for x in acq_list ])
    vwwl_acq_list = np.array([ float(x[3]) for x in acq_list ])
    vrwl_acq_list = np.array([ float(x[4]) for x in acq_list ])
    
    return time_acq_list, vsn_acq_list, vwwl_acq_list, vwbl_acq_list, vrwl_acq_list

def read_lis_iv(fname):
    lines = list(fi.input(files = fname))

    line_num = 0
    first_word = lines[line_num].split()
    first_word = None if len(first_word) == 0 else first_word[0]
    acq_start = 0
    acq_list = []
    while first_word != 'y':
        if acq_start == 1:
            acq_list.append(lines[line_num].split())
        if first_word == 'vgate':
            line_num += 2
            acq_start = 1
        else:
            line_num += 1
        first_word = lines[line_num].split()
        first_word = None if len(first_word) == 0 else first_word[0]

    vg_acq_list = np.array([ float(x[0]) for x in acq_list ])
    id_acq_list = np.array([ float(x[1]) for x in acq_list ])

    return vg_acq_list, id_acq_list

def read_lis_cv(fname):
    lines = list(fi.input(files = fname))

    line_num = 0
    first_word = lines[line_num].split()
    first_word = None if len(first_word) == 0 else first_word[0]
    acq_start = 0
    acq_list = []
    while first_word != 'y':
        if acq_start == 1:
            acq_list.append(lines[line_num].split())
        if first_word == 'time':
            line_num += 2
            acq_start = 1
        else:
            line_num += 1
        first_word = lines[line_num].split()
        first_word = None if len(first_word) == 0 else first_word[0]

    vg_acq_list = np.array([ float(x[2]) for x in acq_list ])
    ig_acq_list = np.array([ float(x[1]) for x in acq_list ])

    return vg_acq_list, ig_acq_list

def generate_spice_netlist(fname, params):
    i_lines = list(fi.input(files = fname+'_template.sp'))
    o_lines = []

    for line in i_lines:
        line_list = line.split()
        if len(line_list) == 4:
            if line_list[0] == ".PARAM" and line_list[3][0]=='@':
                line_list[3] = str(params[line_list[1]])
        if len(line_list) > 1:
            if line_list[0] == '*out*':
                line_list[0] = ''
                if params['out_mode'] == 1:
                    line_list[2] = 'I(vrb)'
            if params['gc_mode'] == 0:
                if line_list[0] == '*ww*' or line_list[0] == '*wb*' or line_list[0] == '*rwd*':
                    line_list[0] = ''
            if params['gc_mode'] == 1:
                if line_list[0] == '*wwd*' or line_list[0] == '*wbd*' or line_list[0] == '*rwd*':
                    line_list[0] = ''
            if params['gc_mode'] == 2:
                if line_list[0] == '*wwd*' or line_list[0] == '*wbd*' or line_list[0] == '*rw*':
                    line_list[0] = ''  
        o_lines.append(' '.join(line_list))

    # print(o_lines)
    with open(fname+'.sp', 'w') as fp:
        fp.write('\n'.join(o_lines))
    fp.close()

def read_lis_general(fname):
    lines = list(fi.input(files = fname))

    line_num = 0
    first_word = lines[line_num].split()
    first_word = None if len(first_word) == 0 else first_word[0]
    acq_start = 0
    acq_list = []
    occ = 0
    while True:
        if first_word == 'y':
            next_word = lines[line_num+1].split()
            next_word = None if len(next_word) == 0 else next_word[0]
            if next_word == 'x':
                line_num += 5
                occ = line_num
            else:
                break
        if acq_start == 1:
            if occ == 0:
                acq_list.append(lines[line_num].split())
            else:
                acq_list[line_num-occ] += lines[line_num].split()
        if first_word == 'x':
            line_num += 4
            acq_start = 1
        else:
            line_num += 1
        first_word = lines[line_num].split()
        first_word = None if len(first_word) == 0 else first_word[0]

    N1 = len(acq_list)
    N2 = len(acq_list[0])
    lis_M = np.zeros((N1,N2))
    for ii in range(N2):
        lis_M[:,ii] = np.array([ float(x[ii]) for x in acq_list ])
    return lis_M

def engnot(number):
    not_dict = {-6:'a', -5:'f', -4:'p', -3:'n', -2:'u', -1:'m', 0:'', 1:'k', 2:'M', 3:'G', 4:'T'}
    reqd_not = None
    for ii in not_dict:
        if number/10**(3*ii) >= 1 and number/10**(3*ii) < 1000:
            reqd_exp = not_dict[ii]
            reqd_mantissa = number/10**(3*ii)
    return f'{int(reqd_mantissa)}{reqd_exp}'

def read_lis_modified(fname, sweep_variable='vgate'):
    lines = list(fi.input(files = fname))

    filtered_lines = []
    capture = False

    header = None
    data_list = []
    data_list_index = 0
    for line in lines:
        if line.startswith('x'):
            capture = True
            data_list.append([])
            continue
        if capture:
            if line.startswith('y'):
                capture = False
                data_list_index += 1
            else:
                cleaned_line = [item for item in line.strip('\n').split(' ') if item]
                if len(cleaned_line) > 0:
                    if cleaned_line[0] == sweep_variable:
                        header = cleaned_line
                    elif is_number(cleaned_line[0]):
                        data_list[data_list_index].append([float(s) for s in cleaned_line])
                    else:
                        header[1:] = [f"{head}_{elem}" for head, elem in zip(header[1:], cleaned_line)]
                    # print(cleaned_line)

    # Create the DataFrame
    num_datasets = data_list_index
    df_list = [pd.DataFrame(data=data_list[ii], columns=header, dtype=float) for ii in range(num_datasets)]

    # # Display the DataFrame
    # print(df_list)

    return num_datasets, df_list
            
            
            
    