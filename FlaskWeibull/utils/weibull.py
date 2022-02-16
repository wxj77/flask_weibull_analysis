import pdb

import os
import numpy as np
import time
import pandas as pd
import scipy
from pkg_resources import resource_filename

from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P
from reliability.Other_functions import crosshairs
from reliability.Probability_plotting import Exponential_probability_plot, Weibull_probability_plot, Exponential_probability_plot_Weibull_Scale


image_dir_path = resource_filename(__name__, '../static/image/')
print('image dir path is:', image_dir_path)

import matplotlib.pyplot as plt

class Weibull2PLikelihood:
    # defination of Weibull: https://reliability.readthedocs.io/en/latest/Equations%20of%20supported%20distributions.html#relationships-between-the-five-functions
    # https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html
    def __init__(self, failures, right_censored = []):
        self.failures = np.array(failures)
        self.right_censored = np.array(right_censored)
        
    def _fit_weibull_2P(self, show_probability_plot=True, print_results=True, \
                        CI=0.99, quantiles=None, CI_type='time', method='MLE', \
                        optimizer=None, force_beta=None, downsample_scatterplot=True, **kwargs):
        self.result = Fit_Weibull_2P(self.failures, self.right_censored, show_probability_plot, print_results, \
                                     CI, quantiles, CI_type, method, \
                                     optimizer, force_beta, downsample_scatterplot, **kwargs)
        
    def _log_linspace(self, start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
        start_new = np.log(start)
        stop_new = np.log(stop)
        return np.exp(np.linspace(start = start_new, stop = stop_new, num=num, \
                                  endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis))

    def _exp_linspace(self, start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
        start_new = np.exp(start)
        stop_new = np.exp(stop)
        return np.log(np.linspace(start = start_new, stop = stop_new, num=num, \
                                  endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis))
    
    def _loglik_weibull(self, dist):
        '''
        Args:
            failures: arr
            right_censored: arr
            dist: distribution
        Return:
            loglik: log likelihood.
        '''
        loglik = 0
        for ii in self.failures:
            loglik+=np.log(dist.PDF(ii))
        for ii in self.right_censored:
            loglik+=np.log(dist.SF(ii))
        return loglik
    
    def _get_loglik_profile(self, num = 50):
        if not hasattr(self, 'result'):
            self._fit_weibull_2P(show_probability_plot=False, print_results=False)
        #starttime = time.time()
        lin_a = self._log_linspace(self.result.alpha_lower, self.result.alpha_upper, num = num)
        lin_b = self._log_linspace(self.result.beta_lower, self.result.beta_upper, num = num)
        X, Y =  np.meshgrid(lin_a, lin_b)
                
        def jlw(x, y):
            return self._joint_loglik_weibull(x, y, self.failures, self.right_censored)
        Z = np.vectorize(jlw)(X, Y)
        return X, Y, Z
    
    def _trapzoid_normalize(self, X, Y, Z):
        '''
        Args:
            X: mesh grid
            Y: mesh grid
            Z: value
        Return:
            x: mesh grid
            y: mesh grid
            z: normalized value
            dx: x integral bin width
            dy: y integral bin width
        '''
        assert X.shape == Y.shape == Z.shape
        m, n = X.shape
        dx = X[1:,1:]-X[:-1,:-1]
        dy = Y[1:,1:]-Y[:-1,:-1]    
        x = 0.5* (X[1:,1:]+X[:-1,:-1])
        y = 0.5* (Y[1:,1:]+Y[:-1,:-1])
        area = dx * dy
        value = 0.25 * (Z[1:,1:] + Z[1:,:-1] + Z[:-1,1:] + Z[:-1,:-1])
        return x, y, value/np.sum(area * value), dx, dy
            
    def _get_profile(self, num = 50):
        X, Y, Z = self._get_loglik_profile(num = num)
        v_alpha, v_beta, v_prob, d_alpha, d_beta = self._trapzoid_normalize(X, Y, np.exp(Z-np.max(Z))) #-np.max(Z) is to avoid Overflow
        self._profile = (v_alpha, v_beta, v_prob, d_alpha, d_beta)
        return self._profile
    
    def _get_integ(self, f):
        '''
        Args:
            f: a function of alpha and beta
        Return:
            mean: expected value of f
            var: expected value of (f- f_mean)**2 
        '''
        if not hasattr(self, '_profile'):
            self._get_profile()        
        v_alpha, v_beta, v_prob, d_alpha, d_beta = self._profile
        
        mean = np.sum(np.vectorize(f)(v_alpha, v_beta) * v_prob * d_alpha * d_beta)
        mean2 = np.sum(np.vectorize(f)(v_alpha, v_beta)**2 * v_prob * d_alpha * d_beta)
        var = mean2 - mean**2
        
        return mean, var
    
    def _plot_profile(self):
        if not hasattr(self, '_profile'):
            self._get_profile() 
        v_alpha, v_beta, v_prob, d_alpha, d_beta = self._profile   
            
        fig, ax = plt.subplots(constrained_layout=True)
        origin = 'lower'
        delta = 0.025

        nr, nc = v_prob.shape

        CS = ax.contourf(v_alpha, v_beta, v_prob, 10, cmap=plt.cm.bone, origin=origin)

        ax.set_title('Normalized Weibull Parameter Likelihood Estimation')
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')

        cbar = fig.colorbar(CS)
        cbar.ax.set_ylabel('likelihood')
        return fig, ax
    
    def _weibull_pdf(self, alpha, beta, x):
        return beta/x * (x/alpha)**beta * np.exp(-(x/alpha)**beta)

    def _weibull_sf(self, alpha, beta, x):
        return np.exp(-(x/alpha)**beta)

    def _weibull_cdf(self, alpha, beta, x):
        return 1. - self.weibull_sf(alpha, beta, x)

    def _weibull_hf(self, alpha, beta, x):
        return beta/x * (x/alpha)**beta     

    def _weibull_chf(self, alpha, beta, x):
        return (x/alpha)**(beta) 
    
    def _joint_weibull(self, alpha, beta, *data):
        res = 1
        #print(data)
        for ii in data[0]:
            res *= self._weibull_pdf(alpha, beta, ii)
        for ii in data[1]:
            res *= self._weibull_sf(alpha, beta, ii)
        return res
        # usage: 
        # np.log(joint_weibull(alpha, beta, failures, right_censored))

    def _joint_loglik_weibull(self, alpha, beta, *data):
        res = np.sum(np.log(self._weibull_pdf(alpha, beta, data[0]))) \
            + np.sum(np.log(self._weibull_sf(alpha, beta, data[1])))
        
        return res
        # usage:
        #joint_loglik_weibull(alpha, beta, failures, right_censored)
        
    def _weighted_percentile(self, data, CI, weights = None):
        if weights is None:
            return np.percentile(data, CI * 100)
        ind = np.argsort(data)
        d = data[ind]
        w = weights[ind]
        p = 1. * w.cumsum() / w.sum()
        y = np.interp(CI, p, d)
        return y
    
    def _get_ft_from_profile(self, f, t):
        if not hasattr(self, '_profile'):
            self._get_profile() 
        v_alpha, v_beta, v_prob, d_alpha, d_beta = self._profile 
        v_rep = np.zeros(list(v_alpha.shape) + list(t.shape))
        for ii, tt in enumerate(t):
            v = np.vectorize(f)(v_alpha, v_beta, tt)
            v_rep[:,:,ii] = v
        return v_rep
    
    def _get_weighted_percentile_ft_from_profile(self, f, t, percents=None):
        '''
        Args: 
            f: function (alpha, beta, t)
            t: arr 1d
            percent: list of percents
        Returns:
            res: list(arr1d) arr1d shape is the same as t, len is len(percents)
        '''
        if not hasattr(self, '_profile'):
            self._get_profile() 
        v_alpha, v_beta, v_prob, d_alpha, d_beta = self._profile 
        if percents is None: 
            return []
        v_rep = self._get_ft_from_profile(f,t)
        res = [np.zeros_like(t) for _ in range(len(percents))]
        for ii, tt in enumerate(t):
            v = v_rep[:,:,ii]
            for jj,pp in enumerate(percents):
                res[jj][ii] = self._weighted_percentile(v.flatten(), pp, weights=(v_prob * d_alpha * d_beta).flatten(), )    
        return res
    
    def _get_median_upper_lower_bound_from_profile(self, f, t = None, CI = .90, plot_profile = True, *args, **kwargs):
        '''
        Args:
            f: function (alpha, beta, t)
            t: arr 1d
            CI: confidence level (default .90)
            plot_profile: plot profile (default True)
        Return:
            res: [median, lower, upper] each if arr1d shape is the same as t.
            (fig, ax): plot, if plot_profile is True
        '''
        
        if t is None:
            t = self._log_linspace(np.min(self.failures), np.max(self.failures) * 2)
        percents = [.5, .5+CI/2., .5-CI/2.]
        res = self._get_weighted_percentile_ft_from_profile(f, t, percents=percents)
        
        if not plot_profile:
            return (t, res)
        else:
            print(kwargs)
            fig, ax = self._plot_ft_CI_from_profile(res, t, *args, **kwargs)
            return (t, res), (fig, ax)

    def _get_median_upper_lower_bound_from_variance(self, f, dfda, dfdb, t = None, CI = .90, plot_profile = True, method='Gaussian',*args, **kwargs):
        '''
        Central limit theorem: 
            f(t) ~ Gaussian
            f(t) ~ ExpGaussian
            
        E(f) ~= f(E(a),E(b))
        var(f) ~= [dfda, dfdb][var(a) cov(a,b)  [dfda, dfdb]^T
                                cov(a,b) var(b)]
        upper/lower bound is given by E(f) +- Z((1-CI)/2) sqrt(Var(f)) 
        Args:
            f, dfda, dfdb : function and partial derivative of function (alpha, beta, t)
            t: arr 1d
            CI: confidence level (default .90)
            plot_profile: plot profile (default True)
            method: ['Gaussian', ExpGaussian']
        Return:
            res: [median, lower, upper] each if arr1d shape is the same as t.
            (fig, ax): plot, if plot_profile is True
        '''
        if not hasattr(self, 'result'):
            self._fit_weibull_2P(show_probability_plot=False, print_results=False)
        
        if t is None:
            t = self._log_linspace(np.min(self.failures), np.max(self.failures) * 2)
        percents = [.5, .5+CI/2., .5-CI/2.]
        
        Z = -scipy.stats.norm.ppf((1. - CI) / 2.)
        
        ea = self.result.alpha
        eb = self.result.beta
        ef = f(ea,eb,t)
        edfda = dfda(ea,eb,t)
        edfdb = dfdb(ea,eb,t)
        vara = self.result.alpha_SE**2
        varb = self.result.beta_SE**2
        covab  = self.result.Cov_alpha_beta
        sigma = vara * (edfda)**2 + 2 * covab * edfda * edfdb + varb * (edfdb)**2
        if method == 'Gaussian':
            res = [ef, ef - Z * np.sqrt(sigma) , ef + Z * np.sqrt(sigma)]
        elif method == 'ExpGaussian':
            res = [ef, ef / np.exp(Z * np.sqrt(sigma) / ef) , ef * np.exp(Z * np.sqrt(sigma) / ef)]
        else:
            raise ValueError('ValueError: method %s does not exist'%(method))
        
        if not plot_profile:
            return (t, res)
        else:
            #print(kwargs)
            fig, ax = self._plot_ft_CI_from_profile(res, t, *args, **kwargs) 
            return (t, res), (fig, ax)
    
    def _plot_ft_CI_from_profile(self, res, t, *args, **kwargs):
        fig, ax = plt.subplots(constrained_layout=True)
        origin = 'lower'
        
        ax.plot(t, res[0], color = 'C0', linestyle='-', linewidth=1)
        ax.fill_between(t, res[1], res[2], color = 'C0', alpha=0.2)
        
        ax.set_title(kwargs.get('title', ''))
        ax.set_xlabel(kwargs.get('xlabel', 'Time'))
        ax.set_ylabel(kwargs.get('ylabel', ''))

        ax.set_xscale(kwargs.get('set_xscale', 'linear'))
        ax.set_yscale(kwargs.get('set_yscale', 'linear'))
        
        ax.grid(visible=True, which='major', color='gray', linestyle='-', linewidth=1)
        ax.grid(visible=True, which='minor', color='gray', linestyle=':', linewidth=1)
        return fig, ax        
    
# define a function of alpha and beta, return the expected mean and var. 
def f(alpha, beta):
    return 1

def f_alpha(alpha, beta):
    return alpha

def f_beta(alpha, beta):
    return beta

def weibull_hf(alpha, beta, x):
    return beta/x * (x/alpha)**beta 

def weibull_hf_dfda(alpha, beta, x):
    return -beta**2/alpha**2 * (x/alpha)**(beta-1) 

def weibull_hf_dfdb(alpha, beta, x):
    return 1/x * (x/alpha)**beta  + beta/x * (x/alpha)**beta * np.log(x/alpha) 

def weibull_chf(alpha, beta, x):
    return (x/alpha)**(beta) 

def weibull_chf_dfda(alpha, beta, x):
    return - beta/alpha * (x/alpha)**(beta)

def weibull_chf_dfdb(alpha, beta, x):
    return (x/alpha)**beta * np.log(x/alpha) 



def process_weibull(failures, right_censored, CI: float = 0.99, t = np.linspace(100,10000,101), prefix: str = ''):
    # example create a Weibull2PLikelihood with data
    w = Weibull2PLikelihood(failures = failures, right_censored = right_censored)

    # fit a Weibull with CI=.99
    w._fit_weibull_2P(show_probability_plot=False, print_results=False, CI=CI,)

    # get alpha beta profile from likelihood functions, with a 100x100 grid with CI
    v_alpha, v_beta, v_prob, d_alpha, d_beta = w._get_profile(num = 100)

    # plot probability
    fig, ax = plt.subplots(constrained_layout=True)
    origin = 'lower'
    Weibull_probability_plot(failures=failures, right_censored=right_censored) #generates the probability plot
    plt.legend()
    filename = os.path.join(image_dir_path, prefix + 'probability_plot.png')
    print('filename', filename)
    fig.savefig(filename)

    # plot profile of alpha and beta
    fig, ax = w._plot_profile()
    filename = os.path.join(image_dir_path, prefix + 'profile.png')
    print('filename', filename)
    fig.savefig(filename)

    # plot HF from profile
    (t0, res0), (fig, ax) = w._get_median_upper_lower_bound_from_profile(
                    w._weibull_hf, t = t, CI = CI, \
                    title = 'Hazard Function\nCI = %.2f'%(CI), ylabel = 'Hazard Function',
                    set_xscale = 'log', set_yscale = 'log', 
                    )
    filename = os.path.join(image_dir_path, prefix + 'HF_get_median_upper_lower_bound_from_profile.png')
    print('filename', filename)
    fig.savefig(filename)

    # plot HF from variance
    (t1, res1), (fig, ax) = w._get_median_upper_lower_bound_from_variance(
                    weibull_hf, weibull_hf_dfda, weibull_hf_dfdb, t = t, CI = CI, \
                    title = 'Hazard Function\nCI = %.2f'%(CI), ylabel = 'Hazard Function',
                    set_xscale = 'log', set_yscale = 'log', 
                    )
    filename = os.path.join(image_dir_path, prefix + 'HF_get_median_upper_lower_bound_from_variance.png')
    print('filename', filename)
    fig.savefig(filename)

    # plot CHF from profile
    (t2, res2), (fig, ax) = w._get_median_upper_lower_bound_from_profile(
                    w._weibull_chf, t = t, CI = CI, \
                    title = 'Cumulative Hazard Function\nCI = %.2f'%(CI), ylabel = 'Cumulative Hazard Function',
                    set_xscale = 'log', set_yscale = 'log', 
                    )
    filename = os.path.join(image_dir_path, prefix + 'CHF_get_median_upper_lower_bound_from_profile.png')
    print('filename', filename)
    fig.savefig(filename)

    # plot CHF from variance
    (t3, res3), (fig, ax) = w._get_median_upper_lower_bound_from_variance(
                    weibull_chf, weibull_chf_dfda, weibull_chf_dfdb, t = t, CI = CI, \
                    title = 'Cumulative Hazard Function\nCI = %.2f'%(CI), ylabel = 'Cumulative Hazard Function',
                    set_xscale = 'log', set_yscale = 'log', 
                    )
    filename = os.path.join(image_dir_path, prefix + 'CHF_get_median_upper_lower_bound_from_variance.png')
    print('filename', filename)
    fig.savefig(filename)

    plt.close('all')

    print('Succeed! End!')

    content = {
        "w": w,
    }
    return content   