import numpy as np
from dataclasses import dataclass
import typing
from scipy.stats.distributions import chi2


def mahalanobis_dist(dx, sigma):
    '''Assume no skew, or correlation
    '''
    dx = dx.reshape(-1,1)
    # print('dx: {0}'.format(dx))
    res = np.sqrt(dx.T @ dx / sigma)
    # print(res.shape)
    return res[0,0]



def multivariate_normal_percentile_dist(xx, sigma):
    n_dims = xx.reshape(-1).size
    m_dist = mahalanobis_dist( xx, sigma )
    chi2inv = chi2.cdf(m_dist, df=n_dims)
    return chi2inv



def smart_adjust_sigma(last_sigma, xx, pp=0.5, low_cap=0.2, high_cap=1.6):
    n_dims = xx.reshape(-1).size

    unit_dist = mahalanobis_dist( xx, 1 )

    new_mdist = np.sqrt(chi2.ppf(pp, df=n_dims))
    rec_new_sigma = (new_mdist / unit_dist)**-2
    safe_new_sigma = np.maximum(
        low_cap*last_sigma,
        np.minimum(
            high_cap * last_sigma,
            rec_new_sigma,
        )
    )
    return safe_new_sigma





@dataclass
class ShotgunOptimizeConfig:

    # First sigma should be same scale as size between minima, in x space (not f(x) space)
    sigma_start : float
    
    # Size of vector used to search
    search_size : int = 25_000
    
    # Maximum number of iterations to run before terminating
    max_iter    : int = 100
    
    # If algo finds an improvement of less than tol percent over n_lookback steps,
    #  as compared against first n_lookback steps,
    #  => then terminates, signalling success
    #  this is like cauchy convergence
    convergence_steps_lookback  : int = 4
    convergence_steps_tol       : float = 1/50

    # Should be between 0.0 and 1.0, but probably around 0.5
    # This is a percentile against which sigma is calibrated
    # - smaller values mean larger sigma, and visa versa
    pp_param: float = 0.6

    # If we attempt to use sigma smaller than this,
    #  return, and signal success
    sigma_tol   : typing.Optional[float] = None





def adjust_sigma(last_sigma, x0, x1):
    j = 2 * np.linalg.norm(x1-x0) / np.sqrt(len(x0))
    new_sigma = (2*last_sigma + j)/3

    print('old sigma:', last_sigma)
    print('new sigma:', new_sigma)
    return new_sigma

    # j = 2 * np.linalg.norm(x1-x0) / np.sqrt(len(x0))
    # return (3*last_sigma + j)/4

        



def single_shotgun_iteration(config, x0, f, sigma, i, constraint_f, search_size_override=None):
    search_size = config.search_size if search_size_override is None else search_size_override
    rand_shape = tuple([search_size] + list(x0.shape))
    # print(rand_shape)

    # sigma = adjust_sigma(config.sigma_start, config.max_iter, i)
    rand_weights = np.random.normal(x0, sigma, rand_shape)

    weights = constraint_f(rand_weights)

    evals = f(weights)

    argmin      = np.argmin(evals, axis=0)
    eval_min    = evals[argmin].copy()
    weights_min = weights[argmin].copy()

    dx = (x0 - weights_min).reshape(-1,1)
    dist = np.linalg.norm( dx,)
    dist_percentile = multivariate_normal_percentile_dist( dx, sigma )

    return {
        'xx': weights_min,
        'fx': eval_min,
        'sigma': sigma,
        'dist': dist,
        'dist_percentile': dist_percentile,
        # 'recommended_next_sigma': next_sigma,
    }






def shotgun_optimize_simple(
        config: ShotgunOptimizeConfig,
        x0: np.ndarray,
        f,
        constraint_f = lambda x: x,
        verbose:bool=False,
    ):
    '''This is one of the only functions here that should be used by end user.
    Optimizer can handle very large search spaces efficiently,
    just buy randomly splattering guesses, finding best one,
    and continuing to look nearby the best guess so far


    f should be callable, taking 2-dimensional array of weights, with (_,w) shape of input
    and should return (_,) shape array- the objective having been performed on the 2nd axis
    '''
    # print('using sigma tol:', config.sigma_tol)

    history = {
        'xx': [],
        'fx': [],
        'sigma': [],
        'dist': [],
        'dist_percentile': [],
        'step_success': [],
    }
    
    best_eval = f(np.expand_dims(x0,0))[0]
    print('starting eval:', best_eval)
    xx = x0
    
    sigma = config.sigma_start
    
    i = None
    success = False
    reason = 'max iterations'
    for i in range(config.max_iter):
        print()
        if success:
            break

        if verbose:
            print('iter {0}, sigma {2:,.2f} (best fx {1:,.4f})'.format(i, best_eval, sigma))
        res = single_shotgun_iteration(
            config,
            sigma=sigma, x0=xx, f=f, i=i,
            constraint_f = constraint_f,
        )
        new_eval = res['fx']

        history['xx'].append(res['xx'])
        history['fx'].append(res['fx'])
        history['dist'].append(res['dist'])
        history['dist_percentile'].append(res['dist_percentile'])
        history['sigma'].append(sigma)

        # sigma = adjust_sigma(sigma, xx, res['xx'])
        
        if new_eval < best_eval:
            # print('found new best eval')
            history['step_success'].append(True)
            # print()

            if best_eval - new_eval < config.eval_tol:
                success = True
                reason = 'eval tol'
                break

            best_eval = new_eval
            xx = res['xx']
            sigma = res['recommended_next_sigma']

        else:
            # print('failed to find best new eval')
            history['step_success'].append(False)
            sigma = adjust_sigma(
                sigma,
                x0 = xx,
                x1 = xx,
            )

        # sigma = res['recommended_next_sigma']


        if config.sigma_tol is not None and sigma < config.sigma_tol:
            success = True
            reason = 'sigma tol'
            break
    
            
            
    return {
        **res,
        'n_iter': i+1,
        'stop_reason': reason,
        'success': success,
        'history': history,
    }




class ObjectiveDistTracker:
    def __init__(self, num_lookback=3, convergence_test=1/20 ):

        self.f_evals = []
        self._lookback = num_lookback
        self._tol = convergence_test

        self._thresh = None
    
    def _cur_val(self):
        return self.f_evals[-self._lookback] - self.f_evals[-1]

    def log_eval(self, fx):
        self.f_evals.append(fx)
        if len(self.f_evals) == self._lookback:
            self._thresh = self._cur_val()
    

    def progress_fraction(self,):
        if len(self.f_evals) <= self._lookback:
            return 1.0
        else:
            # print(self._tol, self._thresh, self._cur_val())
            return np.maximum(0,np.minimum(1,
                (self._cur_val()/self._thresh) - self._tol
            ))
