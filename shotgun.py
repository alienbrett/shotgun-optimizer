import numpy as np
from dataclasses import dataclass
import typing

from shells import *




class ShotgunOptimizer:
    def __init__(self,
            objective_func,
            config: ShotgunOptimizeConfig,
            constraint_func=lambda x: x,
        ):
        self._f = objective_func
        self._config = config
        self._constraint_func = constraint_func


    def adjust_sigma(self, x0, x1, fx0, fx1):
        last_sigma = self.sigma
        # If no success,
        if fx0 < fx1:
            new_sigma = last_sigma * 1/2

        # If we DID improved best_eval:
        else:
            new_sigma = smart_adjust_sigma(
                last_sigma=last_sigma,
                xx = x1,
                pp = self._config.pp_param,
                low_cap = 0.25,
                high_cap = 1.75,
            )
        self.sigma = new_sigma
        return new_sigma


    
    def step(self, i:int):
        converged_power = 1.0
        converged_ratio = 0.4
        converged_frac = self._convergence_tracker.progress_fraction()

        search_size = int(
            (converged_ratio + (1-converged_ratio)*converged_frac)**converged_power * self._config.search_size
        )

        print('using search size {0:,.0f}'.format(search_size))

        res = single_shotgun_iteration(
            self._config,
            sigma=self.sigma, x0=self.xx, f=self._f, i=i,
            constraint_f = self._constraint_func,
            search_size_override = search_size,
        )
        return res

    def _history_init(self,):
        self.history = {
            'xx': [],
            'fx': [],
            'sigma': [],
            'dist': [],
            'dist_percentile': [],
            'step_success': [],
        }

    def _register_history(self, res, step_success):
        self.history['xx'].append(res['xx'])
        self.history['fx'].append(res['fx'])
        self.history['dist'].append(res['dist'])
        self.history['dist_percentile'].append(res['dist_percentile'])
        self.history['sigma'].append(self.sigma)
        self.history['step_success'].append(step_success)



    def run(self, x0, verbose=False):
        
        self._history_init()
        
        best_eval = self._f(np.expand_dims(x0,0))[0]
        # print('starting eval:', best_eval)

        self._convergence_tracker = ObjectiveDistTracker(
            num_lookback = self._config.convergence_steps_lookback,
            convergence_test = self._config.convergence_steps_tol,
        )

        self.xx = x0
        self.sigma = self._config.sigma_start
        i = None
        success = False
        reason = 'max iterations'
        for i in range(self._config.max_iter):
            # print()
            if success:
                break

            if verbose:
                print('iter {0}, sigma {2:,.2f} (best fx {1:,.4f})'.format(i, best_eval, self.sigma))

            # Perform the actual step
            res = self.step(i)
            new_eval = res['fx']

            step_success = new_eval < best_eval
            self._register_history(res, step_success)

            self.adjust_sigma(
                # last_sigma = self.sigma,
                x0 = self.xx,
                x1 = res['xx'],
                fx0 = best_eval,
                fx1 = new_eval,
            )

            if step_success:
                best_eval = new_eval
                self.xx = res['xx']
                self._convergence_tracker.log_eval(best_eval)


            if self._config.sigma_tol is not None and self.sigma < self._config.sigma_tol:
                success = True
                reason = 'sigma tol'
                break
                
            if verbose:
                print(
                    'convergence progress: {0:,.0f}%'.format(
                        100*(1-self._convergence_tracker.progress_fraction())
                    )
                )
            
            if self._convergence_tracker.progress_fraction() <= 0:
                success = True
                reason = 'convergence'
                break
                
                
        self.results = {
            **res,
            'n_iter': i+1,
            'stop_reason': reason,
            'success': success,
            'history': self.history,
        }

        return self.results