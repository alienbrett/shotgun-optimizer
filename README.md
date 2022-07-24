![Shotgun-optimize](https://github.com/alienbrett/shotgun-optimizer/blob/main/shotgun-icon.png?raw=true)

# Shotgun Optimizer
Stochastic optimization for high dimensions in Python


## How does it work?
The routine first generates an array of candidates in input-space using a normal distribution.

After evaluating the objective function on each candidate in this array (in a single step, assuming the objective function is vectorized), the candidate with lowest objective is selected.

The next iteration of candidates is centered around this "best guess". We hope the algorithm has moved closer to the region of input-space that will produce a local minima.

I called the routine "shotgun optimizer" because the routine naively guesses around the current best estimate, like pellets of a shotgun. We hope to find some points with better objective function by guessing many points in high-dimensional space around some reference point.

This implimentation is also able to handle constrained optimization, by allowing the user to project candidate points onto whatever subspace is desired in each loop iteration.

## How is the algorithm adaptive?
The routine adjusts the sigma parameter used when generating candidates in the following way:

* If the algo finds a new best candidate far away from the previous reference point, it will increase sigma, with the logic being that the algorithm actually is looking at too small a region of space if no candidates were found nearby
* If the algo finds new points very close to the reference, then we're likely aproaching a local minima, and we can reduce sigma to search points near our reference point with more fine resolution.

## Why this routine?
I needed an optimization routine that could work in high-dimensional space where dimension reduction preprocessing wasn't feasible. Although the algo isn't fast, it's robust enough to get close to local minima even in very high dimensions (5-10k + dimensions). I also added 

## Where does the algo fall short?
* The routine isn't fast
* The routine isn't very efficient, and requires many input evaluations
* The routine doesn't necessarily converge to global minima
