# sti 

sti is a small Python package that solves local planning problems in well trajectory engineering.

Given start and end states as (north, east, tvd, inc, azi) and a dog leg severity (curvature) limitation, find the shortest possible path from start to end obeying the dog leg severity limitation.

Sti approximates an optimal solution to this problem by provding a a triplet of (inc, azi, md) that when used with the minimum curvature provides a close to optimal solution to the local planning problem.

The name comes from the Norwegian word "sti", which translates into "trail" or "path" in English."

# Usage
sti is in very early development, and a user centric Python API is not yet available

# Precision mode
When in precision mode, sti will use optimization to solve the planning problem. The method works as follows:
1. Obtain an initial estimate by using approximate mode (see below).
2. Use local optimization using the initial estimate as starting point.
3. If local optimization does not find an acceptable soltion, precondition the problem by trying to find a solution that reaches the target position with arbitrary angles, and restart the local optimization with this solution as starting point.
4. If the local optimization fails after preconditioning, revert to global optimization using dual annealing.

If precision of the estimated (inc, azi, md) tuplets are not critical, e.g. when using sit in conjuction with a global planner, it is also possible to use sti in approximate mode.

# Approximate mode
In approximate mode, sti uses a linear model trained on results from precision runs to estimate the (inc, azi, md) tuplets. When projected using minimum curvature, this tuplet will not neccessary solve the local planning problem as formulated above. However, if this is not of critical importance, e.g. when using RRT/RRT-star or similar to solve a global planning problem, this mode is many orders of magnitude more efficient.
