# _sti_ 

_sti_ is a small Python package that solves local planning problems in well trajectory engineering.

**The Problem:** Given start and end states as (north, east, tvd, inc, azi) and a dog leg severity (curvature) limitation, find the shortest possible path from start to end obeying the dog leg severity limitation.

_sti_ approximates an optimal solution to this problem. The name comes from the Norwegian word "sti", which translates into "trail" or "path" in English."

# Internals
_sti_ makes an _Ansatz_ that an optimal extension to 3D of [Dubin's path](https://en.wikipedia.org/wiki/Dubins_path) is also three segments. Internally, _sti_ works by finding two optional helper points in (north, east, tvd), that when tied togheter using the dogleg toolface method will satisfy the given constraints.

# Usage
_sti_ is in very early development, and a user centric Python API is not yet available