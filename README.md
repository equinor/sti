# _sti_ 

_sti_ is an investigation into the feasibility of using deep neural networks for local planning problems in well trajectory engineering. As such, all of the code is currently experimental, and not intended for use in production.

The problem of interest is as follows:  

Given start and end states as `(north, east, tvd, inc, azi)` and a [dog leg severity](https://www.glossary.oilfield.slb.com/en/Terms/d/dog_leg.aspx) limitation, find the shortest possible trajectory from start to end state obeying the dog leg severity limitation.

An efficient solution to this problem can be used in a global planner, such as [RRT/RRT\*](https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree).

_sti_ approximates an optimal solution to this problem by using a deep neural network trained on sample solutions to the problem from numerical optimization.

The name _sti_ comes from the Norwegian word "sti", which translates into "trail" or "path" in English."

# Key ideas
_sti_ makes an _Ansatz_ that an optimal extension to 3D of [Dubin's path](https://en.wikipedia.org/wiki/Dubins_path) is also three segments. Internally, _sti_ works by finding two intermediate points that when tied together is an optimal path from the start to the target state.

## Parametrization
In the current codebase, the two points are `(north, east, tvd)` triplets, tied together by using the dogleg toolface method, producing either straight lines or circular arcs. 

When connecting the points, dogleg toolface parameters are selected such that arrival at the target position is guaranteed, while azimuth, inclination and dog leg severity may be violated.

Abandonded approaches to parametrization of intermediate points are:

* Intermediate points as `(inc, azi)`, tied together using minimum curvature. Abandonded due to problem with non-uniqueness of projection method, many local minima in optimization when producing training data and no arrival guarantee.
* Intermediate points as `(toolface angle, dls, md step)`. Terrible performance in optimization when producing training data due to toolface angle and dls neutralizing each other when 0 or less. Also no arrival guarantee.

## Standardized problem
To achive maximum sample efficiency in the training data, the dimensionality of the problem is reduced by transforming it to standardized format. This is done by:

* Translating the co-ordinate system so that the start position is always at `(0,0,0)`.
* Rotating the co-ordinate system so that
  * Start inclination and azimuth is `(0,0)`, i.e. straight down. This is done by using the bit direction as the `tvd` direction unit vector in the rotation.
  * The target location is always in the north-tvd plane with positive north value. This is done by orthoganalizing & normalizing the vector difference from start to target location wrt. the tvd as defined above, and let this be the `north` direction unit vector in the rotation.
  * The `east` direction is then defined by their cross product, and a check for positive dot product with the target bit direction, enforcing a target azimuth in [0, pi]. ***Note:*** This convention produces both left & right handed systems. In the preliminary results, it appears that the approach taken to calculate the optimal path is not sensitive to this - but more testing should be done.

## Producing training data
To produce data training data for a deep neural network, the local planning problem is solved repeatedly using numerical optimization with randomized input. To achive maximum sample efficiency, the standardized problem is sampled.

To speed up the process, a preliminary neural net is used as an initial guess for the optimization algorithm.

The training data can be boosted by inverting the problem so that start and target are flipped. Note that their directions must also be changed. See [`scripts/reverse_training_data.py`](scripts/reverse_training_data.py) for a preliminary sketch. 

***Note:*** The data is no longer [i.i.d](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) when using this approach. Hence, special care should be taken when testing models trained on boosted data. This is not implemented in the preliminary pipeline.

## Training a neural network model
See [`scripts/mlp_model.py`](scripts/mlp_model.py) for a preliminary training pipeline. Sample data available in [`data/merged`](data/merged).

## API
The code base is currently at an experimental stage - thus no API is provided. No stability of method names and signatures should be assumed.

## Contributing
You're welcome. Add issues, make PRs.

## License
LGPL-3