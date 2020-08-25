import numpy as np
from scipy.optimize import minimize, Bounds, dual_annealing, shgo
from sti.dogleg_tf import dogleg_toolface, get_params_from_state_and_net
from sti.utils import pos_from_state, cart_bit_from_state, translate_state, proj, orthogonalize, l2norm,\
                         normalize, net_to_spherical

# Model loading
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

# QC & Demo
from random import random


def find_sti(start_state, target_state, dls_limit, scale_md=4000):
    """
    Find a sti from start to target within dls limitation.

    Scale md: Float in the order of magnitude of md in the problem.

    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state
    target_state : nd.array
        (north, east, tvd, inc, azi) of target state
    dls_limit : float
        maximum dog leg severity alllowed
    scale_md : float
        positive float used to scale md to ~1 in the objective function


    Returns
    -------
    sti : nd.array
        two intermediate cartesian positions layed out as 
        (n0, e0, t0, n1, e1, t1)
    acceptable : bool
        False if the solution is outside acceptable bounds,
    """

    # Standardize problem
    start_stand, target_stand = standardize_problem(start_state, target_state)
    x0 = standardized_initial_guess(start_stand, target_stand, dls_limit)

    # Solve in standard space
    stand_sti, acceptable = find_sti_opti(start_stand, target_stand, dls_limit, scale_md, x0)

    int_pos_0_stand = stand_sti[0:3]
    int_pos_1_stand = stand_sti[3:6]

    # Translate sti points back to physical space
    int_pos_0 = inverse_standardization_pos(start_state, target_state, int_pos_0_stand)
    int_pos_1 = inverse_standardization_pos(start_state, target_state, int_pos_1_stand)

    sti = np.append(int_pos_0, int_pos_1).flatten()

    return sti, acceptable


def find_sti_opti(start_state, target_state, dls_limit, scale_md, initial_guess):
    """ 
    Fit a sti from start_state to target_state using optimization


    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state
    target_state : nd.array
        (north, east, tvd, inc, azi) of target state
    dls_limit : float
        maximum dog leg severity alllowed
    scale_md : float
        positive float used to scale md to ~1 in the objective function
    initial_guess : nd.array
        two intermediate cartesian positions layed out as 
        (n0, e0, t0, n1, e1, t1)


    Returns
    -------
    sti : nd.array
        two intermediate cartesian positions layed out as 
        (n0, e0, t0, n1, e1, t1)
    acceptable : bool
        False if the solution is outside acceptable bounds,
     """

    VERBOSE = True
    METHOD = 'L-BFGS-B'

    DLS_EPS = 1e-4 # Proceed to global opt. if dls is overshooting (1e-4 ~0.2 degree/30m)
    INC_EPS = 1e-1 # Prooced to global opt. if azi error is larger (0.1 ~ 5 degrees)
    AZI_EPS = 1e-1 # Prooced to global opt. if inc error is larger (0.1 ~ 5 degrees)

    acceptable_solution = False

    lb, ub = get_bounds(start_state, target_state)
    bounds = Bounds(lb, ub)

    objective_function = get_objective_function(start_state, target_state,dls_limit, scale_md)

    if VERBOSE:
        md, dls_mis, inc_err, azi_err = get_error_estimates(start_state, target_state, dls_limit, scale_md, initial_guess)
        inc_mis = inc_err ** (0.5)
        azi_mis = azi_err ** (0.5)
        print("Errors in initial guess. dls_mis: ", dls_mis, " inc_mis: ", inc_mis, " azi_mis: ", azi_mis)


    if VERBOSE:
        print("Performing gradient based optimization.")

    result = minimize(objective_function, initial_guess, bounds=bounds, method=METHOD)#, options={'iprint': 99})
    sti = result.x

    md, dls_mis, inc_err, azi_err = get_error_estimates(start_state, target_state, dls_limit, scale_md, sti)

    qc_vals = [dls_mis < DLS_EPS, (inc_err) ** (0.5) < INC_EPS, (azi_err) ** (0.5) < AZI_EPS]
    # Global optimization if not accepted error
    if not all(qc_vals):
        if VERBOSE:
            print("Result not acceptable. Performing global optimization.")
            print("QC DLS, INC, AZI: ", qc_vals)

        result = dual_annealing(objective_function, bounds=list(zip(lb, ub)))
        sti = result.x


        md, dls_mis, inc_err, azi_err = get_error_estimates(start_state, target_state, dls_limit, scale_md, sti)
        qc_vals = [dls_mis < DLS_EPS, (inc_err) ** (0.5) < INC_EPS, (azi_err) ** (0.5) < AZI_EPS]


        if not all(qc_vals):
            acceptable_solution = False
            if VERBOSE:
                print("Result not acceptable - options exhausted. Beware.")
                print("QC DLS, INC, AZI: ", qc_vals)
        else:
            acceptable_solution = True
    else:
        acceptable_solution = True

    return sti, acceptable_solution


def get_objective_function(start_state, target_state, dls_limit, scale_md):
    """
    Create a problem specific objective function.


    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state
    target_state : nd.array
        (north, east, tvd, inc, azi) of target state
    dls_limit : float
        maximum dog leg severity alllowed
    scale_md : float
        positive float used to scale md to ~1 in the objective function


    Returns
    -------
    objective_function : function
    """


    def objective_function(sti):
        """
        Problem specific objective function


        Parameters
        ----------
        sti : nd.array
            two intermediate cartesian positions layed out as 
            (n0, e0, t0, n1, e1, t1)


        Returns
        ------
        of_val: float
            objective function value
        """


        WEIGHT_DLS = 1000
        WEIGHT_INC = 10000
        WEIGHT_AZI = 10000
        WEIGHT_REG = 100

        md, dls_mis, inc_err, azi_err = get_error_estimates(start_state, target_state, dls_limit, scale_md, sti)

        app_md = approximate_distance(start_state, target_state, dls_limit)

        regterm = (WEIGHT_REG * max(md-4*app_md, 0))**2

        of_val = (md/scale_md) ** 2 + (WEIGHT_DLS * dls_mis / dls_limit) ** 2 + WEIGHT_INC * inc_err + WEIGHT_AZI * azi_err + regterm
        return of_val

    return objective_function


def get_error_estimates(start_state, target_state, dls_limit, scale_md, sti):
    """
    Error estimates

    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state
    target_state : nd.array
        (north, east, tvd, inc, azi) of target state
    dls_limit : float
        maximum dog leg severity alllowed
    scale_md : float
        positive float used to scale md to ~1 in the objective function
    sti : nd.array
        two intermediate cartesian positions layed out as 
        (n0, e0, t0, n1, e1, t1)


    Returns
    -------
    md : float
        measured depth from start to target along sti
    dls_mis : float
        positive number if dls is overshooting its limit
    inc_err: float
        squared error in projected inclination vs objective
    azi_err: float
        squared error in projected azimuth vs objective
    """


    projected_state, dls, md = project_sti(start_state, target_state, sti)

    inc_err = (target_state[3] - projected_state[3])**2

    azi_err = abs(target_state[4] - projected_state[4])

    # Periodic error in azi
    if azi_err > np.pi:
        azi_err = abs(azi_err - 2*np.pi)

    azi_err = azi_err ** 2

    dls_mis = max(0, dls - dls_limit)

    return md, dls_mis, inc_err, azi_err


def project_sti(start_state, target_state, sti):
    """
    Project a sti from start to target resulting
    in a well trajectory


    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state
    target_state : nd.array
        (north, east, tvd, inc, azi) of target state
    sti : nd.array
        two intermediate cartesian positions layed out as 
        (n0, e0, t0, n1, e1, t1)


    Returns
    -------
    projected_state : nd.array
        (north, east, tvd, inc, azi) at end of trajectory
    max_dls : float
        maximum dog leg severity of trajectory
    md : float
        measured depth  from start to target along trajectory
    """

    # HACK Just thorwing something on the wall here to see if the 
    # new approach using intermediate points is better
    
    # Intermediate (north, east, tvd)
    intermed_net0 = sti[0:3]
    intermed_net1 = sti[3:6]

    # Find parameters from start to first intermediate point
    tf0, dls0, md0 = get_params_from_state_and_net(start_state, intermed_net0)
    intermed_state0 = dogleg_toolface(start_state[3], start_state[4], tf0, dls0, md0)

    # Dogleg toolface returns increments
    intermed_state0[0] = intermed_state0[0] + start_state[0]
    intermed_state0[1] = intermed_state0[1] + start_state[1]
    intermed_state0[2] = intermed_state0[2] + start_state[2]

    # Find paramters between intermediate states
    tf1, dls1, md1 = get_params_from_state_and_net(intermed_state0, intermed_net1)
    intermed_state1 = dogleg_toolface(intermed_state0[3], intermed_state0[4], tf1, dls1, md1)

    # Dogleg toolface returns increments
    intermed_state1[0] = intermed_state1[0] + intermed_state0[0]
    intermed_state1[1] = intermed_state1[1] + intermed_state0[1]
    intermed_state1[2] = intermed_state1[2] + intermed_state0[2]

    # Find paramters between intermediate states
    final_net = target_state[0:3]
    tf2, dls2, md2 = get_params_from_state_and_net(intermed_state1, final_net)

    # Max dls and total md
    max_dls = max(dls0, dls1, dls2)
    md = md0 + md1 + md2

    # Projected state
    projected_state = dogleg_toolface(intermed_state1[3], intermed_state1[4], tf2, dls2, md2)

    # Dogleg toolface returns increments
    projected_state[0] = projected_state[0] + intermed_state1[0]
    projected_state[1] = projected_state[1] + intermed_state1[1]
    projected_state[2] = projected_state[2] + intermed_state1[2]

    return projected_state, max_dls, md 


def get_bounds(start_state, target_state):
    """
    Lower and upper bounds for optimization
    algorithm.


    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state
    target_state : nd.array
        (north, east, tvd, inc, azi) of target state


    Returns
    -------
    lb : nd.array
        lower bounds for optimization, shape (6,)
    ub : nd.array
        upper bounds for optimization, shape (6,)
    """

    # HACK Just throwing in some temporary stuff. But seems to work fine,
    # and performance is not super senstive (tried with stricter bounds)
    lb = np.array([-1.0e4]*6)
    ub = np.array([1.0e4]*6)

    return lb, ub


def initial_guess(start_state, target_state, dls_limit):
    """
    Dummy initial guess for optimization

    Used to bootstrap samples for the ML model

    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state
    target_state : nd.array
        (north, east, tvd, inc, azi) of target state
    dls_limit : float
        maximum dog leg severity alllowed


    Returns
    -------
    sti : nd.array
        two intermediate cartesian positions layed out as 
        (n0, e0, t0, n1, e1, t1)

    """
    # Dummy approach, just put two points betweeen start and end

    dn = target_state[0] - start_state[0]
    de = target_state[1] - start_state[1]
    dt = target_state[2] - start_state[2]

    n0 = start_state[0]
    e0 = start_state[1]
    t0 = start_state[2]

    intermed0 = np.array([n0 + dn/3, e0 + de/3, t0 + dt/3])
    intermed1 = np.array([n0 + 2*dn/3, e0 + 2*de/3, t0 + 2*dt/3])

    sti = np.append(intermed0, intermed1).flatten()

    return sti

def standardized_initial_guess(start_state, target_state, dls_limit):
    """
    Machine learning based initial guess for optimization

    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state
    target_state : nd.array
        (north, east, tvd, inc, azi) of target state
    dls_limit : float
        maximum dog leg severity alllowed


    Returns
    -------
    sti : nd.array
        two intermediate cartesian positions layed out as 
        (n0, e0, t0, n1, e1, t1)
    
    Notes
    -----
    NOTE - This function assumes, but does NOT check for standardized
    problem formulation
    """

    # TODO Keep me in memory!
    with open('models/mlp.sav', 'rb') as file:
        model = pickle.load(file)

    # We're using standardized problem, start state is 0
    x = target_state
    x = np.append(x, dls_limit).flatten()

    sti = model.predict(x.reshape(1, -1))
    sti = sti.flatten()

    return sti


def approximate_distance(start_state, target_state, dls_limit):
    """
    Approximate distance, used with a multiplier for regularization
    of global optimization, which can sometimes produce strange results.

    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state
    target_state : nd.array
        (north, east, tvd, inc, azi) of target state
    dls_limit : float
        maximum dog leg severity alllowed

    Returns
    -------
    dist: float
        approximate measured depth from start to target


    Notes
    -----
    This is not a good approximate, should only be used for 
    regularization purposes, with a large (4 ~ 5) multiplier.
    """

    # We dont bother with exact arc angle here
    delta_inc = abs(target_state[3] - start_state[3])
    delta_azi = abs(target_state[4] - start_state[4])

    if delta_inc > np.pi:
        delta_inc = delta_inc - np.pi
    if delta_azi > 2*np.pi:
        delta_azi = delta_azi - 2*np.pi

    # Approximate stuff...
    delta_angle = (delta_inc**2 + delta_azi**2) ** (0.5)

    if delta_angle > 2*np.pi:
        delta_angle = delta_angle - 2*np.pi

    # Euclidian distance
    dn = target_state[0] - start_state[0]
    de = target_state[1] - start_state[1]
    dt = target_state[2] - start_state[2]

    euc_dist = (dn**2 + de**2 + dt**2) ** (0.5)

    dist = euc_dist + delta_angle / dls_limit

    return dist 


def get_stand_translation_vector(start_state):
    """
    Returns a translation vector for use in standardizing
    the problem.

    Use additatiove when standardizing, subtract when inverse.

    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state

    Returns
    -------
    vec: nd.array
        (north, east, tvd) translation

    """
    n = start_state[0]
    e = start_state[1]
    t = start_state[2]

    vec = -1.0 * np.array([n, e, t])

    return vec


def translate_problem(start_state, target_state):
    """
    Translate a problem so that start_state position
    is at (0, 0, 0). Note that orientation is preserved.

    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state
    target_state : nd.array
        (north, east, tvd, inc, azi) of target state

    Returns 
    -------
    new_start : nd.array
        (north, east, tvd, inc, azi) of translated start
    new_target : nd.array
        (north, east, tvd, inc, azi) of translated target
    """

    vec = get_stand_translation_vector(start_state)

    new_start = translate_state(start_state, vec)
    new_target = translate_state(target_state, vec)

    return new_start, new_target


def get_stand_rotation_matrix(start_state, target_state):
    """
    Get rotation matrix for standardizing the problem.

    Use as multplication for standardizing, matrix inverse
    to go back

    The rotation is calculated after translating the
    problem so that start location is at (0,0,0)

    The rotation is defined such that:
        - Start direction is with incliation and azimuth
          0, i.e. straight down.
        - The target east is co-ordinate is always zero
        - The target azimuth is always in [0 ,pi], i.e.
          pointing eastwards

    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state
    target_state : nd.array
        (north, east, tvd, inc, azi) of target state

    Returns 
    -------
    A : nd.array
        (3 x 3) orothoganal rotation matrix
    """

    start, target = translate_problem(start_state, target_state)

    # TVD unit vector
    t = cart_bit_from_state(start)

    # North, target position orothoganilzed on TVD
    n = pos_from_state(target)
    n = orthogonalize(t, n)

    # Catch target paralell to bit direction
    if l2norm(n) == 0.0:    
        n = np.array([1.0, 0.0, 0.0])
        n = orthogonalize(t, n)

        # And a second failsafe if also both a north
        if l2norm(n) == 0.0:    
            n = np.array([0.0, 0.0, -1.0])
            n = orthogonalize(t, n)

    n = normalize(n)

    # East
    e = np.cross(t, n)
    e = normalize(e)

    # This stuff is a bit sketchy... it will
    # create a left hand system. 
    #
    # Works fine when translating points etc.
    # Might break other stuff
    #
    # The good thing about it is that we are guaranteed
    # target bit azi in [0, pi], doubling our sampling
    # efficiency

    bit_target = cart_bit_from_state(target)

    if np.dot(bit_target, e) < 0.0:
        e = -1.0 * e

    A = np.array([n, e, t])

    return A


def inverse_standardization_pos(start_state, target_state, pos):
    """
    Take a (n, e, t) point in the standard space defined
    by start and target and transform back to the physical
    space

    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state
    target_state : nd.array
        (north, east, tvd, inc, azi) of target state
    pos : nd.array
        (north, east, tvd) position in standard space

    Returns
    -------
    pos_inverse : nd.array
        (north, east, tvd) position in physical space
    """
    trans = get_stand_translation_vector(start_state)
    A = get_stand_rotation_matrix(start_state, target_state)

    pos_derot = np.linalg.solve(A, pos)
    pos_inverse = pos_derot - trans

    return pos_inverse


def standardize_pos(start_state, target_state, pos):
    """
    Take a (n, e, t) position and transform to
    system defined by standardization of 
    start and target states.

    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state
    target_state : nd.array
        (north, east, tvd, inc, azi) of target state
    pos : nd.array
        (north, east, tvd) position in physical space

    Returns
    -------
    pos_inverse : nd.array
        (north, east, tvd) position in standardized space
    """

    trans = get_stand_translation_vector(start_state)
    
    start, target = translate_problem(start_state, target_state)
    A = get_stand_rotation_matrix(start, target)

    stand_pos = pos + trans
    stand_pos = np.dot(A, stand_pos)

    return stand_pos



def standardize_problem(start_state, target_state):
    """
    Take a start and target state and transform
    to standardized space, i.e.:

        - Start location at (0,0,0)
        - Start inc & azi at (0,0)
        - Target east co-ordinate at 0
        - Target north co-ordinate always positive
        - Target azimuth in [0, pi]

    Parameters
    ----------
    start_state : nd.array
        (north, east, tvd, inc, azi) of start state
    target_state : nd.array
        (north, east, tvd, inc, azi) of target state

    Returns 
    -------
    standard_start : nd.array
        (north, east, tvd, inc, azi) of standardized start
    standard_target : nd.array
        (north, east, tvd, inc, azi) of standardized targret
    """
    
    start, target = translate_problem(start_state, target_state)

    A = get_stand_rotation_matrix(start, target)

    # These are actually defined by the rotation, but
    # calculating for QC
    # start_pos = pos_from_state(start)
    # start_bit = cart_bit_from_state(start)

    target_pos = pos_from_state(target)
    target_bit = cart_bit_from_state(target)

    # QC only
    # start_pos_rot = np.dot(A, start_pos)
    # start_bit_rot = np.dot(A, start_bit)

    target_pos_rot = np.dot(A, target_pos)
    target_bit_rot = np.dot(A, target_bit)

    # # QC only
    # start_inc_azi = net_to_spherical(start_bit_rot[0], start_bit_rot[1], start_bit_rot[2])

    target_inc_azi = net_to_spherical(target_bit_rot[0], target_bit_rot[1], target_bit_rot[2])

    # QC only
    # standard_start = np.append(start_pos_rot, start_inc_azi).flatten()

    # By definition
    standard_start = np.zeros(5)
    standard_target = np.append(target_pos_rot, target_inc_azi).flatten()

    return standard_start, standard_target


# TODO Move to tests
def demo_standardization():
    for i in range(0, 100):

        # Config
        dls = 0.0005 + 0.005*random() # From 0.85 to 9.45 degree / 30m 
        scale_md = 4000

        s_n = (-3+6*random()) * 1000
        s_e = (-3+6*random()) * 1000
        s_t = (-3+6*random()) * 1000
        s_inc = np.pi * random()
        s_azi = 2*np.pi * random()

        start_state = np.array([s_n, s_e, s_t, s_inc, s_azi])

        t_n = (-3+6*random()) * 1000
        t_e = (-3+6*random()) * 1000
        t_t = (-3+6*random()) * 1000
        t_inc = np.pi * random()
        t_azi = 2*np.pi * random()

        target_state = np.array([t_n, t_e, t_t, t_inc, t_azi])

        sti = find_sti(start_state, target_state, dls, scale_md)

        # Check projected state
        projected_state, dls, md = project_sti(start_state, target_state, sti)

        print("Target: ", target_state)
        print("Projected: ", projected_state)
        print("MD: ", md)
        print("DLS: ", dls)

