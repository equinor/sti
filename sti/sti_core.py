import numpy as np
import os
from scipy.optimize import minimize, Bounds, dual_annealing#, NonlinearConstraint


# For loading a linear model for guessworking
import pickle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# For QC toolface method
import matplotlib.pyplot as plt
from random import random

"""
Given a start and end state in a space of (north, east, tvd, inc, azi),
and a dog leg severity (dls) limit, sti is a set of three inc, azi & md_inc that
connect the start and end state using the minimum curvature method, as short as possible
within the dls limit.

The code is intentionally very low level, to attempt using automatic differentation.

The sti data is for efficiency reason passed around as a numpy array, layed out as follows:
[inc0, inc1, inc2, azi0, azi1, azi2, md_inc0, md_inc1, md_inc2]

State data is layed out as follows, again for effciency reasons:
[north, east, tvd, inc, azi]
"""

def faststi(start_state, target_state, dls_limit=0.002, tol=1e-3, scale_md=100, md_weight=1/1000):
    """ Fit a sti from start_state to target_state """
    VERBOSE = True
    METHOD = 'L-BFGS-B'
    # METHOD = 'trust-constr'

    # Bounds
    lb, ub = __get_sti_bounds()
    bounds = Bounds(lb, ub, keep_feasible=True)
    
    # Objective for first global optimization
    of_feas = __get_optifuns_feasibility(start_state, target_state, dls_limit, scale_md, md_weight)

    if VERBOSE:
        print("\nInitializing problem.")
        print("\nAttempting model prediction & local optimization.")
    # First attempt to fit with L-BFGS-B if the problem is simple.
    # Seems to work only for straight lines
    x0 = __inital_guess(start_state, target_state)
    result = minimize(of_feas, x0, bounds=bounds, method=METHOD)#, options={'iprint': 0, 'ftol' : np.finfo(float).eps})

    sti = result.x
    projection, dls = project_sti(start_state, sti)
    err = __err_squared_state_mismatch(target_state, projection, dls_limit, scale_md)


    # Try to precondition the problem by solving a simpler one
    if(err > 0.1):
        if VERBOSE:
            print("Error above threshold. Attempting precondintioning.")
        x0 = sti
        of_precond = __get_optifuns_precondition(start_state, target_state, dls_limit, scale_md, md_weight)
        result = minimize(of_precond, x0, bounds=bounds, method=METHOD) #, options={'iprint': 0})

        # Input from precdond as initial guess for full constraint problem
        x0 = result.x
        result = minimize(of_feas, x0, bounds=bounds, method=METHOD) #, options={'iprint': 0, 'ftol' : np.finfo(float).eps})

        sti = result.x

        projection, dls = project_sti(start_state, sti)
        err = __err_squared_state_mismatch(target_state, projection, dls_limit, scale_md)

    if(err > 0.1):
        if VERBOSE:
            print("Error above threshold. Proceeding to global optimization")
        bounds_list = list(zip(lb, ub))

        result = dual_annealing(of_feas, bounds_list)
        sti = result.x

        projection, dls = project_sti(start_state, sti)
        err = __err_squared_state_mismatch(target_state, projection, dls_limit, scale_md)
        if VERBOSE:
            print("Final error: ", err)


    # TODO 'trust-constr' method is broken when using numdiff in current scipy, below code will not work
    # # # Local fine tuning
    # x0 = __truncate_to_bounds(sti, lb, ub, 1e-1)

    # of_min = __get_optifuns_min_length(scale_md)
    # nlc = __get_non_linear_constraint(start_state, target_state, dls_limit,scale_md, tol)

    # result = minimize(of_min, x0, method='trust-constr', bounds=bounds, constraints=nlc, options={'verbose':1})
    # sti = result.x

    return sti, err


def project_sti(start_state, sti):
    """ Project the sti using as root"""
    dls = np.array([0.]*3)
    dnorth = np.array([0.]*3)
    deast = np.array([0.]*3)
    dtvd = np.array([0.]*3)
    inc = np.array([0.]*3)
    azi = np.array([0.]*3)
   
    if(np.any(sti[6:9] <= 1e-5)):
        raise ValueError("All md increments must be positive.")

    # Ugly, but hopefully fast and auto-differentiable
    dnorth[0], deast[0], dtvd[0], dls[0] = min_curve_segment(start_state[3], start_state[4], sti[0], sti[3], sti[6])
    dnorth[1], deast[1], dtvd[1], dls[1] = min_curve_segment(sti[0], sti[3], sti[1], sti[4], sti[7])
    dnorth[2], deast[2], dtvd[2], dls[2] = min_curve_segment(sti[1], sti[4], sti[2], sti[5], sti[8])

    p_north = start_state[0] + sum(dnorth)
    p_east = start_state[1] + sum(deast)
    p_tvd = start_state[2] + sum(dtvd)
    p_inc = sti[2]
    p_azi = sti[5]

    return (p_north, p_east, p_tvd, p_inc, p_azi), dls


def __inital_guess(state_from, state_to, linear_model=True):
    """ Produce an inital guess. Will try to use a linear model if available."""

    # TODO Should keep the model in memory between runs. Refactor me.
    if linear_model is True:
        # Try first using a linear model from previous runs
        # TODO Improve me, this is not exactly beautiful
        lm_file = 'linear-mod.sav'
        basedir = os.path.abspath(__file__)
        basedir = os.path.dirname(basedir) 
        pickled_lm_file = os.path.join(basedir,'../models/', lm_file)
        pickled_lm_file = os.path.normpath(pickled_lm_file)

        print("Using linear model to initialize optimization.")
        reg_mod = pickle.load(open(pickled_lm_file, 'rb'))
        reg_x = np.append(state_from, state_to).flatten()
        reg_sti = reg_mod.predict(reg_x.reshape(1, -1)).flatten()
        x0 = reg_sti
    else:
        # We'll need to guess, pure black magic...
        dnorth = state_to[0] - state_from[0]
        deast = state_to[1] - state_from[1]
        dtvd = state_to[2] - state_from[2]

        r = (dnorth**2 + deast**2 + dtvd**2)**(0.5)

        inc_f = state_from[3]
        inc_t = state_to[3]
        azi_f = state_from[4]
        azi_t = state_to[4]

        azi_m = azi_t 

        if deast !=0 and dnorth ==0:
            if deast > 0:
                azi_m = np.pi /2
            else:
                azi_m = 2*np.pi*3/4
        
        inc_m = np.arccos(dtvd / r)

        if dnorth != 0:
            azi_m = np.arctan(deast/dnorth)

        x0 = np.array([inc_m/2, inc_m, inc_t, azi_m/2, azi_m, azi_t, r/3, r/3, r/3])

    return x0 


def __truncate_to_bounds(sti, lb, ub, eps=0):
     # Truncate small negative violations from scipy optimize
     # eps is a workaround for https://github.com/scipy/scipy/issues/11403
     tsti = np.maximum(sti, lb + eps)
     tsti = np.minimum(tsti, ub - eps)
     return tsti
    
def __err_squared_pos_mismatch(state1, state2, scale_md):
    """ Error in bit position without consideration for orientation. """
    d2north = (state1[0] - state2[0])**2 
    d2east = (state1[1] - state2[1])**2
    d2tvd = (state1[2] - state2[2])**2

    terr = (d2north + d2east + d2tvd) / (scale_md**2)

    return terr


def __err_squared_orient_mismatch(state1, state2, dls_limit):
    """ Error in bit orientation without consideration for position. """
    d2inc = (state1[3] - state2[3])**2 / (dls_limit**2)
    d2azi = (state1[4] - state2[4])**2 / (dls_limit**2)

    # If abs(dazi) > pi, dazi is not minimal. Correct it.
    dazi = abs(state1[4] - state2[4])
    if dazi > np.pi:
        dazi = 2*np.pi - dazi
        
    d2azi = dazi**2 / (dls_limit**2)
    terr = d2inc + d2azi

    return terr


def __err_squared_state_mismatch(state1, state2, dls_limit, scale_md):
    """ Error in position and orientation"""
    perr = __err_squared_pos_mismatch(state1, state2, scale_md)
    oerr = __err_squared_orient_mismatch(state1, state2, dls_limit)

    return perr + oerr


def __err_dls_mse(start_state, sti, dls_limit, scale_md):
    """ Return the sum of squares of dog leg severity above the dls_limit scaled by dls_limit"""
    projected_state, dls = project_sti(start_state, sti)
    dls_mis = (np.maximum(dls, dls_limit) - dls_limit) / dls_limit
    dls_mis = dls_mis ** 2

    return sum(dls_mis)


def __err_tot_md_sq(sti, scale_md):
    md = (sti[6] + sti[7] + sti[8]) / scale_md
    return md**2


def __get_optifuns_precondition(start_state, target_state, dls_limit, scale_md, md_weight):
    """ Simpler objective function.

        To be used as a preconditioner to find an initial guess before proceeding to global
        optimization. Position is weighted more than orientation, length and dls.
    """

    def of_precondition(sti):
        WEIGHT_POS = 1.0
        WEIGHT_ORI = 0.0
        WEIGHT_DLS = 0.1
        WEIGHT_MD  = 0.0

        projected_state, dls = project_sti(start_state, sti)

        sq_pos_err = __err_squared_pos_mismatch(projected_state, target_state, scale_md)
        sq_ori_err = __err_squared_orient_mismatch(projected_state, target_state, dls_limit)
        sq_dls_err = __err_dls_mse(start_state, sti, dls_limit, scale_md)
        sq_tot_md_err = __err_tot_md_sq(sti, scale_md)

        terr = WEIGHT_POS*sq_pos_err + WEIGHT_ORI*sq_ori_err + WEIGHT_DLS*sq_dls_err + WEIGHT_MD*sq_tot_md_err

        return terr

    return of_precondition

def __get_optifuns_feasibility(start_state, target_state, dls_limit, scale_md, md_weight):

    def of_feasibility(sti):
        projected_state, dls = project_sti(start_state, sti)

        sq_state_err = __err_squared_state_mismatch(projected_state, target_state, dls_limit, scale_md)
        sq_dls_err = __err_dls_mse(start_state, sti, dls_limit, scale_md)
        sq_tot_md_err = __err_tot_md_sq(sti, scale_md)

        # Additional term that penalizes solution that are much longer in md
        # the the approximate distance from start to target.
        # 
        # This is to try to avoid very creative solutions by global optimisers

        # Note, we use scale MD = 1 to compare directly with the sti total md
        app_dist = (__err_squared_state_mismatch(start_state, projected_state, dls_limit=dls_limit, scale_md=1))**(0.5)
        md = sti[6] + sti[7] + sti[8]

        sq_dist_err = (max(md, 2*app_dist) - 2*app_dist)**2

        return sq_state_err + sq_dls_err + sq_tot_md_err * md_weight + sq_dist_err

    return of_feasibility


def __get_optifuns_min_length(scale_md):
    def of_len(sti):
        return __err_tot_md_sq(sti, scale_md)
    
    return of_len


def __get_non_linear_constraint(start_state, target_state, dls_limit, scale_md, tol, keep_feasible=False):
    def nlc_fun(sti):
        projected_state, dls = project_sti(start_state, sti)
        sq_state_err = __err_squared_state_mismatch(projected_state, target_state, dls_limit, scale_md)
        
        # Super explict, hoping for autodiff
        return (dls[0], dls[1], dls[2], sq_state_err)
    
    nlc_lb = np.array([-np.inf]*4)
    nlc_ub = np.array([dls_limit]*3)
    nlc_ub = np.append(nlc_ub, tol**2).flatten()
    
    nlc = NonlinearConstraint(nlc_fun, nlc_lb, nlc_ub, keep_feasible=keep_feasible)

    return nlc


def __get_sti_bounds():
    """ Bounds on free variables. Super explicit in hope of JAX compatibility."""
    min_md = 1 
    max_md = 10000

    lb = np.array([0, 0, 0, 0, 0, 0, min_md, min_md, min_md])
    ub = np.array([np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi, max_md, max_md, max_md])

    assert(len(ub) == 9)
    assert(len(lb) == 9)

    return lb, ub


def min_curve_segment(inc_upper, azi_upper, inc_lower, azi_lower, md_inc):
    """Inner workhorse, designed for auto differentiability."""
    # Stolen from wellpathpy
    cos_inc = np.cos(inc_lower - inc_upper)
    sin_inc = np.sin(inc_upper) * np.sin(inc_lower)
    cos_azi = 1 - np.cos(azi_lower - azi_upper)

    dogleg = np.arccos(cos_inc - (sin_inc * cos_azi))

    # ratio factor, correct for dogleg == 0 values
    if np.isclose(dogleg, 0.):
        dogleg = 0
        rf = 1
    else:
        rf = 2 / dogleg * np.tan(dogleg / 2)
    
    # delta northing
    upper = np.sin(inc_upper) * np.cos(azi_upper)
    lower = np.sin(inc_lower) * np.cos(azi_lower)
    dnorth = md_inc / 2 * (upper + lower) * rf
    
    # delta easting
    upper = np.sin(inc_upper) * np.sin(azi_upper)
    lower = np.sin(inc_lower) * np.sin(azi_lower)
    deast = md_inc / 2 * (upper + lower) * rf

    # delta tvd
    dtvd = md_inc / 2 * (np.cos(inc_upper) + np.cos(inc_lower)) *rf

    dls = dogleg / md_inc

    return dnorth, deast, dtvd, dls


def dogleg_toolface(inc_upper, azi_upper, toolface, dls, md_inc):
    """
    Use toolface and dls as input. 

    Benefit: Stabilize cornercases of minimum curvature, e.g. when
    going from (0,0) to (np.pi ,0) of length md, where min. curve is
    not well defined.

    Using formulas (6) & (7) from
    https://docs.oliasoft.com/technical-documentation/well-trajectory-design

    Special cases for inc=0 and inc=pi.

    For inc 0 and pi, standard magnetic toolface is used

    Examples:
        inc=0, toolface=pi    -> Down/Upwards south 
        inc=0, toolface=0     -> Down/Upwards north
        inc=0, toolface=pi/2  -> Down/Upwards east
        inc=0, toolface=3pi/2 -> Down/Upwards west
    
    This is done to be able to translate the co-ordinate system with
    out too much brain hurt.

    How to transform toolface:
        Gravity toolface: Plane defined by local (inc, azi) with proj( (0,0,-1)) as north
        Magnetic toolface: Angle in plane defined by  (0, 0, 1) vs (1,0,0) as north

        Process:
        Express toolface direction as inc, azi, need wellpath inc,azi & tf.
        Transform tf (inc, azi) -> (n, e, t)
        Rotate (n, e, t)
        Transform tf (n, e, t) -> (inc, azi)
        (Transform wellpath inc/azi using same rotations)
        Using transformed wellpath inc,azi and tf as inc azi, calculate new tf.

    """

    if toolface < 0 or dls == 0.0:
        # Go straight ahead
        dnorth, deast, dtvd, dls = min_curve_segment(inc_upper, azi_upper, inc_upper, azi_upper, md_inc)
        return np.array((dnorth, deast, dtvd, inc_upper, azi_upper, toolface))
    else:
        # TODO Split in multiple calls to handle dls * md_inc > pi
        return __dogleg_toolface_inner(inc_upper, azi_upper, toolface, dls, md_inc)


def __dogleg_toolface_inner(inc_upper, azi_upper, toolface, dls, md_inc):
    """
    Inner method used by dogleg_toolface to handle arc angles larger
    than pi.

    Assumes dls>0 and dls * md_inc < pi.
    """
    # assert(dls > 0)
    # assert(dls * md_inc < np.pi)

    theta = dls * md_inc
    sin_tf = np.sin(toolface)
    cos_tf = np.cos(toolface)

    # Using formulas (6) & (7) from
    # https://docs.oliasoft.com/technical-documentation/well-trajectory-design
    partial_1 = np.cos(inc_upper) * np.cos(theta)
    partial_2 = cos_tf * np.sin(inc_upper)*np.sin(theta)

    inc_lower = np.arccos(partial_1 - partial_2)
    
    partical_3 = sin_tf * np.sin(theta)  

    delta_azi = 0.0
    # Handle discountinuites by computing both sin & cos deltas, as references above
    # and using arctan2 to compute the change in azimuth
    if np.sin(inc_lower) != 0 and np.sin(inc_upper) != 0:
        tan_delta = sin_tf * np.tan(theta) / (np.sin(inc_upper) + np.cos(inc_upper) * cos_tf * np.tan(theta))
        sin_theta = np.sin(theta)
        sin_delta = sin_tf * sin_theta / np.sin(inc_upper)

        # Handle tan_delta = 0
        if tan_delta != 0:
            cos_delta = sin_delta / tan_delta
        else:
            cos_delta = np.cos(azi_upper)

        delta_azi = np.arctan2(sin_delta, cos_delta)
    else:
        # By definition, there cannot be any change of azimuth if it 
        # start or ends with sin(inc) = 0 
        # TODO Think hard about this and write a better explanation
        delta_azi = 0.0
    
    azi_lower = azi_upper + delta_azi

    # We work with azi in [0, 2pi]
    if azi_lower < 0:
        azi_lower = azi_lower + 2*np.pi


    # Handle inc_upper 0 or pi
    # In these cases, the magnetic toolface is used intially
    # In practice, the result are curves in a specific azimuthal direction,
    # that is only build, no turn.
    if inc_upper == 0 or inc_upper == np.pi:
        azi_lower = toolface
    
    dnorth, deast, dtvd, dls = min_curve_segment(inc_upper, azi_upper, inc_lower, azi_lower, md_inc)

    # Handle cases where we switch form magnetic to gravity toolface and so on
    # 
    # To to this, we need to return the toolface at the end of the segment
    # For magnetic->gravity tf at end of segment will always be 
    #   0 if inc_upper is 0 (only build angle)
    #   pi if inc_upper is pi (only drop angle)
    #
    # To get to a situation were we switch form gravity to magentic toolface using 
    # the dogleg toolface method, we've had to drill with no turn. Hence, in this case,
    # the toolface at the end is actually the opposite direction of the azimuth at the start
    tf_lower = toolface

    # Magnetic to gravity toolface
    if inc_upper == 0:
        tf_lower = 0
    elif inc_upper == np.pi:
        tf_lower = np.pi

    # Gravity to magnetic toolface
    if inc_lower == 0 or inc_lower == np.pi:
        tf_lower = azi_upper + 2*np.pi
        if tf_lower > 2*np.pi:
            tf_lower = tf_lower - 2*np.pi
    
    return np.array((dnorth, deast, dtvd, inc_lower, azi_lower, tf_lower))
        

if __name__ == '__main__':
    
    tf = np.linspace(0,2*np.pi, 100)
    inc = np.copy(tf)
    azi = np.copy(tf)
    md = np.linspace(1,5000, 1000)
    n = np.copy(md)
    e = np.copy(md)
    t = np.copy(md)
    inc = np.copy(md)
    azi = np.copy(md)
    dls = 0.002
    inc_upper = np.pi/2 #random()*np.pi
    azi_upper = random()*2*np.pi
    toolface = random()*2*np.pi

    print("inc: ", inc_upper)
    print("azi: ", azi_upper)
    print("tf: ", toolface)

    for i in range(0, len(md)):
        n[i], e[i], t[i], inc[i], azi[i], _ = dogleg_toolface(inc_upper, azi_upper, toolface, dls, md[i])

    # plt.plot(tf, azi)
    plt.figure()
    plt.subplot(121)
    plt.plot(md, n, 'r')
    plt.plot(md, e, 'g')
    plt.plot(md, t, 'b')
    plt.subplot(122)
    plt.plot(md, inc, 'k')
    plt.plot(md, azi, 'y')
    plt.show()

    # theta = np.linspace(0, 2*np.pi, 1000)
    # val_tan = np.copy(theta)
    # val_sin = np.copy(theta)

    # for i in range(0, len(theta)):
    #     val_tan[i] = tan_delta(theta[i], toolface, inc_upper)
    #     val_sin[i] = sin_delta(theta[i], toolface, inc_upper)
    
    # val_cos = val_sin / val_tan

    # val_f = np.arctan2(val_sin, val_cos)

    # for i in range(0, len(val_f)):
    #     if val_f[i] < 0:
    #         val_f[i] = val_f[i] + 2*np.pi

    # val_tan = np.arctan(val_tan)

    # plt.figure()
    # plt.plot(theta, val_tan, 'r')
    # plt.plot(theta, val_f, 'b')
    # plt.show()
    data = dogleg_toolface(np.pi/2, np.pi, 0, 0.002, 300)
    print(data)