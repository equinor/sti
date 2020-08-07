import numpy as np
from scipy.optimize import minimize, Bounds, dual_annealing#, NonlinearConstraint
from random import random

# Storing data and profiling
from datetime import datetime
import csv

# For loading a linear model for guessworking
import pickle
from sklearn.linear_model import LinearRegression

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
        print("\nCurrent target:")
        print("---------------------")
        print_state(target_state)
        print("\nAttempting simple guess & local optimization.")
    # First attempt to fit with L-BFGS-B if the problem is simple.
    # Seems to work only for straight lines
    x0 = __inital_guess(start_state, target_state)
    result = minimize(of_feas, x0, bounds=bounds, method=METHOD)#, options={'iprint': 0, 'ftol' : np.finfo(float).eps})

    sti = result.x
    projection, dls = __project(start_state, sti)
    err = __err_squared_state_mismatch(target_state, projection, dls_limit, scale_md)


    # Try to precondition the problem by solving a simpler one
    if(err > 0.1):
        if VERBOSE:
            print("Target not found. Attempting precondintioning.")
        x0 = sti
        of_precond = __get_optifuns_precondition(start_state, target_state, dls_limit, scale_md, md_weight)
        result = minimize(of_precond, x0, bounds=bounds, method=METHOD) #, options={'iprint': 0})

        # Input from precdond as initial guess for full constraint problem
        x0 = result.x
        result = minimize(of_feas, x0, bounds=bounds, method=METHOD) #, options={'iprint': 0, 'ftol' : np.finfo(float).eps})

        sti = result.x

        projection, dls = __project(start_state, sti)
        err = __err_squared_state_mismatch(target_state, projection, dls_limit, scale_md)

    if(err > 0.1):
        if VERBOSE:
            print("Target not found. Proceeding to global optimization")
        bounds_list = list(zip(lb, ub))

        result = dual_annealing(of_feas, bounds_list)
        sti = result.x

        projection, dls = __project(start_state, sti)
        err = __err_squared_state_mismatch(target_state, projection, dls_limit, scale_md)
        if VERBOSE:
            print("ERR: ", err)


    # TODO 'trust-constr' method is broken when using numdiff in current scipy, below code will not work
    # # # Local fine tuning
    # x0 = __truncate_to_bounds(sti, lb, ub, 1e-1)

    # of_min = __get_optifuns_min_length(scale_md)
    # nlc = __get_non_linear_constraint(start_state, target_state, dls_limit,scale_md, tol)

    # result = minimize(of_min, x0, method='trust-constr', bounds=bounds, constraints=nlc, options={'verbose':1})
    # sti = result.x

    # print("Result from local optimization:")
    # print_sti(start_state, target_state, sti, dls_limit)

    return sti, err


def create_training_data(n_straight_down, n_step_outs_v, n_step_outs_h, n_below_slot, n_fully_random):
    """ Produce training data for fitting a neural net model."""

    filename = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = filename + ".csv"

    def print_header(text):
        print("\n\n#################################################################")
        print(text)
        print("#################################################################") 
    
    with open(filename,'x') as file:
        headers = __get_header()
        writer = csv.writer(file)
        writer.writerow(headers)


    for i in range(0,n_straight_down):
        print_header("Straight down")

        dls_limit = random()*0.003 + 0.0015
        start_state = [0, 0, 0, 0, 0]
        target_state = [0, 0, random()*2500, 0, 0]

        sti , err= faststi(start_state, target_state, dls_limit=dls_limit)
        print_sti(start_state, target_state, sti, dls_limit)
        print("\nState mismatch: ", "{:.4f}".format(err))

        data = __merge_training_data(start_state, target_state, dls_limit, sti)

        with open(filename,'a') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    for i in range(0, n_step_outs_v):
        print_header("Step out to vertical in N/E sector")

        dls_limit = random()*0.003 + 0.0015
        start_state = [0, 0, 0, 0, 0]
        target_state = [random()*750, random()*750, 2000+random()*2000, 0, 0]

        sti, err = faststi(start_state, target_state, dls_limit=dls_limit)
        print_sti(start_state, target_state, sti, dls_limit)
        print("State mismatch:", err)

        data = __merge_training_data(start_state, target_state, dls_limit, sti)

        with open(filename,'a') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    for i in range(0, n_step_outs_h):
        print_header("Step out to horizontal in N/E sector")
        dls_limit = random()*0.003 + 0.0015
        start_state = [0, 0, 0, 0, 0]

        north = 1000+random()*2000
        east = 1000+random()*2000
        tvd = 2000+random()*2000

        azi = np.arctan(east/north)
        target_state = [north, east, tvd, np.pi/2, azi]

        sti, err = faststi(start_state, target_state, dls_limit=dls_limit)
        print_sti(start_state, target_state, sti, dls_limit)
        print("State mismatch:", err)

        data = __merge_training_data(start_state, target_state, dls_limit, sti)

        with open(filename,'a') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    for i in range(0, n_below_slot):
        print_header("Horizontal below KO")
        dls_limit = random()*0.003 + 0.0015
        start_state = [0, 0, 0, 0, 0]
        target_state = [0, 0, 2000+random()*2000, np.pi/2, random()*2*np.pi]

        sti, err = faststi(start_state, target_state, dls_limit=dls_limit)
        print_sti(start_state, target_state, sti, dls_limit)
        print("State mismatch:", err)

        data = __merge_training_data(start_state, target_state, dls_limit, sti)

        with open(filename,'a') as file:
            writer = csv.writer(file)
            writer.writerow(data)


def __merge_training_data(start_state, target_state, dls_limit, sti):
    data = []
    data.extend(start_state)
    data.extend(target_state)
    data.append(dls_limit)
    data.extend(sti)

    return data


def __get_header():
    header = ["start_north",
              "start_east",
              "start_tvd",
              "start_inc",
              "start_azi",
              "target_north",
              "target_east",
              "target_tvd",
              "target_inc",
              "target_azi",
              "dls_limit",
              "inc1",
              "inc2",
              "inc3",
              "azi1",
              "azi2",
              "azi3",
              "md_inc1",
              "md_inc2",
              "md_inc3",
    ]

    return header


def __merge_info(start_state, target_state, dls_limit, sti):
    merged = np.array()
    merged.append(start_state)
    merged.append(target_state)
    merged.append(dls_limit)
    merged.append(sti)

    merged.flatten()

    return merged


def __inital_guess(state_from, state_to, pickled_lm_file='linear-mod.sav'):
    """ Produce an inital guess. Will try to use a linear model if available."""

    # TODO Should keep the model in memory between runs. Refactor me.
    if pickled_lm_file is not None:
        # Try first using a linear model from previous runs
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


def print_state(state):
    print("North: ", "{:.2f}".format(state[0]))
    print("East : ", "{:.2f}".format(state[1]))
    print("TVD  : ", "{:.2f}".format(state[2]))
    print("Inc. : ", "{:.4f}".format(state[3]))
    print("Azi. : ", "{:.4f}".format(state[4]))


def print_sti(start_state, target_state, sti, dls_limit):
    print("Start state: ")
    print("----------------")
    print_state(start_state)

    print("\nTarget state:")
    print("----------------")
    print_state(target_state)

    projected_state, dls = __project(start_state, sti)
    print("\nProjected state:")
    print("----------------")
    print_state(projected_state)

    tot_md = sti[6] + sti[7] + sti[8]
    print("\nMD start-target: ", "{:.2f}".format(tot_md))
    print("DLS limit: ", "{:.5f}".format(dls_limit))

    print("\nLegs:")
    print("----------------")
    print("Leg 1, inc: ", "{:.4f}".format(sti[0]), " azi: ", "{:.4f}".format(sti[3]), " md_inc: ", "{:.2f}".format(sti[6]), "dls:", "{:.5f}".format(dls[0]))
    print("Leg 2, inc: ", "{:.4f}".format(sti[1]), " azi: ", "{:.4f}".format(sti[4]), " md_inc: ", "{:.2f}".format(sti[7]), "dls:", "{:.5f}".format(dls[1]))
    print("Leg 3, inc: ", "{:.4f}".format(sti[2]), " azi: ", "{:.4f}".format(sti[5]), " md_inc: ", "{:.2f}".format(sti[8]), "dls:", "{:.5f}".format(dls[2]))
    print("--------------------------------------------------------------\n")


def __truncate_to_bounds(sti, lb, ub, eps=0):
     # Truncate small negative violations from scipy optimize
     # eps is a workaround for https://github.com/scipy/scipy/issues/11403
     tsti = np.maximum(sti, lb + eps)
     tsti = np.minimum(tsti, ub - eps)
     return tsti
    

def __project(start_state, sti):
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
    dnorth[0], deast[0], dtvd[0], dls[0] = __min_curve_segment(start_state[0], start_state[3], sti[0], sti[3], sti[6])
    dnorth[1], deast[1], dtvd[1], dls[1] = __min_curve_segment(sti[0], sti[3], sti[1], sti[4], sti[7])
    dnorth[2], deast[2], dtvd[2], dls[2] = __min_curve_segment(sti[1], sti[4], sti[2], sti[5], sti[8])

    p_north = start_state[0] + sum(dnorth)
    p_east = start_state[1] + sum(deast)
    p_tvd = start_state[2] + sum(dtvd)
    p_inc = sti[2]
    p_azi = sti[5]

    return (p_north, p_east, p_tvd, p_inc, p_azi), dls


def __err_squared_pos_mismatch(state1, state2, scale_md):
    """ Error in bit position with consideration for orientation. """
    d2north = (state1[0] - state2[0])**2 
    d2east = (state1[1] - state2[1])**2
    d2tvd = (state1[2] - state2[2])**2

    terr = (d2north + d2east + d2tvd) / (scale_md**2)

    return terr


def __err_squared_orient_mismatch(state1, state2, dls_limit):
    """ Error in bit orientation. """
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
    """ Calculate a scaled L2 mismatch"""
    perr = __err_squared_pos_mismatch(state1, state2, scale_md)
    oerr = __err_squared_orient_mismatch(state1, state2, dls_limit)

    return perr + oerr


def __err_dls_mse(start_state, sti, dls_limit, scale_md):
    """ Return the sum of squares of dls above the dls_limit scaled by dls_limit"""
    projected_state, dls = __project(start_state, sti)
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

        projected_state, dls = __project(start_state, sti)

        sq_pos_err = __err_squared_pos_mismatch(projected_state, target_state, scale_md)
        sq_ori_err = __err_squared_orient_mismatch(projected_state, target_state, dls_limit)
        sq_dls_err = __err_dls_mse(start_state, sti, dls_limit, scale_md)
        sq_tot_md_err = __err_tot_md_sq(sti, scale_md)

        terr = WEIGHT_POS*sq_pos_err + WEIGHT_ORI*sq_ori_err + WEIGHT_DLS*sq_dls_err + WEIGHT_MD*sq_tot_md_err

        return terr

    return of_precondition

def __get_optifuns_feasibility(start_state, target_state, dls_limit, scale_md, md_weight):

    def of_feasibility(sti):
        projected_state, dls = __project(start_state, sti)

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
        projected_state, dls = __project(start_state, sti)
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


def __min_curve_segment(inc_upper, azi_upper, inc_lower, azi_lower, md_inc):
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


if __name__ == '__main__':
    start_time = datetime.now()
    create_training_data(50, 100, 100, 100, 100)
    end_time = datetime.now()
    delta = end_time - start_time
    print("Elapsed walltime:")
    print(delta)
