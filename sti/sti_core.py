import numpy as np
from scipy.optimize import minimize, Bounds, dual_annealing#, NonlinearConstraint
from sti.dogleg_tf import dogleg_toolface, get_params_from_state_and_net

# QC 
from random import random

"""
"""

def find_sti(start_state, target_state, dls_limit, scale_md):
    """ Fit a sti from start_state to target_state """

    VERBOSE = True
    METHOD = 'L-BFGS-B'
    DLS_EPS = 1e-2 # Proceed to global optimiziation if overriding dls limit this much
    INC_EPS = 1e-1 # Prooced to global opt. if azi error is larger (0.1 ~ 5 degrees)
    AZI_EPS = 1e-1 # Prooced to global opt. if inc error is larger (0.1 ~ 5 degrees)

    acceptable_solution = False

    x0 = initial_guess(start_state, target_state, dls_limit)
    lb, ub = get_bounds(start_state, target_state)
    bounds = Bounds(lb, ub)

    objective_function = get_objective_function(start_state, target_state,dls_limit, scale_md)

    if VERBOSE:
        print("Performing gradient based optimization.")

    result = minimize(objective_function, x0, bounds=bounds, method=METHOD)#, options={'iprint': 99})
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

    def objective_function(sti):
        WEIGHT_DLS = 1000
        WEIGHT_INC = 10000
        WEIGHT_AZI = 10000
        md, dls_mis, inc_err, azi_err = get_error_estimates(start_state, target_state, dls_limit, scale_md, sti)
        of_val = (md/scale_md) ** 2 + (WEIGHT_DLS * dls_mis / dls_limit) ** 2 + WEIGHT_INC * inc_err + WEIGHT_AZI * azi_err
        return of_val

    return objective_function


def get_error_estimates(start_state, target_state, dls_limit, scale_md, sti):

    projected_state, dls, md = project_sti(start_state, sti)

    inc_err = (target_state[3] - projected_state[3])**2
    azi_err = (target_state[4] - projected_state[4])**2

    dls_mis = max(0, dls - dls_limit)

    return md, dls_mis, inc_err, azi_err



def project_sti(start_state, sti):

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
    intermed_state1[0] = intermed_state1[0]  + intermed_state0[0]
    intermed_state1[1] = intermed_state1[1]  + intermed_state0[1]
    intermed_state1[2] = intermed_state1[2]  + intermed_state0[2]

    # Find paramters between intermediate states
    final_net = target_state[0:3]
    tf2, dls2, md2 = get_params_from_state_and_net(intermed_state1, final_net)

    # Max dls and total md
    max_dls = max(dls0, dls1, dls2)
    md = md0 + md1 + md2

    # Projected state
    projected_state = dogleg_toolface(intermed_state1[3], intermed_state1[4], tf2, dls2, md2)

    # Dogleg toolface returns increments
    projected_state[0] = projected_state[0]  + intermed_state1[0]
    projected_state[1] = projected_state[1]  + intermed_state1[1]
    projected_state[2] = projected_state[2]  + intermed_state1[2]

    return projected_state, max_dls, md 


def get_bounds(start_state, target_state):
    # HACK Just throwing in some temporary stuff. But seems to work fine,
    # and performance is not super senstive (tried with stricter bounds)
    lb = np.array([-1.0e4]*6)
    ub = np.array([1.0e4]*6)

    return lb, ub


def initial_guess(start_state, target_state, dls_limit):
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

if __name__ == '__main__':

    for i in range(0, 100):
        start_state = np.array([0., 0., 0., 0., 0.])

        north = (-1 + 2*random()) * 3000
        east = (-1 + 2*random()) * 3000
        tvd = (-1 + 2*random()) * 3000

        inc = np.pi * random()
        azi = 2*np.pi * random()

        # north = 0.0
        # east = 0.0
        # tvd = 100.0

        # inc = np.pi / 2
        # azi =0.0

        dls = 0.002
        scale_md = 3000

        target_state = np.array([north, east, tvd, inc, azi])

        sti, acceptable = find_sti(start_state, target_state, dls, scale_md)

        projected_state, dls, md = project_sti(start_state, sti)


        print("Target: ", target_state)
        print("Projected: ", projected_state)
        print("MD: ", md)
        print("DLS: ", dls)