import numpy as np
import pandas as pd
from scipy.stats import ortho_group, special_ortho_group
from random import random
import sti.sti_core as sti_core
import sti.sti_utils as sti_utils

def spherical_to_net(inc, azi):
    """ 
    Transform spherical orientation to unit vector of
    (north, east, tvd)
    """
    tvd = np.cos(inc)
    north = np.sin(inc)*np.cos(azi)
    east = np.sin(inc)*np.sin(azi)

    norm = (tvd**2 + north**2 + east**2) ** (0.5)

    north = north / norm
    east = east / norm
    tvd = tvd / norm

    return (north, east, tvd)

def net_to_spherical(north, east, tvd):
    """
    Transform orientation of a (north, east, tvd) tuple
    to (inc, azi)
    """
    r = (north**2 + east**2 + tvd**2) ** (0.5)
    inc = np.arccos(tvd/r)
    azi = np.arctan2(east, north)
    if azi < 0:
        azi = azi + 2*np.pi

    return (inc, azi)


def transform_state(A, state):
    """
    A: Ortho matrix
    state: (north, east, tvd, inc, azi) tuplet
    """

    pos = np.array((state[0], state[1], state[2]))
    ori = spherical_to_net(state[3], state[4])

    pos = np.dot(A,pos)
    ori = np.dot(A,ori)

    oris = net_to_spherical(ori[0], ori[1], ori[2])

    state = np.append(pos, oris).flatten()

    return state

def transform_sti(A, sti):
    inc1, azi1 = sti[0], sti[3]
    inc2, azi2 = sti[1], sti[4]
    inc3, azi3 = sti[2], sti[5]

    ori1 = spherical_to_net(inc1, azi1)
    ori2 = spherical_to_net(inc2, azi2)
    ori3 = spherical_to_net(inc3, azi3)

    ori1 = np.dot(A, ori1)
    ori2 = np.dot(A, ori2)
    ori3 = np.dot(A, ori3)
    
    (inc1, azi1) = net_to_spherical(ori1[0], ori1[1], ori1[2])
    (inc2, azi2) = net_to_spherical(ori2[0], ori2[1], ori2[2])
    (inc3, azi3) = net_to_spherical(ori3[0], ori3[1], ori3[2])

    sti_rot = np.array([0.]*9)

    sti_rot[0] = inc1
    sti_rot[1] = inc2
    sti_rot[2] = inc3
    sti_rot[3] = azi1
    sti_rot[4] = azi2
    sti_rot[5] = azi3
    sti_rot[6] = sti[6]
    sti_rot[7] = sti[7]
    sti_rot[8] = sti[8]
    
    return sti_rot

def test_transform_state(n=100):

    for i in range(0,n):
        north = -1 + 2*random()
        east = -1 + 2*random()
        tvd = -1 + 2*random()

        r = (north**2 + east **2 + tvd**2) ** (0.5)

        north = north/r
        east = east /r
        tvd = tvd /r

        A  = ortho_group.rvs(3)
        state = np.array((random(), random(), random(), np.pi/2*random(), np.pi*random()))

        state = transform_state(A, state)

        B = np.linalg.inv(A)

        state = transform_state(B, state)

        norm = sum((state - state)**2)

        if norm > 1e-4:
            print("error!")

def test_transform_sti(n=100):

    for i in range(0,n):
        sti = np.array([0.]*9)
        sti[0] = np.pi*random()
        sti[1] = np.pi*random()
        sti[2] = np.pi*random()
        sti[3] = 2*np.pi*random()
        sti[4] = 2*np.pi*random()
        sti[5] = 2*np.pi*random()
        sti[6] = 2000*random()
        sti[7] = 2000*random()
        sti[8] = 2000*random()

        A  = ortho_group.rvs(3)

        sti = transform_sti(A, sti)

        B = np.linalg.inv(A)

        sti = transform_state(B, sti)

        norm = sum((sti - sti)**2)

        if norm > 1e-4:
            print("error!")

def test_transform(n=100):

    for i in range(0,n):
        north = -1 + 2*random()
        east = -1 + 2*random()
        tvd = -1 + 2*random()

        r = (north**2 + east **2 + tvd**2) ** (0.5)

        north = north/r
        east = east /r
        tvd = tvd /r

        (inc, azi) = net_to_spherical(north, east, tvd)
        (north_n, east_n, tvd_n) = spherical_to_net(inc, azi)

        norm = (north - north_n)**2 + (east - east_n)**2 + (tvd - tvd_n)**2

        if norm > 1e-4:
            print(norm)

def test_bootstrap_rot1(n=10, filename_data='data.csv'):

    # Read dataset
    df = pd.read_csv(filename_data)

    # Find rows with bad data
    for i in range(0, df.shape[0], n):
        states = np.array(df.iloc[i,0:11].values)

        from_state = states[0:5]
        to_state = states[5:10]
        dls_limit = states[10]
        sti = np.array(df.iloc[i,11:])

        # Error before transform
        projection, dls = sti_core.project_sti(from_state, sti)
        err_before = sti_core.__err_squared_state_mismatch(to_state, projection, dls_limit, scale_md=100)
        # sti_utils.print_sti(from_state, to_state, sti, dls_limit)

        # Draw a random transform
        A  = ortho_group.rvs(3)
        # A = special_ortho_group.rvs(3)

        # Transform
        from_state_t = transform_state(A, from_state)
        to_state_t = transform_state(A, to_state)
        sti_t = transform_sti(A, sti)

        # Validate
        projection_t, dls = sti_core.project_sti(from_state_t, sti_t)
        err_after = sti_core.__err_squared_state_mismatch(to_state_t, projection_t, dls_limit, scale_md=100)

        # sti_utils.print_sti(from_state, to_state, sti, dls_limit)

        norm = (err_before - err_after) ** 2

        if norm > 1e-3:
            print(norm) 
            print("\n\n\n\nBefore")
            print("######################")
            sti_utils.print_sti(from_state, to_state, sti, dls_limit)
            print("Err before:", err_before)
            print("After")
            print("######################")
            sti_utils.print_sti(from_state_t, to_state_t, sti_t, dls_limit)
            print("Err after:", err_after)





if __name__ == '__main__':

    test_transform(1000)
    test_transform_state(100)
    test_transform_sti(100)
    test_bootstrap_rot1()



