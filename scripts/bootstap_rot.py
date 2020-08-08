import numpy as np
from scipy.stats import ortho_group
from random import random

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

    return (inc, azi)

def test_transform(n):

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


def transform_state(A, state):
    """
    A: Ortho matrix
    state: (north, east, tvd, inc, azi) tuplet
    """

    raise NotImplementedError

def transform_sti(A, sti):
    raise NotImplementedError

if __name__ == '__main__':

    test_transform(1000)

    # A  = ortho_group.rvs(3)
    # print(np.dot(A,x))
    # x = np.array([north , east, tvd])