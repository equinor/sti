import numpy as np


def proj(u, v):
    """
    Project v on u
    """
    t = np.dot(v,u)
    n = np.dot(u,u)

    return t/n * u


def l2norm(u):
    """
    Return l2norm of u
    """
    return np.dot(u, u) ** (0.5)


def normalize(u):
    """
    Return unit vector in direction of u
    """
    norm_u = l2norm(u)
    assert norm_u > 0

    return u / norm_u


def orthogonalize(u, v):
    """
    Orthogonalize v wrt. to u
    """
    return v - proj(u, v)


def spherical_to_net(inc, azi):
    """ 
    Transform spherical orientation to unit vector of
    (north, east, tvd)

    Parameters:
    inc: Inclination, measured from positive tvd axis
    azi: Azimuth, measured from north axis

    Returns:
    array-like: (north, east, tvd) unit vector

    """
    tvd = np.cos(inc)
    north = np.sin(inc)*np.cos(azi)
    east = np.sin(inc)*np.sin(azi)

    norm = (tvd**2 + north**2 + east**2) ** (0.5)

    north = north / norm
    east = east / norm
    tvd = tvd / norm

    return np.array((north, east, tvd))


def net_to_spherical(north, east, tvd):
    """
    Transform orientation of a (north, east, tvd) direction
    to (inc, azi)

    Parameters:
    north: North component of orientation
    east: East componenent of orientation
    tvd: TVD component of orientation

    Returns:
    array-like: (inc, azi) orientation in spherical co-ordinates

    """
    r = (north**2 + east**2 + tvd**2) ** (0.5)
    inc = np.arccos(tvd/r)
    azi = np.arctan2(east, north)
    if azi < 0:
        azi = azi + 2*np.pi

    return np.array((inc, azi))    


def pos_from_state(state):
    return state[0:3]


def cart_bit_from_state(state):
    inc = state[3]
    azi = state[4]

    return spherical_to_net(inc, azi)


def translate_state(state, translation):
    """
    Translate a state by a translation vector
    """

    n = state[0] + translation[0]
    e = state[1] + translation[1]
    t = state[2] + translation[2]

    inc = state[3]
    azi = state[4]

    new_state = np.array([n, e, t, inc, azi])

    return new_state