import numpy as np
from scipy.integrate import solve_ivp


def dogleg_toolface(inc0, azi0, dls, toolface, md_inc):
    """ Calculate position increments from a step of the dogleg
    tool face method.

    Parameters:
    inc0: bit inclination at start of step
    azi0: bit azimuth at start of step 
    dls: dog leg severity limit for the step
    toolface: gravity toolface, negative values are straight ahead
    md: step to take

    Returns:
    (north, east, tvd, inc_lower, azi_lower) - position as increments
    """

    """
    Note to self:

    Approach:

    Need to do some numerical wizardry here. Probably prefer
    to stay in Cartesian co-ordinates when doing that.

    Also, probably want to use the instant tf vector, say tfv.

    Let _s note the derivative of a function wrt. to arc length, and
    (n, e, t) be north, east and tvd (right handed system).

    Then, given the direction of the bit and tool face angle, we can
    calculate a tool face vector (n, e, t) for any position of the bit.

    Note that the tool face vector changes along the trajetory,
    while the tool face angle is constant.

    BONUS: Tool face vector is continouos under rotations.

    tfv ( n_s, e_s, t_s, tf) -> (n, e, t)

    So we need a function to calculate this guy, which essentially
    is the unit normal vector of the wellbore, see [1]. Further,
    the doleg leg severity, which is given, is the curvature as
    defined in [1].

    Thus, we have a ODE that defines the curve.

    d/ds( n   ) = n_s unit comp.
    d/ds( e   ) = e_s unit comp.
    d/ds( t   ) = t_s unit comp.                         (1)
    d/ds( n_s ) = dls * tfv(n_s, e_s, t_s, tf).north
    d/ds( e_s ) = dls * tfv(n_s, e_s, t_s, tf).east
    d/ds( t_s ) = dls * tfv(n_s, e_s, t_s, tf).tvd

    Initial values are given by inital position and directions.

    A challenge with this system is the behaviour if we pass through
    a situation where we change from gravity -> magnetic -> gravity
    reference of toolface, i.e. when the horizontal component of
    bit direction is 0.

    Consider for example drilling straight north with toolface angle
    initially set at 0. This would set us up to drill a circle upwards.
    As the inclination passes from pi -> 0, the gravity reference flips,
    and unless we deal with it specifically, would set us up to drill in
    another upward circle. Although physically, one would have to
    manipulate the steering parameters to do so.

    While this can be considered a feature, there are also some challenges
    with this behaviour. E.g. it is not preserved under coordinate rotations,
    which will be used in other parts of the code. Therefore - we have to
    change the behaviour to drill in circles as the physical system would
    with a untwistable drill string and fixed tf.

    Proposed idea: Bring along the initial up direction in the ODE system,
    and calculate the toolface vector from this.

    Augment the ODE system with a local up vector l. We then calculate
    the tfv using this as reference vector.

    For this local up vector, we would like:
    * Ortogonality to the bit direction
    * Ortogonality to the east direction as defined by the cross product
      of it self with the bit direction
    * Unit length

    Let b denote bit direction, translating the above to inner and
    dot products, we obtain:

    <l, b>      = 0
    <l, l x b> = 0
    <l, l>     = 1

    By differentiating wrt. to arc length, and using d/ds b = dls * tfv, we
    obtain the following linear equations for l_s, where all other quantities
    can be calculated from the state:

    <l_s, l> = 0
    <l_s, b> = -<l, dls * tfb>                             (3)
    <_ls>, l x b> = -1/2 * <l, l x dls * tfv>

    This can easily be solved for l_s as input to an ODE solver.

    Our full state is then:

    y = [n, e, t, n_s, e_s, t_s, l_n, l_e, l_t]

    And the dynamics are given by (1), (2) and (3)

    [1] https://en.wikipedia.org/wiki/Curvature#Space_curves
    """
    raise NotImplementedError


def proj(u, v):
    """
    Project v on u
    """
    t = np.dot(v,u)
    n = np.dot(u,u)

    return t/n * u

def tfv_from_cart_direction_with_given_up(north, east, tvd, toolface, up_north, up_east, up_tvd):
    """
    Calculate the toolface vector from bit direction and a given upwards direction

    Notes

    Create a local system based on bit orientation, calculate tfv in this,
    and transform back

    Let the bit direction be positive t, then we project upward direction
    on the bit vector to obtain the ortogonal north in a magnetic system.
    At last, we take the cross product of these to obtain a rhs system.

    In this system, toolface vector is trivially the magnetic toolface vector, which
    we easily can compute. Thereafter, we transform back to to the true (n,e,t)
    system, normalize and return

    """
    t = np.array((north, east, tvd))
    n = np.array((up_north, up_east, up_tvd))
    n = n - proj(t,n)
    e = np.cross(t,n)

    mag_n = np.cos(toolface)
    mag_e = np.sin(toolface)
    mag_t = 0.0

    tfv_mag = np.array((mag_n, mag_e, mag_t))

    A = np.array((n, e, t))
    B = np.linalg.inv(A)

    tfv = np.dot(B,tfv_mag)
    norm = (np.dot(tfv, tfv))**(0.5)

    assert(norm > 0)

    tfv = tfv / norm

    return tfv

def tfv_from_cart_direction(north, east, tvd, toolface):
    """
    Calculate the toolface vector from (n, e, t) direction vector.
    """
    t_n = 0.0
    t_e = 0.0
    t_t = 0.0
    if abs(north) + abs(east) < 1e-6:
        # Magnetic toolface, normalized by definition
        t_n = np.cos(toolface) 
        t_e = np.sin(toolface)
        t_t = 0.0
        return np.array((t_n, t_e, t_t))
    else:
        # Gravity toolface
        tfv = tfv_from_cart_direction_with_given_up(north, east, tvd, toolface, 0, 0, -1)

        return tfv

def ode_rhs(s, y, dls, init_toolface_angle):
    """
    Right hand side in ODE of IVP to project position.

    We let
    
    y = [n, e, t, n_s, e_s, t_s, l_n, l_e, l_t]
    #   [0, 1, 2, 3  , 4  , 5  , 6   , 7  , 8 ]

    This function calculates and returns derivative of this
    state wrt. to arc length. See notes in top of code.

    Short notes:
    * s is the arc length, not used, but needed for ODE solver
    * _s is derivate wrt. arc length
    * n, e, t is the bit position
    * l is the local up, and _n, _e, _t subscripts dir comps.

    """
    y_s = np.array([0.0]*9)
    
    # Trivial part
    bit_n = y[3]
    bit_e = y[4]
    bit_t = y[5]

    bit_norm = (bit_n**2 + bit_e**2 + bit_t**2) ** (0.5)

    bit_n = bit_n / bit_norm
    bit_e = bit_e / bit_norm
    bit_t = bit_t / bit_norm

    y_s[0] = bit_n
    y_s[1] = bit_e
    y_s[2] = bit_t

    # We the calculate the tfv wrt. to the local up
    l_n = y[6]
    l_e = y[7]
    l_t = y[8]

    tfv = tfv_from_cart_direction_with_given_up(bit_n, bit_e, bit_t, init_toolface_angle, l_n, l_e, l_t)

    # We can then calculuate the change in bit direction per arc length
    y_s[3] = dls * tfv[0]
    y_s[4] = dls * tfv[1]
    y_s[5] = dls * tfv[2]

    # At last, we need to compute the change in the local up per arc length
    l = np.array([l_n, l_e, l_t])
    bit = np.array([bit_n, bit_e, bit_t])
    l_s = np.array([0.0, 0.0, 0.0])

    # Create matrix as defined in (3) and solve for l_s
    lxb = np.cross(l, bit)
    lxtfv = np.cross(l, tfv)
    A = np.array([bit, l, lxb])
    rhs = np.array([-np.dot(l, dls*tfv), 0.0, -1/2*np.dot(l, dls*lxtfv)])

    # TODO Just solve for l_s without using linalg.inv
    B = np.linalg.inv(A)
    l_s = np.dot(B,rhs)

    # Small sanity check here..
    grav = np.array([0.0, 0.0, -1.0])
    qcc = np.dot(bit, grav)

    print("QC:", qcc)

    y_s[6] = l_s[0]
    y_s[7] = l_s[1]
    y_s[8] = l_s[2]

    return y_s


def tfv_from_spherical_direction(inc, azi, toolface):
    """
    Calculate the toolface vector from (inc, azi) direction vector.
    """
    raise NotImplementedError


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

    return np.array((north, east, tvd))


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

    return np.array((inc, azi))

if __name__ == '__main__':
    # vec = tfv_from_cart_direction(1.0,0.0,0.0, 0/2*np.pi )
    # print(vec)

    # Start northwards drill in a circle 
    dls = 0.002
    md = 3200
    tf_0 = 1*np.pi / 4
    n_0 = 0.0
    e_0 = 0.0
    t_0 = 0.0
    b_n_0 = 1.0
    b_e_0 = 0.0
    b_t_0 = 0.0
    l_n_0 = 0.0
    l_e_0 = 0.0
    l_t_0 = -1.0

    y_0 = [
        n_0   ,
        e_0   ,
        t_0   ,
        b_n_0 ,
        b_e_0 ,
        b_t_0 ,
        l_n_0 ,
        l_e_0 ,
        l_t_0 ,
        ]

    sol = solve_ivp(ode_rhs,[0, md], y_0, args=(dls, tf_0), dense_output=True, method='Radau')

    # print("Full solution")
    # print(sol.y)
    # print("End point")
    # print(sol.y[-1])

    md_dense = np.linspace(0, md, md)
    z = sol.sol(md_dense)
    import matplotlib.pyplot as plt
    z = z.T

    # z = z[:,0:3]
    # plt.plot(md_dense, z)
    # print(z.shape)
    # plt.xlabel('md')
    # plt.legend(['north', 'east', 'tvd'], shadow=True)
    # plt.show()

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = z[:,0].flatten()
    y = z[:,1].flatten()
    z = z[:,2].flatten()

    print(x.shape)

    ax.plot(x, y, z)

    plt.show()