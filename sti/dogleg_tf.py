import numpy as np
from scipy.integrate import solve_ivp

from random import random


def dogleg_toolface(inc0, azi0, toolface, dls, md):
    """
    Calculate a step with the dogleg toolface method using
    linear algebra

    Parameters:
    inc0: bit inclination at start of step
    azi0: bit azimuth at start of step 
    dls: dog leg severity limit for the step
    toolface: gravity toolface, negative values are straight ahead
    md: step to take
    dense_out: supply dense out for diagnostics and plotting

    Returns:
    state: array of (north, east, tvd, inc_lower, azi_lower)


    Implementation

    Use that dogleg toolface creates a circular wellpath with radius
    R = 1/dls in a plane defined by the bit direction, toolface vector
    and their cross prodcuct.

    Note: This method does not flip the gravity toolface, see discusion
    in notes under dogleg_toolface_ode method.

    We define a co-ordinate transform that shifts and rotates the
    co-ordinate system so that the north, east plane is the circle
    plane in the physical space, and that the bit is at position
    (R,0,0), with bit direction (0,1,0).

    The total arc angle is then theta = md/dls.

    The end bit position and increments in this system is then:

    pos = R * (cos(theta), sin(theta), 0) 
    ori = (-sin(theta), cos(theta), 0)

    And we obtain this in the physical co-ordinate system by inverting
    the transform matrix
    """
    assert(md >= 0.0)

    (b_n_0, b_e_0, b_t_0) = spherical_to_net(inc0, azi0)
    bit = (b_n_0, b_e_0, b_t_0) 

    pos_p, bit_p = dogleg_toolface_inner(b_n_0, b_e_0, b_t_0, toolface, dls, md)
    inc_lower, azi_lower = net_to_spherical(bit_p[0], bit_p[1], bit_p[2])

    state = [
        pos_p[0],
        pos_p[1],
        pos_p[2],
        inc_lower,
        azi_lower
    ]

    state = np.array(state)

    return state

def dogleg_toolface_inner(b_n_0, b_e_0, b_t_0, toolface, dls, md):
    """
    Inner magic of the linear algebra approach.

    Returns position and direction in cartesian co-ordinates as
    two np arrays
    """
    bit = np.array([b_n_0, b_e_0, b_t_0])

    if toolface < 0.0 or dls == 0.0:
        # Straight ahead
        pos = md * bit
        return pos, bit
    else:
        # Do a turn

        R = 1/dls
        theta = md*dls

        # Co-ordinate transform vectors
        tfv = tfv_from_cart_direction(b_n_0, b_e_0, b_t_0, toolface)
        ax_north = -tfv
        ax_east = bit
        ax_tvd = np.cross(ax_north, ax_east) 

        translation = R*tfv

        # Ensure unit vectors, should be OK from calling scope
        # assert( np.dot(ax_north, ax_north) == 1.0)
        # assert( np.dot(ax_east, ax_east) == 1.0)
        # assert( np.dot(ax_tvd, ax_tvd) == 1.0)

        # Build transform matrix
        A = np.array([ax_north, ax_east, ax_tvd])

        # Positions in transformed space
        pos_ax = R * np.array([np.cos(theta), np.sin(theta), 0.0])
        bit_ax = np.array([-np.sin(theta), np.cos(theta), 0.0])

        # Transform back to physical space
        pos_p = np.linalg.solve(A, pos_ax) + translation
        bit_p = np.linalg.solve(A, bit_ax)

    return pos_p, bit_p

def dogleg_toolface_ode(inc0, azi0, toolface, dls, md, dense_output=False):
    """
    Calculate position increments from a step of the dogleg
    tool face method using an ODE solver.
    
    This function is more of academic interest, you're probaly looking
    for dogleg_toolface()

    Parameters:
    inc0: bit inclination at start of step
    azi0: bit azimuth at start of step 
    dls: dog leg severity limit for the step
    toolface: gravity toolface, negative values are straight ahead
    md: step to take
    dense_out: supply dense out for diagnostics and plotting

    Returns:
    state: array of (north, east, tvd, inc_lower, azi_lower)
    sol: OdeSolution
    z: Dense sampling of solution, only if dense_output is true

    Implementation

    Need to do some numerical wizardry here. And we'd like 
    to stay in Cartesian co-ordinates when doing that.

    Also, probably want to use the instant tf vector, say tfv.

    Let _s note the derivative of a function wrt. to arc length, and
    (n, e, t) be north, east and tvd (right handed system).

    Then, given the direction of the bit and tool face angle, we can
    calculate a tool face vector (n, e, t) for any position of the bit.

    Note that the tool face vector changes along the trajetory,
    while the tool face angle is constant.

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
    reference of toolface, e.g. when the horizontal component of
    bit direction passes thorugh 0.

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

    We solve wthis problem by augmenting the ODE system with a local up
    vector l. We then calculate the tfv using this as reference vector.

    For this local up vector, we would like the following properties to
    be preserved by the dynamics:
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

    This can easily be solved for l_s for each step in an ODE solver.

    Our full state is then:

    y = [n, e, t, n_s, e_s, t_s, l_n, l_e, l_t]

    And the dynamics are given by (1), (2) and (3)

    [1] https://en.wikipedia.org/wiki/Curvature#Space_curves
    """

    (b_n_0, b_e_0, b_t_0) = spherical_to_net(inc0, azi0)
    bit = (b_n_0, b_e_0, b_t_0) 
    sol, z = dogleg_toolface_ode_inner(b_n_0, b_e_0, b_t_0, toolface, dls, md, dense_output)

    y_end = sol.y[:,-1]

    # Store the end position, inc and azi
    state = np.array([0.0]*5)
    state[0] = y_end[0]
    state[1] = y_end[1]
    state[2] = y_end[2]

    inc, azi = net_to_spherical(y_end[3], y_end[4], y_end[5])

    state[3] = inc
    state[4] = azi

    return state, sol, z


def dogleg_toolface_ode_inner(b_n_0, b_e_0, b_t_0, toolface, dls, md, dense_output=False):
    """
    Inner method, using Carestian co-ordinates.

    Take a step of dogleg toolface method, initial position at (0, 0, 0).

    Parameters:
    -----------
    b_n_0: Initial bit north direction
    b_e_0: Initial bit east direction
    b_t_0: Initial bit tvd direction
    dls: Dogleg severity
    toolface: Toolface angle
    md: Measured depth (arc length)
    dense_output: Provide sampling every 1th unit for plotting etc.

    """

    bit_norm = (b_n_0**2 + b_e_0**2 + b_t_0**2) ** (0.5)

    b_n_0 = b_n_0 / bit_norm 
    b_e_0 = b_e_0 / bit_norm 
    b_t_0 = b_t_0 / bit_norm 

    # Default to gravity reference
    l_n_0 = 0.0
    l_e_0 = 0.0
    l_t_0 = -1.0

    # Ortogonolize wrt. to bit
    bit0 = np.array([b_n_0, b_e_0, b_t_0])
    l0 = np.array([l_n_0, l_e_0, l_t_0])

    l0 = l0 -proj(bit0, l0)

    l0_norm = (l0[0]**2 + l0[1]**2 + l0[2]**2) ** (0.5)
    l0 = l0 / l0_norm

    l_n_0 = l0[0]
    l_e_0 = l0[1]
    l_t_0 = l0[2]

    # But check it we're pointing straight up or down, then we swtich to north
    if abs(b_e_0) + abs(b_n_0) < 1e-6:
        l_n_0 = 1.0
        l_e_0 = 0.0
        l_t_0 = 0.0

    y_0 = [
        0.0 ,
        0.0 ,
        0.0 ,
        b_n_0 ,
        b_e_0 ,
        b_t_0 ,
        l_n_0 ,
        l_e_0 ,
        l_t_0 ,
        ]

    # Stiff problem, cannot use default RK45
    sol = solve_ivp(ode_rhs,[0, md], y_0, args=(dls, toolface), dense_output=dense_output, method='Radau')

    z = None

    if dense_output:
        md_dense = np.linspace(0, md, num=int(np.ceil(md)))
        z = sol.sol(md_dense)
        z = z.T

    return sol, z


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

    l_s = np.linalg.solve(A, rhs)

    y_s[6] = l_s[0]
    y_s[7] = l_s[1]
    y_s[8] = l_s[2]

    return y_s


def proj(u, v):
    """
    Project v on u
    """
    t = np.dot(v,u)
    n = np.dot(u,u)

    return t/n * u

    
def tfv_from_cart_direction_with_given_up(bit_n, bit_e, bit_t, toolface, up_n, up_e, up_t):
    """
    Calculate the toolface vector from bit direction and a given upwards direction

    Implementation:

    Create a local system based on bit orientation and provided upwards direction, 
    calculate tfv using magnetic tfv and transform back.

    Let the bit direction be positive t, then we project upward direction
    on the bit vector to obtain the ortogonal north in a magnetic system.
    At last, we take the cross product of these to obtain a rhs system.

    In this system, toolface vector is trivially the magnetic toolface vector, which
    we easily can compute. Thereafter, we transform back to to the provided (n,e,t)
    system, normalize and return

    """
    if toolface < 0.0:
        #Straight ahead
        tfv = np.array((bit_n, bit_e, bit_t))
    else:
        t = np.array((bit_n, bit_e, bit_t))
        n = np.array((up_n, up_e, up_t))
        n = n - proj(t,n)
        e = np.cross(t,n)

        mag_n = np.cos(toolface)
        mag_e = np.sin(toolface)
        mag_t = 0.0

        tfv_mag = np.array((mag_n, mag_e, mag_t))

        A = np.array((n, e, t))
        # B = np.linalg.inv(A)
        # tfv = np.dot(B,tfv_mag)
        tfv = np.linalg.solve(A, tfv_mag)
    
    # Return a normalized vector
    norm = (np.dot(tfv, tfv))**(0.5)
    assert(norm > 0)

    tfv = tfv / norm

    return tfv


def tfv_from_cart_direction(bit_n, bit_e, bit_t, toolface):
    """
    Calculate the toolface vector from (n, e, t) bit direction
    and toolface angle.

    """
    t_n = 0.0
    t_e = 0.0
    t_t = 0.0

    if toolface < 0.0:
        #Straight ahead
        tfv = np.array((bit_n, bit_e, bit_t))
        return tfv
    elif abs(bit_n) + abs(bit_e) < 1e-6:
        # Magnetic toolface, normalized by definition
        t_n = np.cos(toolface) 
        t_e = np.sin(toolface)
        t_t = 0.0
        return np.array((t_n, t_e, t_t))
    else:
        # Gravity toolface
        tfv = tfv_from_cart_direction_with_given_up(bit_n, bit_e, bit_t, toolface, 0, 0, -1)

        return tfv

def toolface_from_tfv_and_bit(tfv_n, tfv_e, tfv_t, inc, azi):
    """
    Calculate toolface angle from toolface vector and bit direction.
    """
    bit = spherical_to_net(inc, azi)
    tfv = np.array([tfv_n, tfv_e, tfv_t])

    print("yo!")
    # Expecting tfv orto to bit and unit length
    assert abs(np.dot(bit, tfv)) < 1e-4
    assert abs(np.dot(tfv, tfv) - 1.0) < 1e-4

    if inc==0 or inc==np.pi:
        # Magnetic toolface
        ref_north = np.array([1, 0, 0])
        ref_east = np.array([0, 1, 0])
    else:
        # Gravity toolface
        ref_north = np.array([0, 0, -1])
        ref_east = np.dot(bit, ref_north)
        ref_east = ref_east - proj(ref_north, ref_east)
        ref_east_norm = np.dot(ref_east, ref_east) ** (0.5)
        ref_east = ref_east / ref_east_norm

    cos_tf = np.dot(ref_north, tfv)
    sin_tf = np.dot(ref_east, tfv)

    assert abs(cos_tf) + abs(sin_tf) > 0.0

    toolface = np.arctan2(sin_tf, cos_tf)

    return toolface


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
    Transform orientation of a (north, east, tvd) direction
    to (inc, azi)
    """
    r = (north**2 + east**2 + tvd**2) ** (0.5)
    inc = np.arccos(tvd/r)
    azi = np.arctan2(east, north)
    if azi < 0:
        azi = azi + 2*np.pi

    return np.array((inc, azi))


def get_params_from_states(from_state, to_state):
    """
    Given a from state and to state as (n, e, t, inc, azi),
    find the toolface parameters (tf, dls, md) that connects them
    or (None, None, np.inf) if it is not possible to connect
    the states using the toolface dogleg method.

    For the special case of a straight line, (-1, 0, md) is returned.
    """
    
    # Precision criteria for floating comparisions
    EPS = 1e-6

    pos0 = np.array([from_state[0], from_state[1], from_state[2]])
    pos1 = np.array([to_state[0], to_state[1], to_state[2]])

    inc0 = from_state[3]
    azi0 = from_state[4]

    inc1 = to_state[3]
    azi1 = to_state[4]

    bit0 = spherical_to_net(inc0, azi0)
    bit1 = spherical_to_net(inc1, azi1)

    cos_theta = np.dot(bit0, bit1)
    theta = np.arccos(cos_theta)

    pos_diff = pos1 - pos0
    pos_diff_norm = (np.dot(pos_diff, pos_diff))** (0.5)

    bit_diff = bit1 - bit0
    bit_diff_norm = (np.dot(bit_diff, bit_diff)) ** (0.5)

    state_diff_norm = pos_diff_norm + bit_diff_norm


    # Initialize tf to None
    tf = None

    # Catch special case with no change.
    # Full circle or no change. We return no change.
    if state_diff_norm == 0.0:
        tf = -1.0
        dls = 0.0
        md = 0.0
        return np.array([tf, dls, md])

    # Catch special case with different orientation, but same position
    # Not possible
    if pos_diff_norm < EPS and bit_diff_norm > EPS:
            tf = None
            dls = None
            md = np.inf
            return np.array([tf, dls, md])

    # Catch special cases with equal bit direction
    if abs(cos_theta -1.0) < EPS:
        # Possible cases:
        # 1. From and to are equal. Handled above.
        # 2. Straight line between states.
        # 3. Step out to vertical type well. Not on a circle.

        if abs(np.dot(pos_diff, bit0) / pos_diff_norm -1) < EPS:
            # On a straight line.
            tf = -1.0
            dls = 0.0
            md = pos_diff_norm
            return np.array([tf, dls, md])
        else:
            # Step out to vertical type situation, not possible on a circle
            tf = None
            dls = None
            md = np.inf
            return np.array([tf, dls, md])

    
    # Catch special cases with opposite bit directions
    if abs(cos_theta + 1.0) < EPS:
        # Possible cases:
        # 1. Straight line, opposite orientation. Not on a circle
        # 2. J-bend well. Not on a circle.
        # 3. Half-way around a circle.
        # Only case 3 is premissible, and this is characterized by
        # the position difference being ortogonal on the bit directions.

        if abs(np.dot(pos_diff, bit0)) > EPS:
                tf = None
                dls = None
                md = np.inf
                return np.array([tf, dls, md])
    
    # Catch special cases with ortogonal bit directions
    if abs(cos_theta) < EPS:
        # Possible cases:
        # 1. Around a circle, 1/4th or 3/4th of the way.
        # 2. Bottom position of a J-type well. Not on a circle.
        # Only case 1 is on a circle. Characterized by that the
        # angle between the position difference and any of the 
        # bit directions is pi/2 ... and?

        pos_diff_unit = pos_diff / pos_diff_norm

        if abs(abs(np.dot(pos_diff_unit, bit0)) - np.cos(np.pi/4)) > EPS:
                tf = None
                dls = None
                md = np.inf
                return np.array([tf, dls, md])
        

    # From here we assume all the special cases have been caugth
    # Calculate tool face
    sin_tf = 0.0
    cos_tf = 0.0

    partial_1 = np.cos(inc0)*cos_theta - np.cos(inc1)
    partial_2 = np.sin(inc0)*np.sin(theta)

    if partial_2 != 0.0:
        cos_tf = partial_1 / partial_2

    if np.sin(theta) != 0.0:
        sin_tf = np.sin(inc1)*np.sin(azi1-azi0) / np.sin(theta)

    if abs(sin_tf) + abs(cos_tf) > EPS:
        # Can use arctan to determine tf in [0, 2*pi]
        tf = np.arctan2(sin_tf, cos_tf)
        if tf < 0.0:
            tf = tf + 2*np.pi
    else:
    # We've drilled in a half circle. The toolface vector
    # in the start point is directly proportional to the
    # difference between the start and end point
        tfv = pos_diff / pos_diff_norm
        tfv = tfv - proj(bit0, tfv)
        tfv_norm = np.dot(tfv, tfv) ** (0.5)
        assert tfv_norm > 0.0
        tfv = tfv / tfv_norm

        tf = toolface_from_tfv_and_bit(tfv[0] ,tfv[1], tfv[2], inc0, azi0)
        
    
    
    # Calculate radius and dls using chord length. 
    # Assuming that special cases have been handled above.
    assert pos_diff_norm > 0.0
    assert np.sin(theta/2) != 0.0

    r = pos_diff_norm / (2 * np.sin(theta/2))
    dls = 1/r

    # For md step, we have two possibilities:
    # 1. r * theta
    # 2. r * (2*pi - theta) 
    # This is so since theta is the shortest arc between the
    # points, but we need to take orientation into account
    #
    # Also, the trigonometric wizardry above breaks down
    # and given wrong tf when theta is above pi. Need to correct
    # that as well

    md1 = r * theta
    md2 = r * (2*np.pi - theta)

    tf2 = tf + np.pi

    state_r1 = dogleg_toolface(inc0, azi0, tf, dls, md1)
    state_r2 = dogleg_toolface(inc0, azi0, tf, dls, md2)
    state_r3 = dogleg_toolface(inc0, azi0, tf2, dls, md1)
    state_r4 = dogleg_toolface(inc0, azi0, tf2, dls, md2)

    pos_r1 = np.array([state_r1[0], state_r1[1], state_r1[2]])
    pos_r2 = np.array([state_r2[0], state_r2[1], state_r2[2]])
    pos_r3 = np.array([state_r3[0], state_r3[1], state_r3[2]])
    pos_r4 = np.array([state_r4[0], state_r4[1], state_r4[2]])

    diff_1 = sum(abs(pos_r1 - pos_diff))
    diff_2 = sum(abs(pos_r2 - pos_diff))
    diff_3 = sum(abs(pos_r3 - pos_diff))
    diff_4 = sum(abs(pos_r4 - pos_diff))

    md = 0.0
    if diff_1 < 1e-3:
        md = md1
    elif diff_2 < 1e-3:
        md = md2
    elif diff_3 < 1e-3:
        md = md1
        tf = tf2
    elif diff_4 < 1e-3:
        md = md2
        tf = tf2
    else:
        raise AssertionError

    if tf > 2*np.pi:
        tf = tf - 2*np.pi

    return np.array([tf, dls, md])


if __name__ == '__main__':
    pass