import pytest
import numpy as np
from sti.dogleg_tf import dogleg_toolface, dogleg_toolface_ode, spherical_to_net, net_to_spherical 
from random import random

class TestDoglegToolface():

    def test_straight_north_tf(self):
        data = dogleg_toolface(np.pi/2, 0.0, -1.0, 0.002, 300)
        assert data[0] == pytest.approx(300)
        assert data[1] == pytest.approx(0.0)
        assert data[2] == pytest.approx(0.0)
        assert data[3] == pytest.approx(np.pi/2)
        assert data[4] == pytest.approx(0.0)

    def test_straight_north_dls(self):
        data = dogleg_toolface(np.pi/2, 0.0, 0.0, 0.0, 300)
        assert data[0] == pytest.approx(300)
        assert data[1] == pytest.approx(0.0)
        assert data[2] == pytest.approx(0.0)
        assert data[3] == pytest.approx(np.pi/2)
        assert data[4] == pytest.approx(0.0)

    def test_straight_down_tf(self):
        data = dogleg_toolface(0.0, 0.0, -1.0, 0.002, 300)
        assert data[0] == pytest.approx(0.0)
        assert data[1] == pytest.approx(0.0)
        assert data[2] == pytest.approx(300)
        assert data[3] == pytest.approx(0.0)
        assert data[4] == pytest.approx(0.0)

    def test_straight_up_tf(self):
        data = dogleg_toolface(np.pi, 0.0, -1.0, 0.002, 300)
        assert data[0] == pytest.approx(0.0)
        assert data[1] == pytest.approx(0.0)
        assert data[2] == pytest.approx(-300.0)
        assert data[3] == pytest.approx(np.pi)
        assert data[4] == pytest.approx(0.0)

    def test_random_straight_lines(self):
        # TODO Make me
        pass

    def test_start_northwards(self):
        # data = (north, east, tvd, inc_lower, azi_lower)

        # Turn to the north east
        data = dogleg_toolface(np.pi/2, 0., 2*np.pi * 1/4, 0.002, 300)
        assert data[0] > 0.0
        assert data[1] > 0.0
        assert data[2] == pytest.approx(0.0)

        # Directly north and down
        data = dogleg_toolface(np.pi/2, 0., 2*np.pi * 2/4, 0.002, 300)
        assert data[0] > 0.0
        assert data[1] == pytest.approx(0.0)
        assert data[2] > 0.0

        # Turn north west
        data = dogleg_toolface(np.pi/2, 0., 2*np.pi * 3/4, 0.002, 300)
        assert data[0] > 0.0
        assert data[1] < 0.0
        assert data[2] == pytest.approx(0.0)

        # Directly north and upward, at 2pi toolface
        data = dogleg_toolface(np.pi/2, 0., 2*np.pi * 4/4, 0.002, 300)
        assert data[0] > 0.0
        assert data[1] == pytest.approx(0.0)
        assert data[2] < 0.0

        # North and upward, at 0 toolface
        data = dogleg_toolface(np.pi/2, 0.0, 0.0, 0.002, 300)
        assert data[0] > 0.0
        assert data[1] == pytest.approx(0.0)
        assert data[2] < 0.0
    
    def test_azi_change(self):
        # Go north and down so far that azi should flip
        data = dogleg_toolface(np.pi/4, 0., np.pi, 0.002, 1400)
        assert data[1] == pytest.approx(0.0)
        assert data[2] > 0.0
        assert data[4] == pytest.approx(np.pi)


    def test_start_southwards(self):
        # print("From (pi/2, pi\n")

        # South westwards
        data = dogleg_toolface(np.pi/2, np.pi, 2*np.pi * 1/4, 0.002, 300)
        assert data[0] < 0.0
        assert data[1] < 0.0
        assert data[2] == pytest.approx(0.0)

        # Directly south and down
        data = dogleg_toolface(np.pi/2, np.pi, 2*np.pi * 2/4, 0.002, 300)
        assert data[0] < 0.0
        assert data[1] == pytest.approx(0.0)
        assert data[2] > 0

        # South eastwards
        data = dogleg_toolface(np.pi/2, np.pi, 2*np.pi * 3/4, 0.002, 300)
        assert data[0] < 0.0
        assert data[1] > 0.0
        assert data[2] == pytest.approx(0.0)

        # Directly south and up
        data = dogleg_toolface(np.pi/2, np.pi, 2*np.pi * 4/4, 0.002, 300)
        assert data[0] < 0.0
        assert data[1] == pytest.approx(0.0)
        assert data[2] < 0.0

        # Directly south and up, toolface 0
        data = dogleg_toolface(np.pi/2, np.pi, 0.0, 0.002, 300)
        assert data[0] < 0.0
        assert data[1] == pytest.approx(0.0)
        assert data[2] < 0.0

    def test_start_downwards(self):

        # North east
        data = dogleg_toolface(0., 0., 2*np.pi * 1/8, 0.001, 800)
        assert data[0] > 0.0
        assert data[1] > 0.0
        assert data[2] > 0.0

        # East and down
        data = dogleg_toolface(0., 0., 2*np.pi * 1/4, 0.002, 300)
        assert data[0] == pytest.approx(0.0)
        assert data[1] > 0.0
        assert data[2] > 0.0

        # South and down
        data = dogleg_toolface(0., 0., 2*np.pi * 2/4, 0.002, 300)
        assert data[0] < 0.0
        assert data[1] == pytest.approx(0.0)
        assert data[2] > 0.0

        # West and down
        data = dogleg_toolface(0., 0., 2*np.pi * 3/4, 0.002, 300)
        assert data[0] == pytest.approx(0.0)
        assert data[1] < 0.0
        assert data[2] > 0.0

        # North and down
        data = dogleg_toolface(0., 0., 2*np.pi * 4/4, 0.002, 300)
        assert data[0] > 0.0
        assert data[1] == pytest.approx(0.0)
        assert data[2] > 0.0

        # North andf down, tf=0
        segment = dogleg_toolface(0., 0., 0, 0.002, 300)
        assert data[0] > 0.0
        assert data[1] == pytest.approx(0.0)
        assert data[2] > 0.0

        # South east and down
        data = dogleg_toolface(0., 0., np.pi * 3/4, 0.002, 300)
        assert data[0] < 0.0
        assert data[1] > 0.0
        assert data[2] > 0.0
    
    def test_start_upwards(self):

        # North east
        data = dogleg_toolface(np.pi, 0., 2*np.pi * 1/8, 0.001, 800)
        assert data[0] > 0.0
        assert data[1] > 0.0
        assert data[2] < 0.0

        # East
        data = dogleg_toolface(np.pi, 0., 2*np.pi * 1/4,  0.002, 300)
        assert data[0] == pytest.approx(0.0)
        assert data[1] > 0.0
        assert data[2] < 0.0

        # South
        data = dogleg_toolface(np.pi, 0., 2*np.pi * 2/4, 0.002, 300)
        assert data[0] < 0.0
        assert data[1] == pytest.approx(0.0)
        assert data[2] < 0.0

        # West
        data = dogleg_toolface(np.pi, 0., 2*np.pi * 3/4, 0.002, 300)
        assert data[0] == pytest.approx(0.0)
        assert data[1] < 0.0
        assert data[2] < 0.0

        # North
        data = dogleg_toolface(np.pi, 0., 2*np.pi * 4/4, 0.002, 300)
        assert data[0] > 0.0
        assert data[1] == pytest.approx(0.0)
        assert data[2] < 0.0

        # North, tf=0
        data = dogleg_toolface(np.pi, 0., 0, 0.002, 300)
        assert data[0] > 0.0
        assert data[1] == pytest.approx(0.0)
        assert data[2] < 0.0

        # South east 
        data = dogleg_toolface(np.pi, 0., np.pi * 3/4, 0.002, 300)
        assert data[0] < 0.0
        assert data[1] > 0.0
        assert data[2] < 0.0

    def test_addition(self):
        # Verify addition identiy of gravity toolface

        dls = 0.002
        md = 300

        eps = 1e-3

        toolface_vals = np.linspace(0, 2*np.pi, num=100)

        azi_vals = np.linspace(0, 2*np.pi, num=100)

        for toolface in toolface_vals:
            for azi in azi_vals:
                full_step = dogleg_toolface(np.pi/2, azi, toolface, dls, md)
                part_step_1 = dogleg_toolface(np.pi/2, azi, toolface, dls, md/2)
                part_step_2 = dogleg_toolface(part_step_1[3], part_step_1[4], toolface, dls, md/2)

                north = part_step_1[0] + part_step_2[0]
                east = part_step_1[1] + part_step_2[1]
                tvd = part_step_1[2] + part_step_2[2]
                inc_lower = part_step_2[3]
                azi_lower = part_step_2[4]
                
                # Horribe foating point accuary with all this trigonometric wizardy
                eps_pos = 1e-0
                assert abs(full_step[0] - north) < eps_pos
                assert abs(full_step[1] - east)  < eps_pos
                assert abs(full_step[2] - tvd) <  eps_pos

                dn_part = np.cos(azi_lower)
                de_part = np.sin(azi_lower)
                dn_full = np.cos(full_step[4])
                de_full = np.sin(full_step[4])

                eps_dir = 1e-2
                assert abs(dn_full - dn_part) < eps_dir
                assert abs(de_full - de_part) < eps_dir
                assert abs(full_step[3] - inc_lower) < eps_dir

        

    def test_start_circle(self):
    
        # Circle east
        dls = 0.002
        r = 1/dls
        data = dogleg_toolface(np.pi/2, 0., np.pi/2, dls, 2*np.pi*r)

        print("n: ", data[0])
        print("e: ", data[1])
        print("t: ", data[2])

        assert data[0] == pytest.approx(0.0)
        assert data[1] == pytest.approx(0.0)
        assert data[2] == pytest.approx(0.0)

        # Circle west 
        dls = 0.002
        r = 1/dls
        data = dogleg_toolface(np.pi/2, 0., 2*np.pi*3/4, dls, 2*np.pi*r)

        print("n: ", data[0])
        print("e: ", data[1])
        print("t: ", data[2])

        assert data[0] == pytest.approx(0.0)
        assert data[1] == pytest.approx(0.0)
        assert data[2] == pytest.approx(0.0)

    def test_compare_ode_linalg(self):
        n_tries = 100

        for i in range(0, n_tries):
            inc0 = np.pi * random()
            azi0 = 2*np.pi * random()
            dls = 0.002
            md = 2*np.pi / dls * random()
            tf0 = 2*np.pi * random()

            state_circ = dogleg_toolface(inc0, azi0, tf0, dls, md)
            state_ode, sol, z = dogleg_toolface_ode(inc0, azi0, tf0, dls, md, False)

            diff = state_circ - state_ode
            assert sum(abs(diff)) < 1


    def test_transform(self):
        
        n=100
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