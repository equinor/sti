import pytest
import numpy as np
from sti.sti_core import min_curve_segment, dogleg_toolface
from pydantic import ValidationError

class TestMinCurveSegment:
    def test_min_curvesegment_straight_east(self):
        inc1 = np.pi / 2 
        azi1 = np.pi / 2
        inc2 = np.pi / 2
        azi2 = np.pi / 2 
        md_inc = 893

        north, east, tvd, dls = min_curve_segment(inc1, azi1, inc2, azi2, md_inc)

        assert east == md_inc 
        assert north == pytest.approx(0.0)
        assert tvd == pytest.approx(0.0)
        assert dls == pytest.approx(0.0)


    def test_min_curvesegment_straight_west(self):
        inc1 = np.pi / 2 
        azi1 = 2*np.pi * 3/4
        inc2 = np.pi / 2
        azi2 = 2*np.pi * 3/4
        md_inc = 457

        north, east, tvd, dls = min_curve_segment(inc1, azi1, inc2, azi2, md_inc)

        assert east == -1*md_inc 
        assert north == pytest.approx(0.0)
        assert tvd == pytest.approx(0.0) 
        assert dls == pytest.approx(0.0) 


    def test_min_curvesegment_straight_north(self):
        inc1 = np.pi / 2 
        azi1 = 0
        inc2 = np.pi / 2
        azi2 = 0
        md_inc = 457

        north, east, tvd, dls = min_curve_segment(inc1, azi1, inc2, azi2, md_inc)

        assert east == pytest.approx(0.0)
        assert north == md_inc
        assert tvd == pytest.approx(0.0) 
        assert dls == pytest.approx(0.0) 


    def test_min_curvesegment_straight_south(self):
        inc1 = np.pi / 2
        azi1 = np.pi
        inc2 = np.pi / 2
        azi2 = np.pi
        md_inc = 345

        north, east, tvd, dls = min_curve_segment(inc1, azi1, inc2, azi2, md_inc)

        assert east == pytest.approx(0.0)
        assert north == -1*md_inc
        assert tvd == pytest.approx(0.0) 
        assert dls == pytest.approx(0.0) 


    def test_min_curvesegment_straight_down(self):
        inc1 = 0.0
        azi1 = 0.0
        inc2 = 0.0
        azi2 = 0.0
        md_inc = 454

        north, east, tvd, dls = min_curve_segment(inc1, azi1, inc2, azi2, md_inc)

        assert east == pytest.approx(0.0)
        assert north == pytest.approx(0.0)
        assert tvd == md_inc
        assert dls == pytest.approx(0.0) 


    def test_min_curvesegment_straight_up(self):
        inc1 = np.pi
        azi1 = 0.0
        inc2 = np.pi
        azi2 = 0.0
        md_inc = 454

        north, east, tvd, dls = min_curve_segment(inc1, azi1, inc2, azi2, md_inc)

        assert east == pytest.approx(0.0)
        assert north == pytest.approx(0.0)
        assert tvd == -1*md_inc
        assert dls == pytest.approx(0.0) 


    def test_circle_downward_east(self):
        inc1 = 0
        azi1 = np.pi / 2
        inc2 = np.pi / 2
        azi2 = np.pi / 2
        md_inc = 1000

        radius = 2*md_inc/np.pi

        north, east, tvd, dls = min_curve_segment(inc1, azi1, inc2, azi2, md_inc)

        assert north == pytest.approx(0)
        assert east == pytest.approx(radius)
        assert tvd == pytest.approx(radius)

    def test_circle_downward_west(self):
        inc1 = 0
        azi1 = 2*np.pi * 3/4
        inc2 = np.pi / 2
        azi2 = 2*np.pi * 3/4
        md_inc = 543

        radius = 2*md_inc/np.pi

        north, east, tvd, dls = min_curve_segment(inc1, azi1, inc2, azi2, md_inc)

        assert north == pytest.approx(0)
        assert east == pytest.approx( -1*radius)
        assert tvd == pytest.approx(radius)

    def test_circle_upward_north(self):
        inc1 = np.pi
        azi1 = 0.0
        inc2 = np.pi / 2
        azi2 = 0.0
        md_inc = 452

        radius = 2*md_inc/np.pi

        north, east, tvd, dls = min_curve_segment(inc1, azi1, inc2, azi2, md_inc)

        assert north == pytest.approx(radius)
        assert east == pytest.approx(0.0)
        assert tvd == pytest.approx(-1*radius)

    # def test_zero_md_inc(self):

    #     inc1 = np.pi / 2
    #     azi1 = np.pi
    #     inc2 = np.pi / 2
    #     azi2 = np.pi
    #     md_inc = 0.0

    #     north, east, tvd, dls = min_curve_segment(inc1, azi1, inc2, azi2, md_inc)

    #     assert east == pytest.approx(0.0)
    #     assert north == pytest.approx(0.0)
    #     assert tvd == pytest.approx(0.0) 
    #     assert dls == pytest.approx(0.0) 


# class TestProjectSti():
#     def test_project_circle_back_to_start(self):

#         from_state = np.array([0, 0, 0, np.pi / 2, 0])
#         radius = 3000
#         inc1 = np.pi/2
#         inc2 = np.pi
#         inc3 = np.pi/2
#         azi1 = np.pi
#         azi2 = 0
#         azi3 = 0
#         md_inc1 = 2 * np.pi * radius * 1/4
#         md_inc2 = 2 * np.pi * radius * 1/2
#         md_inc3 = 2 * np.pi * radius * 1/4
#         #               inc1, inc2, inc3, azi1, azi2, azi3, md_inc1, md_inc2, md_inc3
#         sti = np.array([inc1, inc2, inc3, azi1, azi2, azi3, md_inc1, md_inc2, md_inc3])

#         to_state = project_sti(from_state, sti)
#         print(to_state)

#         norm = (from_state - to_state)**2
#         print(norm)

#         assert norm == pytest.approx(0.0)

class TestDogLegToolFace():
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
        data = dogleg_toolface(np.pi/2, 0.,0, 0.002, 300)
        assert data[0] > 0.0
        assert data[1] == pytest.approx(0.0)
        assert data[2] < 0.0

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
        data = dogleg_toolface(np.pi/2, np.pi, 0, 0.002, 300)
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
        data = dogleg_toolface(np.pi, 0., 2*np.pi * 1/4, 0.002, 300)
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

        toolface = np.pi/2

        full_step = dogleg_toolface(np.pi/2, 0., toolface, dls, md)
        part_step_1 = dogleg_toolface(np.pi/2, 0., toolface, dls, md/2)
        part_step_2 = dogleg_toolface(part_step_1[3], part_step_1[4], toolface, dls, md/2)

        north = part_step_1[0] + part_step_2[0]
        east = part_step_1[1] + part_step_2[1]
        tvd = part_step_1[2] + part_step_2[2]

        assert full_step[0] == pytest.approx(north)
        assert full_step[1] == pytest.approx(east)
        assert full_step[2] == pytest.approx(tvd)

        

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
