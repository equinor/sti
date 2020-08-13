import pytest
import numpy as np
from sti.sti_core import min_curve_segment
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

