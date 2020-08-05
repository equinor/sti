from pytest import approx
import sti.minimum_curvature as mc
import numpy as np

class TestMinCurveSegment:
    def test_zero_md_inc(self):
        upper_data = {
            'inc' : 0,
            'azi' : 0,
            'md_inc'  : 0,
        }
        upper = mc.SurveyPoint(**upper_data)
        lower = upper

        segment = mc.getMinCurveSegment(upper, lower)

        assert segment.dnorth == 0
        assert segment.deast == 0
        assert segment.dtvd == 0
        
    def test_straight_down(self):
        upper_data = {
            'inc' : 0,
            'azi' : 0,
            'md_inc'  : 0,
        }
        upper = mc.SurveyPoint(**upper_data)

        lower_data = {
            'inc' : 0,
            'azi' : 0,
            'md_inc'  : 123,
        }
        lower = mc.SurveyPoint(**lower_data)

        segment = mc.getMinCurveSegment(upper, lower)

        assert segment.dnorth == 0
        assert segment.deast == 0
        assert segment.dtvd == 123

    def test_straight_east(self):
        azi = np.pi / 2
        inc = np.pi / 2
        upper_data = {
            'inc' : inc,
            'azi' : azi,
            'md_inc'  : 0,
        }
        upper = mc.SurveyPoint(**upper_data)

        lower_data = {
            'inc' : inc,
            'azi' :  azi,
            'md_inc'  : 234,
        }
        lower = mc.SurveyPoint(**lower_data)

        segment = mc.getMinCurveSegment(upper, lower)

        assert segment.dnorth == approx(0)
        assert segment.deast == approx(234)
        assert segment.dtvd == approx(0)
    
    def test_straight_west(self):
        azi = 3 / 4 * 2 * np.pi 
        inc = np.pi / 2
        upper_data = {
            'inc' : inc,
            'azi' : azi,
            'md_inc'  : 0,
        }
        upper = mc.SurveyPoint(**upper_data)

        lower_data = {
            'inc' : inc,
            'azi' :  azi,
            'md_inc'  : 544,
        }
        lower = mc.SurveyPoint(**lower_data)

        segment = mc.getMinCurveSegment(upper, lower)

        assert segment.dnorth == approx(0)
        assert segment.deast == approx(-544)
        assert segment.dtvd == approx(0)

    def test_straight_north(self):
        azi = 0
        inc = np.pi / 2
        upper_data = {
            'inc' : inc,
            'azi' : azi,
            'md_inc'  : 0,
        }
        upper = mc.SurveyPoint(**upper_data)

        lower_data = {
            'inc' : inc,
            'azi' :  azi,
            'md_inc'  : 578,
        }
        lower = mc.SurveyPoint(**lower_data)

        segment = mc.getMinCurveSegment(upper, lower)

        assert segment.dnorth == approx(578)
        assert segment.deast == approx(0)
        assert segment.dtvd == approx(0)

    def test_circle_downward_east(self):
        azi = np.pi / 2
        inc = 0
        upper_data = {
            'inc' : inc,
            'azi' : azi,
            'md_inc'  : 0,
        }
        upper = mc.SurveyPoint(**upper_data)


        azi = np.pi / 2
        inc = np.pi / 2
        md_inc = 1000
        lower_data = {
            'inc' : inc,
            'azi' :  azi,
            'md_inc'  : md_inc,
        }
        lower = mc.SurveyPoint(**lower_data)

        segment = mc.getMinCurveSegment(upper, lower)

        radius = 2*md_inc/np.pi

        assert segment.dnorth == approx(0)
        assert segment.deast == approx(radius)
        assert segment.dtvd == approx(radius)

    def test_circle_upward_east(self):
        azi = np.pi / 2
        inc = np.pi / 2
        upper_data = {
            'inc' : inc,
            'azi' : azi,
            'md_inc'  : 0,
        }
        upper = mc.SurveyPoint(**upper_data)


        azi = np.pi / 2
        inc = np.pi 
        md_inc = 500
        lower_data = {
            'inc' : inc,
            'azi' :  azi,
            'md_inc'  : md_inc,
        }
        lower = mc.SurveyPoint(**lower_data)

        segment = mc.getMinCurveSegment(upper, lower)

        radius = 2*md_inc/np.pi

        assert segment.dnorth == approx(0)
        assert segment.deast == approx(radius)
        assert segment.dtvd == approx(-radius)
