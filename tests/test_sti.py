import sti.sti as sti
import pytest

class TestState():
    def test_constructor(self):
        north = 324
        east = 878
        tvd = 234
        inc = 0.4
        azi = 2.1
        
        state = sti.State(north=north, east=east, tvd=tvd, inc=inc, azi=azi)

        assert state.north == pytest.approx(north)
        assert state.east == pytest.approx(east)
        assert state.tvd == pytest.approx(tvd)
        assert state.inc == pytest.approx(inc)
        assert state.azi == pytest.approx(azi)

    def test_subtract_self(self):
        state = sti.State(north=1, east=2, tvd=3, inc=0.5, azi=2)
        state = state - state
        assert state.north == pytest.approx(0)
        assert state.east == pytest.approx(0)
        assert state.tvd == pytest.approx(0)
        assert state.inc == pytest.approx(0)
        assert state.azi == pytest.approx(0)

    def test_multiply(self):
        north = 324
        east = 878
        tvd = 234
        inc = 0.4
        azi = 2.1
        
        state = sti.State(north=north, east=east, tvd=tvd, inc=inc, azi=azi)
        state = state * state

        assert state.north == pytest.approx(north*north)
        assert state.east == pytest.approx(east*east)
        assert state.tvd == pytest.approx(tvd*tvd)
        assert state.inc == pytest.approx(inc*inc)
        assert state.azi == pytest.approx(azi*azi)

    def test_add(self):
        north = 324
        east = 878
        tvd = 234
        inc = 0.4
        azi = 2.1
        
        state = sti.State(north=north, east=east, tvd=tvd, inc=inc, azi=azi)
        state = state + state

        assert state.north == pytest.approx(north+north)
        assert state.east == pytest.approx(east+east)
        assert state.tvd == pytest.approx(tvd+tvd)
        assert state.inc == pytest.approx(inc+inc)
        assert state.azi == pytest.approx(azi+azi)

class TestSti():
    def test_constructor(self):
        start_data = {
            'north': 1,
            'east' : 2,
            'tvd': 3,
            'inc': 0.4,
            'azi': 0.5 
        }

        end_data = {
            'north': 1001,
            'east' : 2002,
            'tvd':  3003,
            'inc': 3.14 / 2,
            'azi': 3.14 / 3
        }

        start = sti.State(**start_data)
        end = sti.State(**end_data)
        w  = sti.Sti(start, end)