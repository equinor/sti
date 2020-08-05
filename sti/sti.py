from pydantic import BaseModel, confloat
from minimum_curvature import SurveyPoint, MinCurveSegment, getMinCurveSegment
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint

class State(BaseModel):
    north: float
    east: float
    tvd: float
    inc: confloat(ge=0, le=np.pi)
    azi: confloat(ge=0, le=2*np.pi)

    def __sub__(self, other):
        north = self.north - other.north
        east = self.east - other.east
        tvd = self.tvd - other.tvd
        inc = self.inc - other.inc
        azi = self.azi - other.azi

        return State(north=north, east=east, tvd=tvd, inc=inc, azi=azi)

    def __add__(self, other):
        north = self.north + other.north
        east = self.east + other.east
        tvd = self.tvd + other.tvd
        inc = self.inc + other.inc
        azi = self.azi + other.azi

        return State(north=north, east=east, tvd=tvd, inc=inc, azi=azi)

    def __mul__(self, other):
        north = self.north * other.north
        east = self.east * other.east
        tvd = self.tvd * other.tvd
        inc = self.inc * other.inc
        azi = self.azi * other.azi

        return State(north=north, east=east, tvd=tvd, inc=inc, azi=azi)
    
    def EuclidianDistance(self, other):
        north = self.north - other.north
        east = self.east - other.east
        tvd = self.tvd - other.tvd

        return (north**2 + east**2 + tvd**2) ** (0.5)

class Sti:
    # Note: Default DLS is equivalent to ~ 3.5 degrees / 30 
    def __init__(self, start: State, end: State, dls_limit: float=0.002, fit: bool=True, tol: float=1e-6):
        self.start = start
        self.end = end
        self.dls_limit = dls_limit
        self.fit = fit
        self.tol = tol

        self.dnorth = end.north - start.north
        self.deast = end.east  - start.east
        self.dtvd = end.tvd - start.tvd

        self.leg1 = SurveyPoint(inc=0, azi=0, md_inc=0)
        self.leg2 = SurveyPoint(inc=0, azi=0, md_inc=0)
        self.leg3 = SurveyPoint(inc=0, azi=0, md_inc=0)

        self.project()
        self.__fit_legs()


    def __set_legs_from_array(self, arr):
        """ Internal method used for fitting with optimization"""
        # TODO This is called alot could perhaps benefit from checking if an update is needed?

        # Truncate small negative violations from scipy optimize
        # TODO Also need to truncate positive violations
        arr[arr < 0] = 0.0


        self.leg1 = SurveyPoint(inc=arr[0], azi=arr[3], md_inc=arr[6])
        self.leg2 = SurveyPoint(inc=arr[1], azi=arr[4], md_inc=arr[7])
        self.leg3 = SurveyPoint(inc=arr[2], azi=arr[5], md_inc=arr[8])
        self.project()

    def __get_opti_funcs(self):
        def objective(arr):
            self.__set_legs_from_array(arr)
            md = self.leg1.md_inc + self.leg2.md_inc + self.leg3.md_inc
            return md

        def objective_grad(arr):
            return np.array([0., 0., 0., 0., 0., 0., 1., 1., 1.])
        
        def objective_hess(arr):
            return np.zeros((9,9))

        # Non-linear constraints on dog leg and target tolerance
        def non_linear_constrain(arr):
            self.__set_legs_from_array(arr)
            val = np.array(self.dls)
            val = np.append(val, self.mismatch)
            print(val)
            return val 
        
        nlc_lower_bound = np.array([-np.inf]*4)
        nlc_upper_bound = np.array([self.dls_limit]*3)
        nlc_upper_bound = np.append(nlc_upper_bound, self.tol)

        nlc = NonlinearConstraint(non_linear_constrain, nlc_lower_bound, nlc_upper_bound)

        return objective, objective_grad, objective_hess, nlc

    def __fit_legs(self):
        # TODO Copy params and set back to default if fitting fails
        
        # Initial guess
        eucdist = self.end.EuclidianDistance(self.start)
        x0 = [self.end.inc, 0, 0, self.end.azi, 0, 0, eucdist, 0, 0]

        objective, objective_grad, objective_hess, constraints = self.__get_opti_funcs()

        #Bounds on angles & md
        lower_bound = np.array([0]*9)
        upper_bound = np.array([[np.pi]*3, [2*np.pi]*3, [np.inf]*3]).flatten()
        bounds = Bounds(lower_bound, upper_bound, keep_feasible=True)

        result = minimize(objective, x0, jac=objective_grad, hess=objective_hess, bounds=bounds, constraints=constraints, method='trust-constr',options={'verbose': 2})
        self.__set_legs_from_array(result.x)

    def __calc_l2_mismatch(self):
        """ Squared Euclidian norm between target and current projected state. """
        state = self.end - self.projected_state
        state = state * state
        l2_norm  = (state.north + state.east + state.tvd + state.azi + state.inc)**(0.5)
        self.mismatch = l2_norm

    
    def project(self):
        """ Project the current legs """
        startSurvey = SurveyPoint(inc=self.start.inc, azi=self.start.azi, md_inc=0)

        seg1 = getMinCurveSegment(startSurvey, self.leg1)
        seg2 = getMinCurveSegment(self.leg1, self.leg2)
        seg3 = getMinCurveSegment(self.leg2, self.leg3)

        dnorth = seg1.dnorth + seg2.dnorth + seg3.dnorth
        deast = seg1.deast + seg2.deast + seg3.deast
        dtvd = seg1.dtvd + seg2.dtvd + seg3.dtvd

        north = self.start.north + dnorth
        east = self.start.east + deast
        tvd = self.start.tvd + dtvd

        state = State(north=north, east=east, tvd=tvd, inc=self.leg3.inc, azi=self.leg3.azi)
        dls = [seg1.dls, seg2.dls, seg3.dls]

        self.projected_state = state
        self.dls = dls
        self.__calc_l2_mismatch()

if __name__ == '__main__':
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

    start = State(**start_data)
    end = State(**end_data)
    w  = Sti(start, end)
    print(w.projected_state.north)
    print(w.projected_state.east)
    print(w.projected_state.tvd)