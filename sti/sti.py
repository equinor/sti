from pydantic import BaseModel, confloat
from minimum_curvature import SurveyPoint, MinCurveSegment, getMinCurveSegment
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.optimize import differential_evolution

class State(BaseModel):
    north: float
    east: float
    tvd: float
    inc: float
    azi: float

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
    def __init__(self, start: State, end: State, dls_limit: float=0.002, fit: bool=True, tol: float=1e-3):
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


        # Stuff used for fittig with optimization, really internal shit...
        self.SCALE_MD = 100
        self.lower_bound = np.array([0.]*6)
        # HACK To void zero length segments, which breaks the Sti.project() method!
        self.lower_bound = np.append(self.lower_bound,[1/self.SCALE_MD]*3)
        self.upper_bound = np.array([[np.pi]*3, [2*np.pi]*3, [10000/self.SCALE_MD]*3]).flatten()

        # Fit parameters with optimization
        self.__fit_legs()

    def __truncate_to_bounds(self, arr, eps=0):
        # Truncate small negative violations from scipy optimize
        # eps is a workaround for https://github.com/scipy/scipy/issues/11403
        val = np.maximum(arr, self.lower_bound + eps)
        val = np.minimum(val, self.upper_bound - eps)

            
        return val


    def __set_legs_from_array(self, arr):
        """ Internal method used for fitting with optimization, expects scaled inputs"""
        # TODO This is called alot could perhaps benefit from checking if an update is needed?
        arr = self.__truncate_to_bounds(arr)

        self.leg1 = SurveyPoint(inc=arr[0], azi=arr[3], md_inc=self.SCALE_MD*arr[6])
        self.leg2 = SurveyPoint(inc=arr[1], azi=arr[4], md_inc=self.SCALE_MD*arr[7])
        self.leg3 = SurveyPoint(inc=arr[2], azi=arr[5], md_inc=self.SCALE_MD*arr[8])
        self.project()

    def __get_objective_functions(self):
        def objective_reach(arr):
            """ Method used to find an initial feasible guess with less focus on length"""
            
            # Scale factor applied to MD to make it less important in the iteations
            MD_WEIGHT = 1/10000

            # Project current values
            self.__set_legs_from_array(arr)

            # Dog leg severity penality
            dls = np.array(self.dls)
            dls_mis = (np.maximum(dls, self.dls_limit) - self.dls_limit) / self.dls_limit
            dls_mis = dls_mis**2
            dls_mis = sum(dls_mis)

            # Penality for missing target
            mismatch = (self.mismatch / self.SCALE_MD)**2 

            md = self.leg1.md_inc + self.leg2.md_inc + self.leg3.md_inc
            md = (md / self.SCALE_MD)**2
            # print("dls_mis: ", dls_mis, " mismatch: ", mismatch, " md: ", MD_WEIGHT*md)

            return dls_mis + mismatch + MD_WEIGHT * md

        def objective_min_md(arr):
            """ Method used to shorten a smooth guess. """
            md = self.leg1.md_inc + self.leg2.md_inc + self.leg3.md_inc
            md = md / self.SCALE_MD

            return md 
        
        def objective_min_md_grad(arr):
            return np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]) / self.SCALE_MD
        
        def objective_min_md_hess(arr):
            return np.zeros((9,9))
 
        return objective_reach, objective_min_md, objective_min_md_grad, objective_min_md_hess

    def __get_nlc(self, keep_feasible=False):
        """ Get non-linear constraints. """
        
        def non_linear_constrain(arr):
            self.__set_legs_from_array(arr)
            val = np.array(self.dls)
            val = np.append(val, self.mismatch)
            return val 
        
        nlc_lower_bound = np.array([-np.inf]*4)
        nlc_upper_bound = np.array([self.dls_limit]*3)
        nlc_upper_bound = np.append(nlc_upper_bound, self.tol)

        nlc = NonlinearConstraint(non_linear_constrain, nlc_lower_bound, nlc_upper_bound, keep_feasible=keep_feasible)

        return nlc
    
    def __fit_legs(self):
        # TODO Copy params and set back to default if fitting fails
        
        objective_reach, objective_min_md, objective_min_md_grad, objective_min_md_hess= self.__get_objective_functions()

        # Bounds on angles & md
        bounds = Bounds(lb=self.lower_bound, ub=self.upper_bound, keep_feasible=True)

        # Try to find an initial feasible solution as a best guess before minizming length
        result = differential_evolution(objective_reach, bounds, maxiter=1000)
        
        print("INITIAL GUESS ")
        print("------------------")
        self.print_debug()
        
        # HACK Check initial geuss has feasible non-linear constraints, dirty way, TODO - improve!
        self.__set_legs_from_array(result.x)
        if (self.mismatch) < self.tol and (max(self.dls) < self.dls_limit):
            print("feasible nlc from inital guess")
            feasible = True
        else:
            print("in-feasible nlc from inital guess")
            feasible = False
        
        # Use results from feasibility evalution when getting the non-linear constraint
        nlc = self.__get_nlc(keep_feasible=feasible)

        # Use previous result to try to shorten well, include eps to fix bug https://github.com/scipy/scipy/issues/11403
        x0 = self.__truncate_to_bounds(result.x, eps=1e-2)
        result = minimize(objective_min_md, x0, bounds=bounds, method='trust-constr', constraints=nlc, options={'verbose': 1})

        print("\nFINAL ESTIMATE ")
        print("------------------")
        self.print_debug()

        # HACK Check initial geuss has feasible non-linear constraints, dirty way, TODO - improve!
        self.__set_legs_from_array(result.x)
        if (self.mismatch) < self.tol and (max(self.dls) < self.dls_limit):
            print("Feasible non-linear constraints")
        else:
            print("In-feasible non-linear constraints")

        self.__set_legs_from_array(result.x)

    def __calc_l2_mismatch(self):
        """ Scaled  norm between target and current projected state. """
        state = self.end - self.projected_state
        state = state * state
        l2_norm  = (state.north + state.east + state.tvd + (self.dls_limit)**(-2)*(state.azi + state.inc))**(0.5)
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

        md = self.leg1.md_inc + self.leg2.md_inc + self.leg3.md_inc

        state = State(north=north, east=east, tvd=tvd, inc=self.leg3.inc, azi=self.leg3.azi)
        dls = [seg1.dls, seg2.dls, seg3.dls]

        self.projected_state = state
        self.dls = dls
        self.__calc_l2_mismatch()
    
    def print_debug(self):
        print("North: ",self.projected_state.north)
        print("East: ",self.projected_state.east)
        print("TVD: ",self.projected_state.tvd)
        print("Inc: ",self.projected_state.inc)
        print("Azi: ",self.projected_state.azi)

        md_tot = self.leg1.md_inc + self.leg2.md_inc + self.leg3.md_inc
        print("MD:", md_tot)

        print("Leg 1, inc: ", self.leg1.inc, " azi: ", self.leg1.azi, " md_inc: ", self.leg1.md_inc, "dls:", self.dls[0])
        print("Leg 2, inc: ", self.leg2.inc, " azi: ", self.leg2.azi, " md_inc: ", self.leg2.md_inc, "dls:", self.dls[1])
        print("Leg 3, inc: ", self.leg3.inc, " azi: ", self.leg3.azi, " md_inc: ", self.leg3.md_inc, "dls:", self.dls[2])
        print("Mismatch: ", self.mismatch, " tol: ", self.tol)

if __name__ == '__main__':
    start_data = {
        'north': 0,
        'east' : 0,
        'tvd': 0,
        'inc': 0,
        'azi': 0 
    }

    end_data = {
        'north': 0,
        'east' : 0,
        'tvd':  2500,
        'inc': np.pi/2,
        'azi': np.pi/4 
    }

    start = State(**start_data)
    end = State(**end_data)
    w  = Sti(start, end)

