from pydantic import BaseModel, confloat
import numpy as np

class SurveyPoint(BaseModel):
    inc: confloat(ge=-1e-3, le=np.pi+1e-3)
    azi: confloat(ge=-1e-3, le=2*np.pi+1e-3)
    md_inc: confloat(ge=0)

class MinCurveSegment(BaseModel):
    dnorth: float
    deast: float
    dtvd: float
    dls: confloat(ge=0)

def __getMinCurveSegmentInner(inc_upper, azi_upper, inc_lower, azi_lower, md_inc):
    """Inner workhorse, designed for auto differentiability."""
    # Stolen from wellpathpy
    cos_inc = np.cos(inc_lower - inc_upper)
    sin_inc = np.sin(inc_upper) * np.sin(inc_lower)
    cos_azi = 1 - np.cos(azi_lower - azi_upper)

    dogleg = np.arccos(cos_inc - (sin_inc * cos_azi))

    # ratio factor, correct for dogleg == 0 values
    if np.isclose(dogleg, 0.):
        dogleg = 0
        rf = 1
    else:
        rf = 2 / dogleg * np.tan(dogleg / 2)
    
    # delta northing
    upper = np.sin(inc_upper) * np.cos(azi_upper)
    lower = np.sin(inc_lower) * np.cos(azi_lower)
    dnorth = md_inc / 2 * (upper + lower) * rf
    
    # delta easting
    upper = np.sin(inc_upper) * np.sin(azi_upper)
    lower = np.sin(inc_lower) * np.sin(azi_lower)
    deast = md_inc / 2 * (upper + lower) * rf

    # delta tvd
    dtvd = md_inc / 2 * (np.cos(inc_upper) + np.cos(inc_lower)) *rf

    dls = dogleg / md_inc

    return dnorth, deast, dtvd, dls
    
def getMinCurveSegment(upperSurvey: SurveyPoint, lowerSurvey: SurveyPoint):
    """Calculate increments in TVD, northing, easting and associated dogleg
    using the minimum curvature method.

    Assumes inc and azi in radians. Returns dog leg severity in rad/length.

    Parameters
    ----------
    upper: Upper survey point
    lower: Lower survey point

    Returns
    -------
    minCurveSegment: MinCurveSegment
    """
    dnorth, deast, dtvd, dls = __getMinCurveSegmentInner(upperSurvey.inc, upperSurvey.azi, lowerSurvey.inc, lowerSurvey.azi, lowerSurvey.md_inc)

    minCurveSegment = MinCurveSegment(dnorth=dnorth, deast=deast, dtvd = dtvd, dls=dls)

    return minCurveSegment
