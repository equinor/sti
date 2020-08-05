from pydantic import BaseModel, confloat
import numpy as np

class SurveyPoint(BaseModel):
    inc: confloat(ge=0, le=np.pi)
    azi: confloat(ge=0, le=2*np.pi)
    md_inc: confloat(ge=0)

class MinCurveSegment(BaseModel):
    dnorth: float
    deast: float
    dtvd: float
    dls: confloat(ge=0)

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
    # Stolen from wellpathpy
    cos_inc = np.cos(lowerSurvey.inc - upperSurvey.inc)
    sin_inc = np.sin(upperSurvey.inc) * np.sin(lowerSurvey.inc)
    cos_azi = 1 - np.cos(lowerSurvey.azi - upperSurvey.azi)

    dogleg = np.arccos(cos_inc - (sin_inc * cos_azi))
    md_inc = lowerSurvey.md_inc

    # ratio factor, correct for dogleg == 0 values
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        rf = 2 / dogleg * np.tan(dogleg / 2)
        rf = np.where(dogleg == 0., 1, rf)
    
    # delta northing
    upper = np.sin(upperSurvey.inc) * np.cos(upperSurvey.azi)
    lower = np.sin(lowerSurvey.inc) * np.cos(lowerSurvey.azi)
    dnorth = md_inc / 2 * (upper + lower) * rf
    
    # delta easting
    upper = np.sin(upperSurvey.inc) * np.sin(upperSurvey.azi)
    lower = np.sin(lowerSurvey.inc) * np.sin(lowerSurvey.azi)
    deast = md_inc / 2 * (upper + lower) * rf

    # delta tvd
    dtvd = md_inc / 2 * (np.cos(upperSurvey.inc) + np.cos(lowerSurvey.inc)) *rf

    minCurveSegment = MinCurveSegment(dnorth=dnorth, deast=deast, dtvd = dtvd, dls=dogleg)

    return minCurveSegment
