import csv
import numpy as np
from sti.sti_core import find_sti, project_sti, standardize_problem, inverse_standardization_pos, standardized_initial_guess
from datetime import datetime
from random import random


def create_training_data(n_points):
    """
    Create training data and write to file

    We sample with uniform input to spherical co-ordinates.
    The idea is to get dense sampling in the most difficult
    reigon to approximate, that is large change in angle
    with small euclidian distance, which requires circle turns
    and what not.
    
    Also, as this method will be used to calculate drillable
    distances, we can expect most from - to problems to be
    quite a lot less than 10000 meters appart.
    
    At last we exploit the symmetry of the problem as much as
    possible by standardizing the input probem.
    
    First, we translate the problem so that the start
    location is at 0
    
    Then we create a new co-ordinate system where the bit
    start direction is (0,0,1) and the target position is
    ortogonal to (0,1,0) by leting north be the orthogonal
    residual of the target position to the tvd as defined by
    the bit. East is then defined by their cross product
    
    Also, we flip the sign of east so that the target bit
    azimuth is always in [0, pi]
    
    By doing so, we can sample only:
    - Target position in north-tvd plane
    - Target inclination in [0, pi]
    - Target azimuth in [0, pi]
    """

    filename = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = "data/raw/" + filename + ".csv"
    
    with open(filename,'x') as file:
        headers = __get_header()
        writer = csv.writer(file)
        writer.writerow(headers)

    for i in range(0, n_points):
        # By definition
        start_state = np.array([0., 0., 0., 0., 0.])

        dls_limit = 0.0005 + 0.005*random() # From 0.85 to 9.45 degree / 30m 

        radius_frac = random() * random()
        radius = 10000 * radius_frac
        pos_inc = np.pi * random()
        pos_azi = 0.0#2*np.pi * random()

        north = radius * np.sin(pos_inc) * np.cos(pos_azi)
        east = radius * np.sin(pos_inc) * np.sin(pos_azi)
        tvd = radius * np.cos(pos_inc)


        # In the standardized problem, east is selected so that 
        # the target bit azi is always in [0, pi]
        azi = 1*np.pi * random()
        inc = np.pi * random()

        scale_md = 4000

        target_state = np.array([north, east, tvd, inc, azi])

        sti, acceptable = find_sti(start_state, target_state, dls_limit, scale_md)

        projected_state, dls_actual, md = project_sti(start_state, target_state, sti)

        print("Start: ", start_state)
        print("Target: ", target_state)
        print("Projected: ", projected_state)

        # See where the model would take us
        start_stand, target_stand = standardize_problem(start_state, target_state)
        stand_sti = standardized_initial_guess(start_stand, target_stand, dls_limit)

        int_pos_0_stand = stand_sti[0:3]
        int_pos_1_stand = stand_sti[3:6]

        # Translate sti points back to physical space
        int_pos_0 = inverse_standardization_pos(start_state, target_state, int_pos_0_stand)
        int_pos_1 = inverse_standardization_pos(start_state, target_state, int_pos_1_stand)

        sti_pred = np.append(int_pos_0, int_pos_1).flatten()
        projected_pred, _ , md_pred = project_sti(start_state, target_state, sti_pred)
        print("Proj. mod.", projected_pred)
        print("MD: ", md)
        print("Md mod: ", md_pred)
        print("DLS Actual: ", dls_actual)
        print("DLS Limit: ", dls_limit)

        if acceptable:
            print("Error below threshold. Storing data point.")
            data = __merge_training_data(start_state, target_state, dls_limit, sti)
            with open(filename,'a') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            # Store the datapoint
        else:
            print("Error above threshold. Will not store data point.")

def __merge_training_data(start_state, target_state, dls_limit, sti):
    data = []
    data.extend(start_state)
    data.extend(target_state)
    data.append(dls_limit)
    data.extend(sti)

    return data


def __get_header():
    header = ["start_north",
              "start_east",
              "start_tvd",
              "start_inc",
              "start_azi",
              "target_north",
              "target_east",
              "target_tvd",
              "target_inc",
              "target_azi",
              "dls_limit",
              "n0",
              "e0",
              "t0",
              "n1",
              "e1",
              "t1",
    ]

    return header


def __merge_info(start_state, target_state, dls_limit, sti):
    merged = np.array()
    merged.append(start_state)
    merged.append(target_state)
    merged.append(dls_limit)
    merged.append(sti)

    merged.flatten()

    return merged


if __name__ == '__main__':
    create_training_data(1000)
