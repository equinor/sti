import csv
import numpy as np
import pandas as pd
from sti.sti_core import project_sti, standardize_pos, standardize_problem


def reverse_training_data():
    """
    Takes the merged traning data and reverses the
    order so that the target is the start, and vice
    versa. The intermediate points is also flipped.

    Finally, the problem is standardized to the new
    order.

    The point is to the boost the training
    data for the neural network.
    """

    filename = 'data/merged/data.csv'
    filename_out = 'data/merged/data_boost.csv'
    
    df = pd.read_csv(filename)

    n_rows = df.shape[0]

    with open(filename_out,'x') as file:
        headers = __get_header()
        writer = csv.writer(file)
        writer.writerow(headers)

    for i in range(0, n_rows):
        org_start =  df.iloc[i, 0:5]
        org_target = df.iloc[i, 5:10]
        dls = df.iloc[i,10]
        org_sti = df.iloc[i, 11:17]

        org_int_pos0 = df.iloc[i, 11:14]
        org_int_pos1 = df.iloc[i, 14:17]

        # Reverse start and target
        start_state, target_state = standardize_problem(org_target, org_start)

        # Transform and reverse intermediate points
        int_pos0_stand = standardize_pos(org_target, org_start, org_int_pos1)
        int_pos1_stand = standardize_pos(org_target, org_start, org_int_pos0)

        # Create sti
        sti = np.append(int_pos0_stand, int_pos1_stand).flatten()

        # Verify results
        projected_state, dls_actual, md = project_sti(start_state, target_state, sti)

        diff = target_state - projected_state
        sq_err = np.dot(diff, diff)

        acceptable = (sq_err < 1e-3)

        if acceptable:
            print("Row : ",i, " Error below threshold. Storing data point.")
            data_org = __merge_training_data(org_start, org_target, dls, org_sti)
            data = __merge_training_data(start_state, target_state, dls, sti)
            with open(filename_out,'a') as file:
                writer = csv.writer(file)
                writer.writerow(data_org)
                writer.writerow(data)
        else:
            print("Error above threshold. Will not store data point.")
            print("Error:  ", sq_err)
            print("Target:     ", target_state)
            print("Projection: ", projected_state)


# TODO Duplicate function from create_training_data.py
def __merge_training_data(start_state, target_state, dls_limit, sti):
    data = []
    data.extend(start_state)
    data.extend(target_state)
    data.append(dls_limit)
    data.extend(sti)

    return data


# TODO Duplicate function from create_training_data.py
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


if __name__ == '__main__':
    reverse_training_data()
