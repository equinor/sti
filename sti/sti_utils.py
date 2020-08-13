import csv
import numpy as np
from sti.sti_core import faststi, project_sti
from datetime import datetime
from random import random


def create_training_data(n_straight_down, n_step_outs_v, n_step_outs_h, n_below_slot, n_fully_random):
    """ Produce training data for fitting a neural net model."""

    filename = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = filename + ".csv"

    MAX_ERR = 0.1

    def print_header(text):
        print("\n\n#################################################################")
        print(text)
        print("#################################################################") 
    
    with open(filename,'x') as file:
        headers = __get_header()
        writer = csv.writer(file)
        writer.writerow(headers)


    for i in range(0,n_straight_down):
        print_header("Straight down")

        dls_limit = random()*0.003 + 0.0015
        start_state = [0, 0, 0, 0, 0]
        target_state = [0, 0, random()*2500, 0, 0]

        sti , err= faststi(start_state, target_state, dls_limit=dls_limit)
        print_sti(start_state, target_state, sti, dls_limit)
        print("\nState mismatch: ", "{:.4f}".format(err))

        if err < MAX_ERR:
            data = __merge_training_data(start_state, target_state, dls_limit, sti)
            with open(filename,'a') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        else:
            print("Error above threshold, will not store datapoint.")

    for i in range(0, n_step_outs_v):
        print_header("Step out to vertical")

        dls_limit = random()*0.003 + 0.0015
        start_state = [0, 0, 0, 0, 0]
        target_state = [-750 + random()*1500, -750 + random()*1500, 2000+random()*2000, 0, 0]

        sti, err = faststi(start_state, target_state, dls_limit=dls_limit)
        print_sti(start_state, target_state, sti, dls_limit)
        print("State mismatch:", err)

        if err < MAX_ERR:
            data = __merge_training_data(start_state, target_state, dls_limit, sti)
            with open(filename,'a') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        else:
            print("Error above threshold, will not store datapoint.")

    for i in range(0, n_step_outs_h):
        print_header("Step out to horizontal")
        dls_limit = random()*0.003 + 0.0015
        start_state = [0, 0, 0, 0, 0]

        north = -2000+random()*4000
        east = -2000+random()*4000
        tvd = 2000+random()*2000

        azi = np.arctan2(east, north)
        if azi < 0:
            azi = azi + 2*np.pi

        target_state = [north, east, tvd, np.pi/2, azi]

        sti, err = faststi(start_state, target_state, dls_limit=dls_limit)
        print_sti(start_state, target_state, sti, dls_limit)
        print("State mismatch:", err)

        if err < MAX_ERR:
            data = __merge_training_data(start_state, target_state, dls_limit, sti)
            with open(filename,'a') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        else:
            print("Error above threshold, will not store datapoint.")

    for i in range(0, n_below_slot):
        print_header("Horizontal below KO")
        dls_limit = random()*0.003 + 0.0015
        start_state = [0, 0, 0, 0, 0]
        target_state = [0, 0, 2000+random()*2000, np.pi/2, random()*2*np.pi]

        sti, err = faststi(start_state, target_state, dls_limit=dls_limit)
        print_sti(start_state, target_state, sti, dls_limit)
        print("State mismatch:", err)

        if err < MAX_ERR:
            data = __merge_training_data(start_state, target_state, dls_limit, sti)
            with open(filename,'a') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        else:
            print("Error above threshold, will not store datapoint.")
    
    for i in range(0, n_fully_random):
        print_header("Fully random - but KO at (0,0,0)")
        dls_limit = random()*0.003 + 0.0015
        start_state = [0, 0, 0, random()*np.pi, random()*2*np.pi]
        target_state = [-4000+random()*8000, -4000+random()*8000, -4000+random()*8000, random()*np.pi/2, random()*2*np.pi]

        sti, err = faststi(start_state, target_state, dls_limit=dls_limit)
        print_sti(start_state, target_state, sti, dls_limit)
        print("State mismatch:", err)

        if err < MAX_ERR:
            data = __merge_training_data(start_state, target_state, dls_limit, sti)
            with open(filename,'a') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        else:
            print("Error above threshold, will not store datapoint.")


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
              "inc1",
              "inc2",
              "inc3",
              "azi1",
              "azi2",
              "azi3",
              "md_inc1",
              "md_inc2",
              "md_inc3",
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

def print_state(state):
    print("North: ", "{:.2f}".format(state[0]))
    print("East : ", "{:.2f}".format(state[1]))
    print("TVD  : ", "{:.2f}".format(state[2]))
    print("Inc. : ", "{:.4f}".format(state[3]))
    print("Azi. : ", "{:.4f}".format(state[4]))


def print_sti(start_state, target_state, sti, dls_limit):
    print("Start state: ")
    print("----------------")
    print_state(start_state)

    print("\nTarget state:")
    print("----------------")
    print_state(target_state)

    projected_state, dls = project_sti(start_state, sti)
    print("\nProjected state:")
    print("----------------")
    print_state(projected_state)

    tot_md = sti[6] + sti[7] + sti[8]
    print("\nMD start-target: ", "{:.2f}".format(tot_md))
    print("DLS limit: ", "{:.5f}".format(dls_limit))

    print("\nLegs:")
    print("----------------")
    print("Leg 1, toolface: ", "{:.4f}".format(sti[0]), " dls: ", "{:.4f}".format(sti[3]), " md_inc: ", "{:.2f}".format(sti[6]), "dls:", "{:.5f}".format(dls[0]))
    print("Leg 2, toolface: ", "{:.4f}".format(sti[1]), " dls: ", "{:.4f}".format(sti[4]), " md_inc: ", "{:.2f}".format(sti[7]), "dls:", "{:.5f}".format(dls[1]))
    print("Leg 3, toolface: ", "{:.4f}".format(sti[2]), " dls: ", "{:.4f}".format(sti[5]), " md_inc: ", "{:.2f}".format(sti[8]), "dls:", "{:.5f}".format(dls[2]))
    print("--------------------------------------------------------------\n")


if __name__ == '__main__':
    start_time = datetime.now()
    create_training_data(0, 15, 15, 15, 15)
    end_time = datetime.now()
    delta = end_time - start_time
    print("Elapsed walltime:")
    print(delta)
