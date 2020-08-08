import pandas as pd
import numpy as np
import sti.sti_core as sti_core

"""
Helper script to remove bad training data
"""
MAX_ERR = 0.1

# Input data
filename_data ='data.csv'
filename_clean = 'data_cleaned.csv'

# Read dataset
df = pd.read_csv(filename_data)

# Find rows with bad data
bad = []
for i in range(0, df.shape[0]):
    states = np.array(df.iloc[i,0:11].values)
    from_state = states[0:5]
    to_state = states[5:10]
    dls_limit = states[10]

    sti = np.array(df.iloc[i,11:])
    projection, dls = sti_core.project_sti(from_state, sti)
    err = sti_core.__err_squared_state_mismatch(to_state, projection, dls_limit, scale_md=100)

    if err > MAX_ERR:
        print("i: ", i, "err: ", err)
        bad.append(i)

df = df.drop(bad)
df.to_csv(filename_clean)