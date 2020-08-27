import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sti.sti_core import project_sti



def make_histograms():
    """
    Make histograms to evaluate a model
    on a holdout dataset
    """

    filename_holdout = 'data/merged/holdout.csv'
    
    df = pd.read_csv(filename_holdout)

    n_rows = df.shape[0]

    inc_err = []
    azi_err = []
    dls_err = []
    md_err = []

    if n_rows >= 25000:
        print("Capping dataset")
        n_rows = 25000

    for i in range(0, n_rows):
        if i % 100 == 0:
            print("Working... ", i, " of ", n_rows , "...")
        start_state =  df.iloc[i, 0:5].values
        target_state = df.iloc[i, 5:10].values
        dls_limit = df.iloc[i,10]
        sti = df.iloc[i, 11:17].values

        # Correct md
        _, _, org_md = project_sti(start_state, target_state, sti)

        # Load model
        with open('models/mlp.sav', 'rb') as file:
            model = pickle.load(file)

        # We're going to assume standarized format here
        assert sum(abs(start_state)) < 1e-3
        assert abs(target_state[1]) < 1e-3 

        # Predictor
        x = target_state
        x = np.append(x, dls_limit).flatten()

        # Predict using model
        pred_sti = model.predict(x.reshape(1, -1))
        pred_sti = pred_sti.flatten()

        # Projection using predicted sti
        pred_state, pred_dls, pred_md = project_sti(start_state, target_state, pred_sti)

        inc_diff = target_state[3] - pred_state[3]
        # if inc_diff < 0:
        #     inc_diff = inc_diff + np.pi
        # elif inc_diff > np.pi:
        #     inc_diff = inc_diff - np.pi
        inc_err.append(inc_diff)

        azi_diff = target_state[4] - pred_state[4]
        if azi_diff < -np.pi:
            azi_diff = 2*np.pi + azi_diff
        elif azi_diff > np.pi:
            azi_diff = 2*np.pi - azi_diff

        azi_err.append(azi_diff)

        dls_overshoot = pred_dls - dls_limit
        dls_err.append(dls_overshoot)

        md_err.append(org_md - pred_md)

    bins = 250
    plt.subplot(221)
    plt.hist(inc_err, bins=bins)
    plt.title('Inc. error')
    plt.xlabel('Incliation error, radians')
    plt.subplot(222)
    plt.hist(azi_err, bins=bins)
    plt.title('Azi. error')
    plt.xlabel('Azimuth error, radians')
    plt.subplot(223)
    plt.hist(dls_err, bins=bins, range=(0, 0.005))
    plt.title('Dog leg severity overshoot')
    plt.xlabel('DLS, radians/m')
    plt.subplot(224)
    plt.hist(md_err, bins=bins, range=(-2500,2500))
    plt.title('Measured depth error')
    plt.xlabel('MD, m')

    plt.show()


if __name__ == '__main__':
    make_histograms()
