import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\sivaa\Downloads\ENJOYSPORT.csv")
attr = np.array(df)[:, 0:-1]
targets = np.array(df)[:, -1]


def training(attr, targets):
    for i, val in enumerate(targets):
        if val == 1:
            spec_hypo = attr[i].copy()
            break
    for i, val in enumerate(attr):
        if targets[i] == 1:
            for x in range(len(spec_hypo)):
                if val[x] != spec_hypo[x]:
                    spec_hypo[x] = '?'
                else:
                    pass
    return spec_hypo


find_s = training(attr, targets)
print("Final Specific Hypothesis: ",find_s)
