import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\sivaa\Downloads\ENJOYSPORT.csv")
concepts = np.array(df)[:, :-1]
target = np.array(df)[:, -1]


def learn(concepts, target):
    spec_hypo = concepts[0].copy()
    gene_hypo = [['?' for i in range(len(spec_hypo))] for i in range(len(spec_hypo))]
    for i, val in enumerate(concepts):
        if target[i] == 1:
            for x in range(len(spec_hypo)):
                if val[x] != spec_hypo[x]:
                    spec_hypo[x] = '?'
                    gene_hypo[x][x] = '?'
        if target[i] == 0:
            for x in range(len(spec_hypo)):
                if val[x] != spec_hypo[x]:
                    gene_hypo[x][x] = spec_hypo[x]
                else:
                    gene_hypo[x][x] = '?'
    indices = [i for i, val in enumerate(gene_hypo) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        gene_hypo.remove(['?', '?', '?', '?', '?', '?'])
    return spec_hypo,gene_hypo


final_spec, final_gene = learn(concepts,target)
print('Specific hypo',final_spec,'/nGeneral hypo',final_gene)
