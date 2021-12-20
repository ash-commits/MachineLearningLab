import numpy as np

import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

heartDisease = pd.read_csv(r"C:\Users\sivaa\Downloads\heart.csv")
heartDisease = heartDisease.replace('?',np.nan)

print('Sample instances from dataset: ')
print(heartDisease.head())

print("Attributes and data types: ",heartDisease.dtypes)
model = BayesianModel([('age','target'),('sex','target'),('exang','target'),('cp','target'),('target','restecg'),('target','chol')])
print(model)

print("Learning using maximum likelihood estimators: ")
model.fit(heartDisease,estimator=MaximumLikelihoodEstimator)
print(model.get_cpds('age'))
print(model.get_cpds('sex'))
print(model.get_cpds('exang'))
print(model.get_cpds('cp'))
print(model.get_cpds('restecg'))

print('Inferring with Bayesian Network')
HeartDisease_infer = VariableElimination(model)
q1 = HeartDisease_infer.query(variables = ['target'], evidence = {'restecg':1})
q2 = HeartDisease_infer.query(variables = ['target'], evidence = {'age':40})
q3 = HeartDisease_infer.query(variables = ['target'], evidence = {'cp':3})
