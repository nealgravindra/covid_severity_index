import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import glob
import pickle
import random

!pip install xgboost
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit, train_test_split, KFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid

import shap

from hyperopt.pyll.stochastic import sample
from timeit import default_timer as timer
from hyperopt import tpe, Trials, fmin, hp, STATUS_OK

from scipy import stats

method = 'XGB'
model_name = 'CSIbst'
min_feat = ['last_SPO2','last_o2_flow','last_RR','bun',
                        'ast','age','last_SBP',
                        'glucose',
                        'WBC','PROCAL','FERRITIN','CRP',
                        'creatinine','chloride','alt']

# hyperopt
def objective(hyperparameters):
        '''Objective function to evaluate sequential based model optimiatization via `hyperopt`.
    
    This function will be called by `hyperopt.fmin` with a value generated from `space`
    as the first arg.  It can return either a scalar-valued loss, or a dictionary.  
    
    NOTE: A returned dictionary must contain a 'status' key with a value from `STATUS_STRINGS`,  
        and a 'loss' key if the status is `STATUS_OK`.
                  
    Args: 
        hyperparameters (dict): set to optimize
        params (dict, GLOBAL): set of params in model to fix; globally set to avoid `hyperot` 
            errors.
        int_parms (list, GLOBAL): list of parameters to check to make sure 
            globally set to avoid `hyperot` errors.
        
    Returns:
        dict: with loss value for objective function specified in body and status flag
    '''
            start = timer()

            #     ITERATION += 1

                if INT_PARAMS is not None:
                            # Make sure parameters that need to be integers are integers
                            for parameter_name in INT_PARAMS:
                                            hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

                                                # merge parameters
                                                    trial_params = {**hyperparameters, **PARAMS}

                                                        if False:
                                                                    # by xgb.cv
                                                                    metric = 'auc'
                                                                            xgb_optim = xgb.cv(
                                                                                            trial_params,
                                                                                            dtrain,
                                                                                            num_boost_round=100,
                                                                                            nfold=5,
                                                                                            metrics={metric},
                                                                                            early_stopping_rounds=10)
                                                                                    best_score = xgb_optim.sort_values(by='test-{}-mean'.format(metric),ascending=False)
                                                                                            best_score = best_score['test-{}-mean'.format(metric)].iloc[0]

                                                        elif True:
                                                                    # by custom CV (take best so consistent with bootstrapping)

                                                                    # cv
                                                                    kfolds = 10
                                                                            kf = KFold(n_splits=kfolds, shuffle=True)

                                                                                    num_round = 20000
                                                                                            early_stopping = 500

                                                                                                    eval_metric = []
                                                                                                            for idx_train, idx_val in kf.split(X_train_sub):
                                                                                                                            X_train_cv, X_val = X_train_sub.iloc[idx_train,:], X_train_sub.iloc[idx_val,:]
                                                                                                                            #             y_train_cv, y_val = y_train.iloc[idx_train,:],y_train.iloc[idx_val,:]
                                                                                                                                        y_train_cv, y_val = y_train.iloc[idx_train],y_train.iloc[idx_val]

                                                                                                                                                    if False:
                                                                                                                                                                        # imputation
                                                                                                                                                                        X_train_cv = SimpleImputer(verbose=1, strategy='median').fit_transform(X_train_cv)
                                                                                                                                                                                        X_val = SimpleImputer(verbose=1, strategy='median').fit_transform(X_val)

                                                                                                                                                                                                    dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv)
                                                                                                                                                                                                                dval = xgb.DMatrix(X_val, label=y_val)

                                                                                                                                                                                                                            estimator = xgb.train(
                                                                                                                                                                                                                                                trial_params,
                                                                                                                                                                                                                                                dtrain,
                                                                                                                                                                                                                                                num_boost_round=num_round,
                                                                                                                                                                                                                                                evals=[(dtrain,'train'), (dval, 'val')],
                                                                                                                                                                                                                                                verbose_eval=False,
                                                                                                                                                                                                                                                early_stopping_rounds=early_stopping)
                                                                                                                                                                                                                                        eval_metric.append(estimator.best_score)
                                                                                                                                                                                                                                                best_score = np.max(eval_metric)


                                                                                                                                                    else:
                                                                                                                                                                # similar to eval step
                                                                                                                                                                num_round = 20
                                                                                                                                                                        early_stopping = 10

                                                                                                                                                                                model = xgb.train(
                                                                                                                                                                                                trial_params,
                                                                                                                                                                                                dtrain,
                                                                                                                                                                                                num_boost_round=num_round,
                                                                                                                                                                                                evals=[(dtrain,'train'), (dtest, 'test')],
                                                                                                                                                                                                early_stopping_rounds=early_stopping,
                                                                                                                                                                                                verbose_eval = num_round,
                                                                                                                                                                                            )
                                                                                                                                                                                        best_score = model.best_score

                                                                                                                                                                                            # loss fx
                                                                                                                                                                                                loss = 1 - best_score

                                                                                                                                                                                                    run_time = timer() - start

                                                                                                                                                                                                        if False :
                                                                                                                                                                                                            #         # Write to the csv file ('a' means append)
                                                                                                                                                                                                            #         of_connection = open(OUT_FILE, 'a')
                                                                                                                                                                                                            #         writer = csv.writer(of_connection)
                                                                                                                                                                                                            #         writer.writerow([loss, hyperparameters, ITERATION, run_time, best_score])
                                                                                                                                                                                                            #         of_connection.close()
                                                                                                                                                                                                                    None

                                                                                                                                                                                                                        # Dictionary with information for evaluation
                                                                                                                                                                                                                            return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,
                                                                                                                                                                                                                                                'train_time': run_time, 'status': STATUS_OK}

                                                                                                                                                                                                                        # fixed params
                                                                                                                                                                                                                        PARAMS = {
                                                                                                                                                                                                                                # https://xgboost.readthedocs.io/en/latest/parameter.html
                                                                                                                                                                                                                                'verbosity':0,

                                                                                                                                                                                                                                'booster':'gbtree', # default='gbtree',
                                                                                                                                                                                                                            #     'eta':0.1, # default=0.1, set initially high, decrease later
                                                                                                                                                                                                                            #     'max_depth':3, # default = 0.6
                                                                                                                                                                                                                            #     'min_child_weight': 1, # default = 1
                                                                                                                                                                                                                            #     'max_delta_step':1, # default=0; if cae about predicting right probability, you cannot rebalance dataset; set param to finite number, e.g., 1, to help convergence
                                                                                                                                                                                                                            #     'subsample': 0.8, # default=1
                                                                                                                                                                                                                            #     'colsample_bytree': 0.8, # default=1, can set others
                                                                                                                                                                                                                            #     'alpha':0.1, # defualt=0
                                                                                                                                                                                                                            #     'lamda':0.1, # default=0
                                                                                                                                                                                                                            #     'scale_pos_weight':np.sum(y_test==0) / np.sum(y_test==1), # default=0; if care only about AUC, balance pos/neg weights via scale_pos_weight, see: https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py, https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

                                                                                                                                                                                                                                # learning
                                                                                                                                                                                                                                'objective':'binary:logistic',
                                                                                                                                                                                                                                'eval_metric':'auc',
                                                                                                                                                                                                                            }


                                                                                                                                                                                                                        # define bayesian domain
                                                                                                                                                                                                                        space = {'eta':hp.loguniform('eta',np.log(1e-5),np.log(0.1)),
                                                                                                                                                                                                                                          'max_depth':hp.quniform('max_depth',1,50,1),
                                                                                                                                                                                                                                          'min_child_weight':hp.quniform('min_child_weight',1,100,1),
                                                                                                                                                                                                                                          'max_delta_step':hp.quniform('max_delta_step',1,100,1),
                                                                                                                                                                                                                                          'scale_pos_weight':hp.quniform('scale_pos_weight',1,100,1),
                                                                                                                                                                                                                                          'gamma':hp.uniform('gamma',0,100),
                                                                                                                                                                                                                                          'subsample':hp.uniform('subsample',0.1,1),
                                                                                                                                                                                                                                          'colsample_bytree':hp.uniform('colsample_bytree',0,1),
                                                                                                                                                                                                                                          'alpha':hp.uniform('alpha',0,1),
                                                                                                                                                                                                                                          'lamda':hp.uniform('lamda',0,1),
                                                                                                                                                                                                                                          'sampling_method':hp.choice('sampling_method',['uniform','gradient_based']),
                                                                                                                                                                                                                                          'colsample_bytree':hp.uniform('colsample_bytree',0,1),
                                                                                                                                                                                                                                          'colsample_bylevel':hp.uniform('colsampple_bylevel',0,1),
                                                                                                                                                                                                                                          'colsample_bynode':hp.uniform('colsample_bynode',0,1),
                                                                                                                                                                                                                                         }


                                                                                                                                                                                                                        INT_PARAMS = ['max_depth','min_child_weight','max_delta_step']

                                                                                                                                                                                                                        X_train_sub = X_train
                                                                                                                                                                                                                        if False:
                                                                                                                                                                                                                                X_train_sub.loc[X_train_sub['ox_status240_ROOM_AIR']==1,'last_o2_flow'] = 0
                                                                                                                                                                                                                                X_train_sub = X_train_sub.loc[:,min_feat]

                                                                                                                                                                                                                                ###################################################

                                                                                                                                                                                                                                # train xgboost and val based off optim
                                                                                                                                                                                                                                dtrain = xgb.DMatrix(X_train_sub, label=y_train)


                                                                                                                                                                                                                                max_evals = 1000
                                                                                                                                                                                                                                ITERATION = 0

                                                                                                                                                                                                                                # run hyperopt
                                                                                                                                                                                                                                trials = Trials() # record results

                                                                                                                                                                                                                                global INT_PARAMS
                                                                                                                                                                                                                                global PARAMS

                                                                                                                                                                                                                                # Run optimization
                                                                                                                                                                                                                                best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,
                                                                                                                                                                                                                                                        max_evals = max_evals)

                                                                                                                                                                                                                                if True:
                                                                                                                                                                                                                                        # save trials neatly
                                                                                                                                                                                                                                        hyperparam_opt = pd.DataFrame(trials.results).sort_values(by='loss',ascending=True)
                                                                                                                                                                                                                                            hyperparam_opt.to_csv(os.path.join(pdfp,'{}_{}hyperopt.csv'.format(model_name,method)))


                                                                                                                                                                                                                                            # display output
                                                                                                                                                                                                                                            ## w/o trials class, # print('\n best trial:\n  {}'.format(trials.best_trial))
                                                                                                                                                                                                                                            print('hyperopt results')
                                                                                                                                                                                                                                            for i in range(1): # print top n
                                                                                                                                                                                                                                                    for k,v in hyperparam_opt.iloc[i,1].items():
                                                                                                                                                                                                                                                                if isinstance(v,str):
                                                                                                                                                                                                                                                                                print('{}\t{}'.format(k,v))
                                                                                                                                                                                                                                                                else:
                                                                                                                                                                                                                                                                                print('{}\t{:.8f}'.format(k,v))
