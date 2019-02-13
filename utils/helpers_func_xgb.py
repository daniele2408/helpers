import os
import pandas as pd
import plotly.offline as poff
import plotly.graph_objs as go
import xgboost as xgb
from functools import partial
from helpers.viz.graph_func import plot_this
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


def shap_transform_scale(shap_values, expected_value, model_prediction):
    #Compute the transformed base value, which consists in applying the logit function to the base value    
    expected_value_transformed = expit(expected_value)
    
    #Computing the original_explanation_distance to construct the distance_coefficient later on
    original_explanation_distance = sum(shap_values)

    #Computing the distance between the model_prediction and the transformed base_value
#     distance_to_explain = abs(model_prediction - expected_value_transformed)
    distance_to_explain = model_prediction - expected_value_transformed

    #The distance_coefficient is the ratio between both distances which will be used later on
    distance_coefficient = original_explanation_distance / distance_to_explain

    #Transforming the original shapley values to the new scale
    shap_values_transformed = shap_values / distance_coefficient

    return shap_values_transformed, expected_value_transformed, original_explanation_distance, model_prediction
def grid_learn(alg, X_train, y_train, n_est, n_jobs):
    
    param_test = {
        # 'n_estimators':[1, 2, 3],
        'learning_rate':[0.01, 0.1, 0.5, 1]
    }

    gsearch = GridSearchCV(
        estimator=alg,
        param_grid=param_test,
        scoring='neg_log_loss',
        n_jobs=n_jobs,
        iid=False,
        cv=5)

    gsearch.fit(X_train, y_train)

    return sorted(gsearch.cv_results_, key=lambda x: x[1]), gsearch.best_params_, gsearch.best_score_

def get_n_est(alg, dtrain, predictors, target, cv_folds=5, early_stopping_rounds=50):
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics=['auc', 'logloss'], early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        n_estimators=cvresult.shape[0]

        return n_estimators, cvresult

def modelfit(alg, dtrain, n_est, predictors, target, useTrainCV=True, foldObj=None, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(
                xgb_param, xgtrain, 
                num_boost_round=alg.get_params()['n_estimators'],
                nfold=cv_folds,
                folds=foldObj,
                metrics=['auc', 'logloss'],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    else:
        alg.set_params(n_estimators=n_est, early_stopping_rounds=early_stopping_rounds)
 
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
    #Predict training set:
#     dtrain_predictions = alg.predict(dtrain[predictors])
#     dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    return alg


def objective(space, n_est, X_train, y_train, X_test, y_test, loss='logloss'):
    
    clf = XGBClassifier(n_estimators = n_est,
                        max_depth = space['max_depth'],
                        min_child_weight = space['min_child_weight'],
                        subsample = space['subsample'],
                        learning_rate = 0.01, # space['learning_rate'],
                        gamma = space['gamma'],
                        n_jobs=4,
                        colsample_bytree = space['colsample_bytree'],
                        objective='binary:logistic'
                       )
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    print(clf.get_params)

    clf.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        eval_metric= 'logloss')
    
    pred = clf.predict_proba(X_test)[:,1]
    
    if loss=='logloss':
        loss_f = log_loss(y_test, pred)
    elif loss=='auc':
        loss_f = (1-roc_auc_score(y_test, pred))
    else:
        raise 'Errore, selezionare una funzione da minimizzare'
    
    return {'loss':loss_f, 'status':STATUS_OK}


def apply_hyperopt(space, n_est, X_train, y_train, X_test, y_test, max_evals=20):

    trials = Trials()

    best = fmin(
        fn=partial(objective, n_est=n_est, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test),
        space=space,
        algo=partial(tpe.suggest, n_startup_jobs=10),
        max_evals=max_evals,
        trials=trials
    )

    best_trial_vals, best_trial_loss, trials_results, trials_raw = trials.best_trial['misc']['vals'], trials.best_trial['result']['loss'], trials.results, trials.trials


    return best_trial_vals, best_trial_loss, trials_results, trials_raw


def graph_cv(df, train_cols, test_cols, filename):
    
    x = df.shape[0]
    
    go_train = [
        go.Scatter(
            x = x,
            y = df[traincol],
            name = traincol,
        
        ) for traincol in train_cols
    ]
    
    go_test = [
        go.Scatter(
        x = x,
        y = df[testcol],
        name = testcol,
        
        ) for testcol in test_cols
    ]
    
    data = list()
    data.extend(go_train)
    data.extend(go_test)
    layout = go.Layout(yaxis={'range':(0,1)})
    fig = go.Figure(data=data, layout=layout)
    
    poff.plot(fig, filename=filename, auto_open=False)

def plot_shap(gg, mm, row, datasetOBJ, modelOBJ):
    data = pd.read_csv(r'C:\Users\LEI00020\Desktop\pipelineAPB\unico\collaudo\model_apb_winner_dataset_cervedcrif\shap\data_2018{}{}.csv'.format(mm,gg))
    sx = data.iloc[:,-1]
    data = data.iloc[:,:-1]
    shap = pd.read_csv(r'C:\Users\LEI00020\Desktop\pipelineAPB\unico\collaudo\model_apb_winner_dataset_cervedcrif\shap\shap_2018{}{}.csv'.format(mm,gg))

    dizle = datasetOBJ.diz_le

    for k in dizle.keys():
        # data[k] = data[k].astype(int)  # a volte non sono in int...
        data[k] = data[k].apply(lambda x: dizle[k].inverse_transform(x))
    data = round(data,2)
    data['valorecommercialeveic'] /= 100

    lookforsx=False
    if not lookforsx:
        nrow = row
    else:
        id_sinistro = '1-8001-2018-190785'
        nrow = sx.tolist().index(id_sinistro)
        print('Il sx {} è alla riga {}'.format(id_sinistro, nrow))
    ev = shap.base_value.unique().tolist()[0]
    score = shap.score.tolist()[nrow]
    sp_tr, ev_tr, orig_dist, model_pred = shap_transform_scale(shap.iloc[nrow,:-2].tolist(), ev, score)

    cop = [e for e in zip(sp_tr, data.columns)]

    neg = sorted([e for e in cop if e[0] > 0], reverse=True)
    pos = sorted([e for e in cop if e[0] < 0], reverse=True)

    tot = pos + neg

    num = [e[0] for e in tot]
    val = [e[1] for e in tot]

    dftemp = pd.DataFrame({'num':num, 'val':val})
    dftemp['segno_pos'] = dftemp.num.apply(lambda x: True if x > 0 else False)

    dftemp['top'] = dftemp.apply(lambda x: -1*x.num if not x.segno_pos else x.num, axis=1)
    dftemp['bottom'] = dftemp.apply(lambda x: model_pred if not x.segno_pos else model_pred-x.num, axis=1)
    dftemp['data_val'] = [data[e].iloc[nrow] for e in dftemp.val.tolist()]
    hidena=True
    if hidena:
        dftemp = dftemp[(dftemp.data_val!='-999')&(dftemp.data_val!=-999.0)].copy()

    idsx = sx.iloc[nrow]
    saveplot = os.path.join(modelOBJ.model_plot_path, 'shap_sx_{}.html'.format(idsx))
    print('Salvo in {}'.format(saveplot))
    plot_this(dftemp.bottom, dftemp.top, dftemp.segno_pos, dftemp.val, dftemp.data_val,model_pred, nrow, gg, mm, idsx, filename=saveplot)

def plot_shap_test(gg, mm, row, datasetOBJ, modelOBJ):
    data = pd.read_csv(r'C:\Users\LEI00020\Desktop\pipelineAPB\unico\collaudo\model_apb_winner_dataset_cervedcrif\shap\data_2018{}{}.csv'.format(mm,gg))
    sx = data.iloc[:,-1]
    data = data.iloc[:,:-1]
    shap = pd.read_csv(r'C:\Users\LEI00020\Desktop\pipelineAPB\unico\collaudo\model_apb_winner_dataset_cervedcrif\shap\shap_2018{}{}.csv'.format(mm,gg))

    dizle = datasetOBJ.diz_le

    for k in dizle.keys():
        # data[k] = data[k].astype(int)  # a volte non sono in int...
        data[k] = data[k].apply(lambda x: dizle[k].inverse_transform(x))
    data = round(data,2)
    data['valorecommercialeveic'] /= 100

    lookforsx=False
    if not lookforsx:
        nrow = row
    else:
        id_sinistro = '1-8001-2018-190785'
        nrow = sx.tolist().index(id_sinistro)
        print('Il sx {} è alla riga {}'.format(id_sinistro, nrow))
    ev = shap.base_value.unique().tolist()[0]
    score = shap.score.tolist()[nrow]
    sp_tr, ev_tr, orig_dist, model_pred = shap_transform_scale(shap.iloc[nrow,:-2].tolist(), ev, score)

    cop = [e for e in zip(sp_tr, data.columns)]

    neg = sorted([e for e in cop if e[0] > 0], reverse=True)
    pos = sorted([e for e in cop if e[0] < 0], reverse=True)

    tot = pos + neg

    num = [e[0] for e in tot]
    val = [e[1] for e in tot]

    dftemp = pd.DataFrame({'num':num, 'val':val})
    dftemp['segno_pos'] = dftemp.num.apply(lambda x: True if x > 0 else False)

    dftemp['top'] = dftemp.apply(lambda x: -1*x.num if not x.segno_pos else x.num, axis=1)
    dftemp['bottom'] = dftemp.apply(lambda x: model_pred if not x.segno_pos else model_pred-x.num, axis=1)
    dftemp['data_val'] = [data[e].iloc[nrow] for e in dftemp.val.tolist()]
    hidena=True
    if hidena:
        dftemp = dftemp[(dftemp.data_val!='-999')&(dftemp.data_val!=-999.0)].copy()

    idsx = sx.iloc[nrow]
    saveplot = os.path.join(modelOBJ.model_plot_path, 'shap_sx_{}.html'.format(idsx))
    print('Salvo in {}'.format(saveplot))
    plot_this(dftemp.bottom, dftemp.top, dftemp.segno_pos, dftemp.val, dftemp.data_val,model_pred, nrow, gg, mm, idsx, filename='prova.html', iftest=True, raw=tot)


if __name__=='__main__':

    from static import Model_winner as model_
    from static import Dataset_cervedcrif as dataset_

    plot_shap('17', '08', 13, dataset_, model_)