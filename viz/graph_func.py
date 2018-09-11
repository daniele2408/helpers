import numpy as np
import os
import pandas as pd
import plotly.offline as poff
import plotly.graph_objs as go
from sklearn.metrics import auc, confusion_matrix, log_loss, roc_auc_score, roc_curve, f1_score, precision_recall_curve
from tqdm import tqdm

def annotations_roc(ls_ls, ls_score, ls_soglie, fpr, tpr, thr):
    ls_ann = list()
    for metrica in ls_ls:
        if metrica == 'max_mean_acc':
            findmean = max(ls_soglie, key=lambda x: ((1-x[1][ls_score.index('miss_error_0')])+(1-x[1][ls_score.index('miss_error_1')]))/2)
            soglia, metrica_val = findmean[0], ((1-findmean[1][ls_score.index('miss_error_0')])+(1-findmean[1][ls_score.index('miss_error_0')]))/2

        elif metrica == 'max_min_acc':
            findmean = max(ls_soglie, key=lambda x: min((1-x[1][ls_score.index('miss_error_0')]),(1-x[1][ls_score.index('miss_error_1')])))
            soglia, metrica_val = findmean[0], ((1-findmean[1][ls_score.index('miss_error_0')])+(1-findmean[1][ls_score.index('miss_error_0')]))/2

        else:
            findmax = max(ls_soglie, key=lambda x: x[1][ls_score.index(metrica)])
            soglia, metrica_val = findmax[0], findmax[1][ls_score.index(metrica)]
        x = [e for e in zip(fpr, tpr, thr)][thr.index(soglia)][0]
        y = [e for e in zip(fpr, tpr, thr)][thr.index(soglia)][1]
        
        ls_ann.append({'borderwidth':.5,'bordercolor':'black','x':x,'y':y,'xref':'x','yref':'y','text':'{} = {}'.format(metrica, round(metrica_val,4))})
        
    return ls_ann

def get_predict(df, soglia):
    temp = df.copy()

    temp['predict'] = temp.score.apply(lambda x: 1 if x>=soglia else 0)
    
    return temp

def compute_scores(df, soglia, verbose=False):
    aux = df.copy()
    #df_cm = pd.crosstab(temp.true_bool, get_predict(temp, soglia)['predict'], dropna=False)
    temp = get_predict(aux, soglia)[['true_bool', 'predict']]
    TN = temp[(temp.iloc[:,0]==0)&(temp.iloc[:,1]==0)].count()[0]
    TP = temp[(temp.iloc[:,0]==1)&(temp.iloc[:,1]==1)].count()[0]
    
    FP = temp[(temp.iloc[:,0]==0)&(temp.iloc[:,1]==1)].count()[0]
    FN = temp[(temp.iloc[:,0]==1)&(temp.iloc[:,1]==0)].count()[0]
    
    if verbose:
        print('TN: ',TN,'TP: ', TP,'FP: ', FP,'FN: ', FN, sep='\n')
    # TP, TN, FP, FN = df_cm.iloc[1,1], df_cm.iloc[0,0], df_cm.iloc[0,1], df_cm.iloc[1,0]
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    recall = TP/(TP+FN)
    fpr = FP/(TN+FP)
    precision = 1 if FP == 0 else TP/(TP+FP)
    f1score = 0 if precision == 0 or recall == 0 else 2*(precision*recall/(precision+recall))
    f2score = 0 if precision == 0 or recall == 0 else 5*(precision*recall/(4*precision+recall))
    f05score = 0 if precision == 0 or recall == 0 else 1.25*(precision*recall/(0.25*precision+recall))
    miss_error_0 = FP/(TN+FP)
    miss_error_1 = FN/(FN+TP)
    scores = [accuracy, recall, precision, fpr, f1score, f2score, f05score, miss_error_0, miss_error_1]
    scores = [e if e!=np.NaN else 0 for e in scores]
    return scores

def roc_curve_annotated(ytrue, yprob, nometrue, nomeprob, ls_score, rootDir, filename):
    '''Funzione che realizza uan roc curve in .html per i risultati di un modello

    Arguments:

        ytrue, (Series): serie della variabile target
        yprob, (Series): serie di probabilità predette per il target
        nometrue, (string): nome colonna true
        nomeprob, (string): nome colonna prob
        ls_score, (list): lista di stringhe contenenti le metriche da annotare
        rootDir, (string): path della cartella dove salvare i grafici
        filename, (string): path del file .html

    Returns:
        None: l'output è un .html generato

    '''

    os.chdir(rootDir)

    fpr, tpr, thr = roc_curve(ytrue, yprob)

    fpr = [e for e in fpr if 0<=e<=1]
    tpr = [e for e in tpr if 0<=e<=1]
    thr = [e for e in thr if 0<=e<=1]

    fpr.insert(0,0)
    fpr.append(1)
    tpr.insert(0,0)
    tpr.append(1)
    thr.insert(0,1)
    thr.append(0)

    auc = roc_auc_score(ytrue, yprob)

    df_full = pd.DataFrame({nometrue: ytrue, nomeprob: yprob})

    lista_tuple_soglia = list()
    for soglia in tqdm(thr):
        if 0 <= soglia <= 1:
            lista_tuple_soglia.append((soglia,compute_scores(df_full, soglia)))
    
    roc = go.Scatter(
        x = fpr,
        y = tpr,
        text = thr,
        hoverinfo = 'text',
        name = 'roc'
    )

    annotazioni = [        dict(
                x = 0.9,
                y = 0.1,
                xref = 'x',
                yref = 'y',
                text = 'AUC = {:.2f}'.format(auc),
                showarrow = False,
                bordercolor='#c7c7c7',
                borderwidth=2,
                borderpad=4,
            )]
    annotazioni.extend(annotations_roc(
        ['accuracy', 'recall', 'precision', 'max_min_acc', 'f1score', 'f2score', 'f05score'],
        ls_score,
        lista_tuple_soglia,
        fpr,
        tpr,
        thr
        ))

    data = [roc]
    layout = go.Layout(
        xaxis = {'dtick':0.1,'title':'false positive rate'},
        yaxis = {'dtick':0.1,'title':'true positive rate'},
        
        title = 'ROC Curve APB score',
        shapes = [
            {'type': 'line',
            'x0': 0,
            'y0': 0,
            'x1': 1,
            'y1': 1,
                        'line': {
                    'color': 'orange',
                    'width': 3,
                }}
        ],
        annotations = annotazioni
    )


    fig = go.Figure(data=data, layout=layout)

    poff.plot(fig, auto_open=False, filename=os.path.join(rootDir,filename))

def real_quantili(df, n, nometrue, nomeprob):
    aux = df.copy()
    aux.sort_values(nomeprob, ascending=False, inplace=True)
    aux['indice'] = [i for i in range(aux.shape[0])]
    aux['bin'] = pd.qcut(aux.indice, n)
    response_rate = aux.groupby('bin')[nometrue].mean()
    capture_rate = aux.groupby('bin')[nometrue].sum() / aux[nometrue].sum()
    min_score = aux.groupby('bin')[nomeprob].min()
    lift_chart = pd.DataFrame({'response_rate':response_rate, 'capture_rate': capture_rate, 'min_score':min_score, 'cumulative_data_fraction':[i/100 for i in range(n,100+n,n)]})
#     lift_chart['resp_rate_cumsum'] = lift_chart.resp_rate.cumsum()
    lift_chart['cumulative_capture_rate'] = lift_chart.capture_rate.cumsum()
#     lift_chart['lift'] = lift_chart.capt_rate / pd.Series([i/100 for i in range(1,11)])#(lift_chart.bin / 100)
    lift_chart['cumulative_lift'] = lift_chart.cumulative_capture_rate / (lift_chart.cumulative_data_fraction)
    
    return lift_chart

def lift_chart(df, nometrue, nomeprob, rootDir, filename):
    
    df_lift_sel = real_quantili(df, 10, nometrue, nomeprob)
    
    x = df_lift_sel.cumulative_data_fraction

    trace = go.Scatter(
        x = x,
        y = df_lift_sel.cumulative_capture_rate,
        name = 'Capture rate cumulato'
    )

    trace2 = go.Scatter(
        x = x,
        y = df_lift_sel.cumulative_lift,
        name = 'Lift cumulata',
        yaxis='y2'
    )

    trace3 = go.Bar(
        x = x,
        y = df_lift_sel.response_rate,
        name = 'Response rate',
        opacity = 0.5,
    )

    trace4 = go.Bar(
        x = x,
        y = df_lift_sel.capture_rate,
        name = 'Capture rate',
        opacity = 0.5,
    )

    # trace5 = go.Scatter(
    #     x = x,
    #     y = df_lift_sel.cumulative_response_rate,
    #     name = 'Response rate cumulato'
    # )

    trace6 = go.Scatter(
        x = x,
        y = df_lift_sel.min_score,
        name = 'min_score',
        yaxis = 'y2'
    )

    data = [trace, trace2, trace3, trace4]
    layout = go.Layout(
        title = 'Lift Chart',
        xaxis = {
            'title':'percentili',
            'tickformat':"%",
            'range':[0,1.01],
            'dtick':0.1,
            'showgrid':True,
            
        },
        yaxis = {
    #         'title':'percentuale VARRESP cumulata',
            'tickformat':"%",
            'range':[0,1.03],
            'dtick':0.1,
            'showgrid':False,
            
        },
        yaxis2 = {
            'title':'Lift',
        "overlaying":'y',
        "side":'right',
        'range':[0,df_lift_sel.cumulative_lift.max()*1.1],
        "showline":False,
        "showgrid":False,
        "zeroline":False
        },
        showlegend=True,
        # shapes= genera_rect({'red':0.2,'orange':0.4,'yellow':0.8,'green':1}, df_lift, 'quantili', opc=0.2)
    )
    fig = go.Figure(data=data, layout=layout)
    poff.iplot(fig)
    # plot(fig, filename=r'C:/Users/LEI00020/gitfolder/antifrodereboot/Sources/prjDeploy/Deployment/Deployment/output/liftchart_new_grid.html')
    poff.plot(fig, auto_open=False, filename=os.path.join(rootDir, filename))