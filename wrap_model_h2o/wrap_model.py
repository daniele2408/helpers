import h2o
import os
import pandas as pd
import pickle
from datetime import datetime
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from sklearn.metrics import roc_curve

def generate_doc(dfPath, seed, predittori, target, column_types, model_id, ratio_train, ratio_scale, time_elapsed, trial, iperparam):
    '''
    Funzione che genera un testo di sintesi sul modello.
    '''

    print(iperparam)

    diz_str = {
        'model_id':model_id,
        'seed':seed,
        'dfPath':dfPath,
        'trperc':ratio_train*100,
        'scperc':ratio_scale*100,
        'mins':time_elapsed,
        'target':target,
        'trial':'sì' if trial else 'no',
        'ls_pred':'\n'.join([str(k)+' (tipo: '+column_types[k]+')' for k in predittori]),
        'iperparam':'\n'.join([str(k)+': '+ str(v) for k,v in iperparam.items() if k not in {'calibration_frame'}])
            
            }
            
    doc = '''
    ################################################
    Documento generato il {:%Y-%m-%d %H:%M}
    ################################################

    Il modello {model_id} è stato applicato al dataset in {dfPath} (trial: {trial}) dedicando {trperc}% al train e {scperc}% allo scaling (seed={seed}), impiegando {mins:.1f} minuti a girare.

    Target: {target}

    Predittori:

    {ls_pred}

    Parametri:

    {iperparam}

    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    '''.format(datetime.now(),**diz_str)

    return doc


def get_info(model, mdl_perf, df_test_preds, target, prob_name):
    '''
    Metodo che recupera le informazioni di un gbm ricreando test e train (con stesso seed! e le ratio di split) e applicandovi il modello.

    :return tupla di dict contenenti parametri, metriche e le serie per costruire la ROC curve.
    '''

    params = {k:v['actual'] for k,v in model.params.items()}

    # salvo le metriche su test
    print('Salvo le metriche')

    auc = mdl_perf.auc()
    logloss = mdl_perf.logloss()
    mean_per_class_error = mdl_perf.mean_per_class_error()
    cm = mdl_perf.confusion_matrix().to_list()
    gain_lift = mdl_perf.gains_lift().as_data_frame()

    print(df_test_preds[target].value_counts(dropna=False))

    # dati per creare la roc e poterla confrontare con altri modelli
    print('Calcolo la roc')
    fpr, tpr, thr = roc_curve(df_test_preds[target], df_test_preds[prob_name])

    diz_metriche = {
        'auc':auc,
        'logloss':logloss,
        'mean_per_class_error':mean_per_class_error,
        'cm':cm,
        'gain_lift':gain_lift,
        'varimp':model.varimp(use_pandas=True)
        }

    dati_pred = {
        'test_pred':df_test_preds,
        'fpr':fpr,
        'tpr':tpr,
        'thr':thr
    }

    return params, diz_metriche, dati_pred


def split_h2o_and_factor(df, target, myseed, column_types, ratio_train, ratio_scale=0):
    '''
    Funzione che crea test train split per un mdoello h2o

    :param df: DataFrame dataset da splittare
    :target: stringa variabile target
    :param myseed: int, random seed per lo split
    :param column_types: dict, nome colonna -> tipo colonna
    :ratio_train: float, % di dataset per il train
    :ratio_scale: float, % di dataset per il validation set

    :returns tupla di DataFrame dal dataset splittato
    '''
    df_h2o = h2o.H2OFrame(df, column_types=column_types)
    for col, type_ in df_h2o.types.items():
        if type_ == 'enum':
            df_h2o[col] = df_h2o[col].asfactor()
    df_h2o[target] = df_h2o[target].asfactor()

    if ratio_scale > 0:
        res = df_h2o.split_frame(ratios=[ratio_train, ratio_scale], seed = myseed)
    else:
        res = df_h2o.split_frame(ratios=[ratio_train], seed = myseed)

    return res # df_h2o.split_frame(ratios=[ratio_train, ratio_scale], seed = myseed) if ratio_scale > 0 else df_h2o.split_frame(ratios=[ratio_train], seed = myseed)


def train_model_GBM(
    dfPath,
    predittori, 
    target,
    column_types,
    model_id,
    rootDir,
    trial=False,
    ratio_train=.7,
    ratio_scale=.15,
    nthreads=3,
    port=54321,
    myseed=12345,
    prob_name='cal_p1',
    shutfirst=False,
    verbose=False,
    startserver=True,
    keepserver=True,
    **iperparam):
    '''
    Funzione che prende un path di un .csv contenente un dataset preprocessato e altre informazioni per poter applicare un GBM h2o.

    :param dfPath: stringa col path del dataset
    :param predittori: lista delle x_vars
    :param target: stringa variabile target
    :param column_types: dizionario col tipo di ogni predittore
    :param model_id: stringa nome del modello, serve anche per creare la cartella
    :param rootDir: path cartella root
    :param trial: booleano per non dover usare tutto il dataset ma solo 10k righe
    :param ratio_train: float, % per il train set
    :param ratio_scale: float, % per il validation set
    :param nthreads: intero, numero thread per server h2o
    :param port: intero, porta per server h2o
    :param myseed: intero, random seed per lo split
    :param prob_name: stringa, nome della colonna delle prob_pred
    :param shutfirst:
    :param verbose:
    :param startserver:
    :param startserver:
    :param keepserver:
    :param **iperparam: parametri per l'istanza di H2OGradientBoostingEstimator
    
    :return gbm: modello gbm
    :return testfull: test set + predict test, in formato DataFrame
    :return trainfull: train set + predict train, in formato DataFrame
    :return perfs: oggetto per richiamare i metodi per le perfomrance (logloss, auc, ecc...)
    '''

    if trial:
        df = pd.read_csv(dfPath, nrows=10000)
    else:
        df = pd.read_csv(dfPath)

    model_id = model_id + '_' + datetime.strftime(datetime.now(), format="%Y%m%d%H%M")  # il model_id avrà il timestamp incorporato

    pathModel = os.path.join(rootDir, 'modello_{}'.format(model_id))
    pathPredsData = os.path.join(pathModel, 'predittori.p')
    pathColtypesData = os.path.join(pathModel, 'coltypes.p')

    pathData = os.path.join(pathModel, 'data', 'dataset.csv')
    pathDataTrain = os.path.join(pathModel, 'data', 'train')
    trainFilePath = os.path.join(pathModel, 'dataset_train_preds.csv')
    pathDataTest = os.path.join(pathModel, 'data', 'test')
    testFilePath = os.path.join(pathDataTest, 'dataset_test_preds.csv')

    # paramsPath = os.path.join(pathModel, 'parametri.p')
    diz_metrichePath = os.path.join(pathModel, 'metriche.p')
    dati_predPath = os.path.join(pathDataTest, 'dati_predPath.p')

    os.makedirs(pathDataTest, exist_ok=True)
    os.makedirs(pathDataTrain, exist_ok=True)

    # mi accerto che ci sia almeno un 15% del dataset per il test
    ratio_test = 1 - (ratio_scale + ratio_train)
    assert ratio_test >= .15, "Percentuale dataset per test troppo bassa ({:.2f})".format(ratio_test)

    try:
        # inizializzo il server
        if shutfirst:
            h2o.cluster().shutdown()

        if startserver:
            os.environ['http_proxy'] = ''
            os.environ['https_proxy'] = ''
            h2o.init(nthreads=nthreads, max_mem_size = '28G', port=port)

        # creo train test e scaling con una funzione per poterla richiamare anche poi
        if verbose:
            print('Faccio lo split del dataset')
        if ratio_scale > 0:
            train, test, scaling = split_h2o_and_factor(df, target, myseed, column_types, ratio_train, ratio_scale)
        else:
            train, test = split_h2o_and_factor(df, target, myseed, column_types, ratio_train)

        # creo modello e lo traino
        iperparam['model_id'] = model_id
        if ratio_scale > 0:
            iperparam['calibrate_model'] = True
            iperparam['calibration_frame'] = scaling

        if verbose:
            print('Lancio il modello')
        t0 = datetime.now()
        gbm = H2OGradientBoostingEstimator(**iperparam)
        gbm.train(x=predittori, y=target, training_frame=train)
        t1 = datetime.now()
        time_elapsed = (t1-t0).seconds / 60
        print('Il modello ci ha impiegato {:.1f} minuti'.format(time_elapsed))

        if verbose:
            print('Calcolo le performance')
        # tengo da parte le performance per darle al return
        perfs = gbm.model_performance(test)

        if verbose:
            print('Salvo il modello e i predittori')
        # salvo il modello
        h2o.save_model(gbm, path=pathModel, force=True)

        # salvo predittori e coltypes
        pickle.dump(predittori, open(pathPredsData, 'wb'))
        pickle.dump(column_types, open(pathColtypesData, 'wb'))

        if verbose:
            print('Aggiungo le predizioni ai set')
        # unisco test e predizioni con probabilità
        testpreds = gbm.predict(test)
        testfullH2O = test.cbind(testpreds)
        testfull = testfullH2O.as_data_frame()

        testfull = testfull[testfull[target].isin([0,1])].dropna(subset=[target, prob_name]).copy()
        if verbose:
            print('Shape del testfull: {}'.format(testfull.shape))

        # alternativa
        # testpreds = gbm.predict(test)
        # testpreds_df = testpreds.as_data_frame()
        # test_df = test.as_data_frame()
        # testfull = pd.concat([testpreds_df, test_df], axis=1)

        # unisco train e predizioni con probabilità
        trainpreds = gbm.predict(train)
        trainfullH2O = train.cbind(trainpreds)
        trainfull = trainfullH2O.as_data_frame()

        if verbose:
            print('Salvo i set integrati con le predizioni')
        # salvo test e train
        testfull.to_csv(testFilePath, index=False)
        trainfull.to_csv(trainFilePath, index=False)

        # salvo anche il dataset
        # if not os.path.exists(pathData):
        #     df.to_csv(pathData, index=False)

        if verbose:
            print('Genero un documento d\'informazione')

        doc = generate_doc(dfPath, myseed, predittori, target, column_types, model_id, ratio_train, ratio_scale, time_elapsed, trial, iperparam)

        with open(os.path.join(pathModel, 'info.txt'), 'a') as f:
            f.write(doc)

        if verbose:
            print('Ricavo informazioni')

        params, diz_metriche, dati_pred = get_info(gbm, perfs, testfull, target, prob_name)
        
        if verbose:
            print('Salvo le informazioni')

        # pickle.dump(params, open(paramsPath, 'wb'))
        pickle.dump(diz_metriche, open(diz_metrichePath, 'wb'))
        pickle.dump(dati_pred, open(dati_predPath, 'wb'))


    except Exception as err:
        print(err)
        raise err
    finally:
        if not keepserver:
            h2o.remove_all()
            h2o.cluster().shutdown()

    return gbm, testfull, trainfull, perfs, params, diz_metriche, dati_pred


class ModelWrap:

    '''
    Classe per gestire i modelli creati con h2o e tutte le informazioni connesse in modo da poterli confrontare tra di loro in modo agevole

    :param rootDir: percorso cartella principale
    :param datasetDir: percorso di partenza dataset usato
    :param model_id: stringa, nome mdoello da estrarre
    :param target: stringa, nome variabile target
    :param prob_name: stringa, nome variabile prob_pred
    '''

    def __init__(self, rootDir, datasetDir, model_id, target, prob_name='cal_p1'):

        pathModel = os.path.join(rootDir, 'modello_{}'.format(model_id))
        pathPredsData = os.path.join(pathModel, 'predittori.p')
        pathColtypesData = os.path.join(pathModel, 'coltypes.p')

        pathData = os.path.join(pathModel, 'data', 'dataset.csv')    
        pathDataTrain = os.path.join(pathModel, 'data', 'train')
        trainFilePath = os.path.join(pathDataTrain, 'dataset_train_preds.csv')
        pathDataTest = os.path.join(pathModel, 'data', 'test')
        testFilePath = os.path.join(pathDataTest, 'dataset_test_preds.csv')

        model = os.path.join(pathModel, model_id)

        # paramsPath = os.path.join(pathModel, 'parametri.p')
        diz_metrichePath = os.path.join(pathModel, 'metriche.p')
        dati_predPath = os.path.join(pathDataTest, 'dati_predPath.p')

        self.model = h2o.load_model(model)
        self.predittori = pickle.load(open(pathPredsData, 'rb'))
        self.coltype = pickle.load(open(pathColtypesData, 'rb'))
        self.target = target

        self.dataset = pd.read_csv(datasetDir)

        self.test = pd.read_csv(testFilePath)
        # self.train = pd.read_csv(trainFilePath)
        self.prob_name = prob_name

        # self.params = pickle.load(open(paramsPath, 'rb'))
        self.diz_metriche = pickle.load(open(diz_metrichePath, 'rb'))
        self.dati_pred = pickle.load(open(dati_predPath, 'rb'))

        
        
    def get_info(self, ratio_train, ratio_scale, myseed):
        '''
        Metodo che recupera le informazioni di un gbm ricreando test e train (con stesso seed! e le ratio di split) e applicandovi il modello.

        :return tupla di dict contenenti parametri, metriche e le serie per costruire la ROC curve.
        '''
        # salvo gli iperparametri usati
        params = self.model.params

        # ricavo il test set
        print('Faccio lo split')
        if ratio_scale > 0:
            train, validation, test = split_h2o_and_factor(self.dataset, self.target, myseed, self.coltype, ratio_train, ratio_scale)
        else:
            train, test = split_h2o_and_factor(self.dataset, self.target, myseed, self.coltype, ratio_train, ratio_scale)

        print(test)
        print('Ricavo la predizione sul test in formato DataFrame')
        # ricavo le test_preds
        testpreds = self.model.predict(test)
        testfullH2O = test.cbind(testpreds)
        df_test_preds = testfullH2O.as_data_frame()

        print(df_test_preds.head())

        # salvo le metriche su test
        print('Salvo le metriche')
        mdl_perf = self.model.model_performance(test)

        auc = mdl_perf.auc()
        logloss = mdl_perf.logloss()
        mean_per_class_error = mdl_perf.mean_per_class_error()
        cm = mdl_perf.confusion_matrix().to_list()
        gain_lift = mdl_perf.gains_lift().as_data_frame()

        print(df_test_preds[self.target].value_counts(dropna=False))

        # dati per creare la roc e poterla confrontare con altri modelli
        print('Calcolo la roc')
        fpr, tpr, thr = roc_curve(df_test_preds[self.target], df_test_preds[self.prob_name])

        diz_metriche = {
            'auc':auc,
            'logloss':logloss,
            'mean_per_class_error':mean_per_class_error,
            'cm':cm,
            'gain_lift':gain_lift,
            'varimp':self.model.varimp(use_pandas=True)
            }

        dati_pred = {
            'test_pred':df_test_preds,
            'fpr':fpr,
            'tpr':tpr,
            'thr':thr
        }

        return params, diz_metriche, dati_pred