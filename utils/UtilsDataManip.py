import pandas as pd
import numpy as np
import re
import os
from itertools import combinations

def delete_outlier(df, cols, q=.99, verbose=False):
    
    assert all([x in df.columns for x in cols]), 'Il DataFrame manca di alcune colonne.'
    
    for c in cols:
        s1 = df.shape[0]
        df = df[(df[c]<=df[c].quantile(q))|df[c].isnull()]
        s2 = df.shape[0]
        
        if verbose:
            print('La colonna {} ha {} outlier (su {}) sopra il {}% percentile'.format(c, s1-s2, s1, q*100))
            
    return df


def cast_type_col(df, diz):
    
    assert all([x in df.columns for x in diz.keys()]), 'Il DataFrame manca di alcune colonne.'
    
    for col, tipo in diz.items():
        df[col] = df[col].astype(tipo)
        
    return df


def dropna_rows_min_perc(df, minperc, verbose=False):
    nrow = df.shape[0]
    todrop = list()
    for c in df.columns:
        nummiss = df[c].isnull().sum() / nrow
        if nummiss > minperc:
            todrop.append(c)
            
    if len(todrop) > 0:
        newdf = df.dropna(subset=todrop.copy())
                
    if verbose:
        print('Abbiamo droppato {} righe su {}'.format(nrow-newdf.shape[0], nrow))
                          
    return newdf
        

def dropna_cols_min_perc(df, minperc, verbose=False):
    
    nrow = df.shape[0]
    todrop = dict()
    for c in df.columns:
        nummiss = df[c].isnull().sum() / nrow
        if nummiss > minperc:
            todrop[c] = nummiss
            
    if len(todrop) > 0:
        newdf = df.drop(todrop, axis=1)
        
    if verbose:
        for k,v in todrop.items():
            print('Abbiamo droppato la colonna {} con {:.4f}% di missing'.format(k, v*100))
            
    return newdf
        

def check_missing(df,asc=True, name='nm_missing'):
    ls_ind = list()
    ls_val = list()
    for c in df.columns:
        ls_ind.append(c)
        ls_val.append(df.loc[:,c].isnull().sum())

    return pd.Series(data=ls_val, index=ls_ind, name=name).sort_values(ascending=asc)


def check_mixed(test, out=False):
    col_mix = list()
    for c in test.columns:
        tipi = set(test.loc[:,c].dropna().apply(type).tolist())
        if len(tipi)>1:
            print(c, tipi)
            col_mix.append(c)
    if out:
        return col_mix

def are_uniques(df, subset):
    mix = check_mixed(df[subset],out=True)
    if len(mix) > 0:
        cols = [e for e in mix]
        print(cols)
        raise ValueError('Warning: mixed type columns (i.e. {}) in this DataFrame'.format(cols))
    if df[subset].drop_duplicates().shape[0] != df[subset].shape[0]:
        return False
    else:
        return True

def lookup_excel(path,skiprows,sheet_name,lookin,lookfor=':'):
    
    df = pd.read_excel(io=path, sheet_name=sheet_name, skiprows=skiprows)
    
    if lookfor!=':':
        df[lookin] = df[lookin].apply(lambda x: x.strip() if type(x)==str else x)
        return df.set_index(lookin).loc[lookfor,:]
    else:
        return df
    

def find_pk(df):
    '''
    returns primary keys in a pandas DataFrame (or throws an error if there are none)
    :param df: pandas DataFrame
    :return: list, columns acting as primary keys
    '''
    if not are_uniques(df, list(df.columns)):
        raise ValueError('There are no primary keys, perform a .drop_duplicate()')
    else:
        cols = df.columns
        for c in cols:
            if df[c].is_unique:
                return c
        for i in range(2,df.shape[1]):
            for e in combinations(cols, i):
                if are_uniques(df,list(e)):
                    return list(e)
    return list(df.columns)


def get_pattern(s):
    '''
    ottiene il pattern per numero (d) lettera (l) simbolo di una stringa
    '''
    patt = list()
    for e in s:
        if e.isdigit():
            patt.append('d')
        elif e.isalpha():
            patt.append('l')
        else:
            patt.append(e)
    return ''.join(patt)


def get_pattern_distr(df, col, ascending=False, verbose=False):
    '''
    applica get_pattern ad una colonna di un dataframe e restituisce la distribuzione, comodo per controllare cose come codici
    '''
    aux = df.copy()
    aux['pattern_count'] = aux[col].apply(lambda x: get_pattern(str(x)))
    newdf = aux['pattern_count'].value_counts().sort_values(ascending=ascending).copy()

    lng = pd.Series([len(x) for x in list(newdf.index)], index=newdf.index, name='lng')
    newdf = pd.concat([newdf,lng], axis=1).copy()
    if verbose:
        print(newdf.sort_values('lng', ascending=False))
    return newdf


def lfw(path,sheet_name,word,col,idx,skiprows=1):
    '''
    looks for words inside a columns in db excel sheets to return matching fields
    :path: filepath for excel file
    :sheet_name: sheet_name where to look for
    :word: word, or list of words, to look for (using AND)
    :col: columns where to look for words
    :idx: usually 'Nome campo', columns to use as index to return
    :returns: list of index-fields containing matching words in the sheet
    '''
    df = pd.read_excel(path,skiprows=skiprows,sheet_name=sheet_name)
    dd = df.set_index(idx).to_dict()

    if type(word) == list:
        pat = re.compile(''.join(['(?=.*{})'.format(w) for w in word]))
    else:
        pat = re.compile(str(word))
        
    ls_kv = list()
    for k,v in dd[col].items():
        if pat.search(str(v).lower()):
            ls_kv.append([k,v])
    return ls_kv


def lfw_xl(path, word, col='Descrizione', sheet_name=None, idx='Nome Campo', skiprows=1):
    '''
    applies lfw() function to an excel file, iterating sheets if not specified one
    :path: filepath for excel file
    :word: word, or list of words, to look for (using AND)b
    :col: columns where to look for words
    :sheet_name: sheet_name where to look for. If None, it will iterate every sheet
    :returns: output from lfw for every sheet (or just one sheet)
    '''
    xl = pd.ExcelFile(path)
    if sheet_name == None:
        sheet_name = xl.sheet_names
    for sheet in xl.sheet_names:
        if sheet.startswith('TBCC'):
            res = lfw(path, sheet_name=sheet, word=word, col=col, idx=idx,skiprows=skiprows)
            if len(res)!=0:
                print('\n######### '+sheet+' #########')
                for e in res:
                    print(e[0],e[1])


def check_join(df1, col1, df2, col2):
    '''
    check distinct values shared between two columns from different dataframes
    ;df1: first dataframe
    :col1: column from first dataframe
    :df2: second dataframe
    :col2: column from second dataframe
    :returns: report about number of shared values
    '''
    if type(col1) != type(col2):
        raise TypeError('Input columns must be both strings or both lists')
    if type(col1) == list and len(col1) != len(col2):
        raise ValueError('Input columns must be of the same lenght')


    if isinstance(col1, list):
        unici1 = set(df1[col1].apply(lambda x: '_'.join([str(x[i]) for i in range(len(x))]), axis=1).tolist())
        unici2 = set(df2[col2].apply(lambda x: '_'.join([str(x[i]) for i in range(len(x))]), axis=1).tolist())
    else:
        unici1 = set(df1[col1].tolist())
        unici2 = set(df2[col2].tolist())
    n_inters, n1, n2 = len(unici1.intersection(unici2)), len(unici1), len(unici2)
    return '{} in comune, il {:.2f}% del primo ({}) e il {:.2f}% del secondo ({})'.format(n_inters, n_inters/len(unici1)*100, n1, n_inters/len(unici2)*100, n2)


def try_path(folder):
    '''print a filepath creating a folder if it doesn\'t exist'''
    try:
        os.makedirs(folder)
    except FileExistsError as exc:
        pass
    return folder


def check_col_types(df, orderby='nm_missing', ascending=False):
    ''' tabella di report contenente diverse informazioni sulle variabili di un DataFrame'''
    newdf = df.dropna()
    cols = list(newdf.columns)
    cols_type = [newdf[c].dtype for c in df.columns]
    
    new = pd.DataFrame({'colonna':cols, 'tipo':cols_type})
    
    new['valori_unici'] = new.colonna.apply(lambda x: len(set(df[x].unique().tolist())))

    new = new.join(check_missing(df), on='colonna')

    new['nm_missing_perc'] = new.nm_missing / df.shape[0]

    new.sort_values(orderby, ascending=ascending, inplace=True)
    
    return new


def get_vars_corr_over(df, xvars, cap):
    '''ottengo la lista delle coppie di variabili che hanno una correlazione assoluta maggiore della soglia impostata'''
    if 0 > cap or 1 < cap:
        raise ValueError('cap deve essere compreso tra 0 ed 1')
    
    diz_corr = df[xvars].corr().to_dict()
    
    ls_high_corr = list()
    for k,v in diz_corr.items():
        for k1, v1 in diz_corr[k].items():
            if k != k1 and np.abs(v1) > cap:  # non voglio certo le correlazioni con se stesso
                ls_high_corr.append([k,k1,v1])

    # prendo elementi alterni perché voglio tenere una sola coppia tra (a,b,valore) e (b,a,valore)
    return sorted([e for i,e in enumerate(ls_high_corr) if i % 2 == 0], key=lambda x: x[2], reverse=True)


def pad_id_sx(x):
    # funzione bizantina per applicare il padding corretto ad ogni elemento del num_sinistro
    dd = {0:0,1:4,2:4,3:7}
    temp = x.split('-')
    return '-'.join([e.zfill(dd[i]) for i,e in enumerate(temp)])


def depad_id_sx(x):
    # tolgo il padding ad un id sx
    return '-'.join([c.lstrip('0') for c in x.split('-')])


def list_to_query(ls):
    '''
    prende una lista e la inserisce tra gli apici per formare una query (WHERE IN ...)
    '''
    return '(\'' + '\', \''.join([e for e in ls]) + '\')'


def depack_dataset(df, subset=None, checkvalue=True):
    '''
    Funzione che restituisce un dizionario con le colonne divise per tipologia (cont cat) e i valori unici se è cat e il range se è cont
    checkvalue serve per controllare se esistono già -999, se invece sono stati già inseriti al posto dei nan basta disattivarlo
    '''

    if subset is None:
        subset = [e for e in df.columns]

    aux = df[subset].copy()
    diz_tipo = {'cat':{}, 'cont':{}}
    for c in subset:
        # sostituisco i nan con -999, devo sapere però che tipo è prima
        tipo = aux[c].dtype
        if tipo == 'object':
            if checkvalue:
                assert '-999' not in set(aux[c].unique()), 'Non possiamo usare -999 come replace NaN, già presente'
            aux[c].replace(np.NaN, '-999', inplace=True)
            diz_tipo['cat'][c] = list(aux[aux[c]!='-999'][c].unique())

        else:
            assert aux[aux[c]==-999].shape[0] == 0, 'Non possiamo usare -999 come replace NaN, già presente'
            aux[c].replace(np.NaN, -999, inplace=True)
            diz_tipo['cont'][c] = (str(aux[aux[c]!=-999][c].min()), str(aux[aux[c]!=-999][c].max()))

    return diz_tipo