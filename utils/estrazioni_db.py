import numpy as np
import pandas as pd

from itertools import zip_longest
from sqlalchemy import create_engine
from helpers.utils.UtilsDataManip import list_to_query

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


diz_oldnew = {
    'crif_anagrafiche_cdc':'AN_CN_AT_000140',
    'crif_cbs_anagrafiche_almeno_una_cdc':'AN_SC_000070',
    'crif_propensione_prestiti_personali':'zone_pp',
    'crif_contratti_in_default_t0':'CN_DF_DS_000020',
    'crif_limite_utilizzo_cdc':'CN_AT_000540_M24',
    'crif_tan_medio_pp':'CN_AT_000082_M24',
    'crif_cbs_anagrafiche':'AN_SC_000010',
    'crif_anagrafiche_attive':'AN_CN_AT_000020',
    'crif_propensione_prestiti_finalizzati':'zone_pf',
    'crif_cbs_anagrafiche_almeno_un_pp':'AN_SC_000030',
    'crif_propensione_cdc':'zone_ca',
    'crif_anagrafiche_mi':'AN_CN_AT_000040',
    'crif_propensione_mutui':'zone_mu',
    'crif_stranieri':'AN_000280',
    'crif_score_affidabilita_seg2':'zone_risk2',
    'crif_valore_reddito':'REDDITO_ME',
    'crif_score_affidabilita_seg1':'zone_risk1',
    'crif_imm_fascia':'Imm_Fascia',
    'crif_imm_descr_tip_prev':'Imm_Descr_tip_prev',
    'crif_imm_compr_min':'Imm_Compr_min',
    'crif_imm_compr_max':'Imm_Compr_max',
    'crif_imm_loc_min':'Imm_Loc_min',
    'crif_imm_loc_max':'Imm_Loc_max',
    'crif_imm_categoria':'imm_categoria'
    }

def query_indirizzi(df, db, geokey='microzona', chiave_indirizzo=None, feat_indirizzo=None):
    '''
    Funzione che restituisce una pd.DataFrame con indirizzo e microcella da usare come chiave per ulteriori query
    '''
    aux = df.copy()
    aux.fillna('', inplace=True)
    aux[[feat_indirizzo['CAP']]] = aux[feat_indirizzo['CAP']].apply(lambda x: int(x) if '.' in str(x) else x)
    if chiave_indirizzo is None:
        assert feat_indirizzo, 'Specificare le colonne con le quali costruire l\'indirizzo'

    if chiave_indirizzo is None:
        aux['indirizzo_completo'] = aux.apply(lambda x: ','.join(
                                                                            [
                                                                                str(x[feat_indirizzo['INDIRIZZO']])+' '+str(x[feat_indirizzo['CIVICO']]),
                                                                                str(x[feat_indirizzo['CITTA']]),
                                                                                str(x[feat_indirizzo['PROVINCIA']]),
                                                                                str(x[feat_indirizzo['CAP']]).zfill(5),
                                                                                'ITALIA'
                                                                            ]
        ), axis=1)

    

    aux['indirizzo_completo'] = aux['indirizzo_completo'].apply(lambda x: x.replace("\'", "\'\'"))

    ls_indirizzi = aux.indirizzo_completo.unique()
    # prendo le microzone
    qry_geo = '''
    SELECT address, {}
    FROM public.id_cerved_crif
    WHERE address in {}
    '''

    ls_df_microzona = list()
    for chunk in grouper(ls_indirizzi, 1000, ''):
        tempqry = qry_geo.format(geokey, list_to_query(chunk))
        dfchunk = pd.read_sql(tempqry, db)
        ls_df_microzona.append(dfchunk)

    indirizzi_full = pd.concat(ls_df_microzona, axis=0, ignore_index=True)
    
    return indirizzi_full



def query_crif(df, select, db, chiave_indirizzo=None, feat_indirizzo=None):
    '''
    Funzione per recuperare info crif a partire da un indirizzo standard in un DataFrame
    '''

    aux = df.copy()
    aux.fillna('', inplace=True)
    aux[[feat_indirizzo['CAP']]] = aux[feat_indirizzo['CAP']].apply(lambda x: int(x) if '.' in str(x) else x)
    if chiave_indirizzo is None:
        assert feat_indirizzo, 'Specificare le colonne con le quali costruire l\'indirizzo'

    if chiave_indirizzo is None:
        aux['indirizzo_completo'] = aux.apply(lambda x: ','.join(
                                                                            [
                                                                                str(x[feat_indirizzo['INDIRIZZO']])+' '+str(x[feat_indirizzo['CIVICO']]),
                                                                                str(x[feat_indirizzo['CITTA']]),
                                                                                str(x[feat_indirizzo['PROVINCIA']]),
                                                                                str(x[feat_indirizzo['CAP']]).zfill(5),
                                                                                'ITALIA'
                                                                            ]
        ), axis=1)

    

    aux['indirizzo_completo'] = aux['indirizzo_completo'].apply(lambda x: x.replace("\'", "\'\'"))

    ls_indirizzi = aux.indirizzo_completo.unique()
    # prendo le microzone
    qry_geo = '''
    SELECT address, microzona
    FROM public.id_cerved_crif
    WHERE address in {}
    '''

    qry_crif = '''
    select sezrid09, {}
    from public.crif_data
    where sezrid09 in {}
    '''

    ls_df_microzona = list()
    for chunk in grouper(ls_indirizzi, 1000, ''):
        tempqry = qry_geo.format(list_to_query(chunk))
        dfchunk = pd.read_sql(tempqry, db)
        ls_df_microzona.append(dfchunk)

    indirizzi_full = pd.concat(ls_df_microzona, axis=0, ignore_index=True)
    
    ls_microzona = [e for e in indirizzi_full.microzona.unique().tolist() if type(e)==str]

    ls_df_crif= list()
    for chunk in grouper(ls_microzona, 100, ''):
        tempqry = qry_crif.format(', '.join(['"'+e+'"' for e in select]), list_to_query(chunk))
        dfchunk = pd.read_sql(tempqry, db)
        ls_df_crif.append(dfchunk)

    return pd.concat(ls_df_crif, axis=0, ignore_index=True)


def query_cerved(df, geokey, idkey, select, table, db, feat_indirizzo=None):
    '''
    Funzione per recuperare info crif a partire da un indirizzo standard in un DataFrame
    '''

    ls_microzona = df[geokey].dropna().unique().tolist()

    print('Abbiamo {} celle da cercare'.format(len(ls_microzona)))

    qry_cerved = '''
    select {p[chiave]}, {p[feat_select]}
    from {p[table]}
    where {p[chiave]} in {p[lista_chiavi]}
    '''

    ls_df_crif= list()
    for chunk in grouper(ls_microzona, 1000, ''):
        ls_qry = list_to_query(list(chunk))
        diz_qry = {
            'chiave':'"'+idkey+'"',
            'feat_select':', '.join(['"'+e+'"' for e in select]),
            'table':table,
            'lista_chiavi':ls_qry
        }
        tempqry = qry_cerved.format(p=diz_qry)
        dfchunk = pd.read_sql(tempqry, db)
        ls_df_crif.append(dfchunk)

    return pd.concat(ls_df_crif, axis=0, ignore_index=True)


def geo_loc_address(df, db, chiave_indirizzo=None, feat_indirizzo=None, chunksize=1000):

    aux = df.copy()
    aux.fillna('', inplace=True)
    aux[[feat_indirizzo['CAP']]] = aux[feat_indirizzo['CAP']].apply(lambda x: int(x) if '.' in str(x) else x)
    if chiave_indirizzo is None:
        assert feat_indirizzo, 'Specificare le colonne con le quali costruire l\'indirizzo'

    if chiave_indirizzo is None:
        aux['indirizzo_completo'] = aux.apply(lambda x: ','.join(
                                                                            [
                                                                                str(x[feat_indirizzo['INDIRIZZO']])+' '+str(x[feat_indirizzo['CIVICO']]),
                                                                                str(x[feat_indirizzo['CITTA']]),
                                                                                str(x[feat_indirizzo['PROVINCIA']]),
                                                                                str(x[feat_indirizzo['CAP']]).zfill(5),
                                                                                'ITALIA'
                                                                            ]
        ), axis=1)

    aux['indirizzo_completo'] = aux['indirizzo_completo'].apply(lambda x: x.replace("\'", "\'\'"))

    ls_indirizzi = aux.indirizzo_completo.unique()

    qry_geo = '''
    SELECT address, lat, lng, source_precision
    FROM public.addresses
    WHERE address in {}
    '''

    ls_df_coord = list()
    for chunk in grouper(ls_indirizzi, chunksize, ''):
        tempqry = qry_geo.format(list_to_query(chunk))
        dfchunk = pd.read_sql(tempqry, db)
        ls_df_coord.append(dfchunk)

    res = pd.concat(ls_df_coord, axis=0, ignore_index=True)
    
    return res

def iterate_over_list(db, ls_valori, chunksize=None, table=None, select=None, filteriter=None, qry=None, verbose=True):
    '''
    Funzione per iterare grosse liste quando si estrae con un WHERE IN

    Parameters:
        * db: oggetto connessione db
        * ls_valori: lista dei valori usati per filtrare
        * table: stringa tabella
        * select: lista stringhe campi selezionati
        * filteriter: stringa campo dove si applica WHERE $campo IN
        * chunksize: numero di righe da prendere a iterazione
        * qry: stringa opzionale di query, se vogliamo inserirne una personalizzata

    result:
        * DataFrame con tabella risultante
    '''

    assert not (qry is None and not np.all([table, select, filteriter])), "Non abbastanza parametri definiti"

    lenls = len(ls_valori)
    if chunksize is None:
        chunksize = int(lenls / 10)

    if qry is None:
        qry = "SELECT {} FROM {} WHERE {} IN {}"

        ls_df = list()
        for chunk in grouper(ls_valori, chunksize, ''):
            tempqry = qry.format(', '.join(select), table, filteriter, chunk)
            dfchunk = pd.read_sql(tempqry, db)
            ls_df.append(dfchunk)

    else:
        
        ls_df = list()
        c = 0
        for chunk in grouper(ls_valori, chunksize, ''):
            chunk = [e for e in chunk if e != '']
            tempqry = qry.format(list_to_query(chunk))
            dfchunk = pd.read_sql(tempqry, db)
            ls_df.append(dfchunk)

            c += chunksize
            if verbose:
                print("Ne abbiamo lavorati {} su {}".format(c, lenls))

    res = pd.concat(ls_df, axis=0)

    return res

if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\LEI00020\Desktop\pipelineAPB\unico\datasets\datisx2018_new.csv', nrows=10000, encoding='latin1')
    crifsel = [
        "crif_score_affidabilita_seg1",
        "crif_score_affidabilita_seg2",
        "crif_imm_loc_min",
        "crif_stranieri",
        "crif_limite_utilizzo_cdc",
        "crif_anagrafiche_mi",
        "crif_contratti_in_default_t0",
        "crif_anagrafiche_cdc",
    ]

    truesel = [diz_oldnew[e] for e in crifsel]

    diz_feat_ind = {
        'INDIRIZZO':'ANT_SPTIND',
        'CAP':'ANT_SPTCAP',
        'CIVICO':'ANT_SPTNUM',
        'PROVINCIA':'ANT_SPTPROV',
        'CITTA':'ANT_SPTCOMUN'
    }
    with create_engine('postgresql://postgres:postgres@172.25.172.170:5433/Geocoder') as conn:
        res = query_crif(df, truesel, conn, feat_indirizzo = diz_feat_ind)

    print(res)
