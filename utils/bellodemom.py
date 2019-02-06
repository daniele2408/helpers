'''
Funzioni per formattare in modo pretty quei csv brutti brutti
'''

import pandas as pd

def df_to_excel(df, saveTo, cols=None, maps=None):
    '''
    Funzione che salva un DataFrame in excel

    Args:
        - df, (DataFrame): DataFrame da salvare
        - saveto, (str): path dove salvare l'excel
        - cols, (list): lista delle colonne da selezionare, se None sono tutte
        - maps, (dict): dizionario per rinominare le colonne, se None le lascia così come sono
    '''

    if cols:
        df = df[cols].copy()

    if maps:
        df.rename(index=str, columns=maps, inplace=True)

    writer = pd.ExcelWriter(saveTo, engine='xlsxwriter')
    workbook=writer.book
    worksheet=workbook.add_worksheet('Sheet1')
    writer.sheets['Sheet1'] = worksheet

    df.to_excel(writer, sheet_name='Sheet1', index=False)

    writer.save()