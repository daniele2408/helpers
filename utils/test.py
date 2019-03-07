import helpers.utils.UtilsDataManip as utils
import toolbusi.toolreco.get_info_db as getdb
import unittest
from collections import defaultdict

# res = getdb.get_clienti_possessi_ramo(['CMPDNL90M24H501U', 'MLNNDR82S12E463V', 'RVTGNI26H70H883Y'], ret_df=True)
# res = getdb.comprimi_target(res)

# res = getdb.get_prov_by_agz('39317', rettop=3)
# res = getdb.get_prov_by_agz('39317', rettop=1)

# res = getdb.get_cf_agenzie(['CMPDNL90M24H501U'])
# res = getdb.get_cf_recapiti(['CMPDNL90M24H501U'])
# res = getdb.get_agz_cf(['59956'])
# res = getdb.get_cf_privacy(['CMPDNL90M24H501U'])
res = getdb.get_cf_info_auto(['CMPDNL90M24H501U'])

print(res)


class test_get_tariffa_vita(unittest.TestCase):
    def test_retdf_true(self):
        # ATTENZIONE prendere un cliente con almeno una polizza attiva
        res = getdb.get_clienti_possessi_ramo(['CMPDNL90M24H501U'], ret_df=True)
        
        self.assertEqual(list(res.columns), ['cod_fiscale_p_iva', 'ramo_gestione', 'tariffa_vita', 'possessi_vita'])

    def test_retdf_false(self):
        res = getdb.get_clienti_possessi_ramo(['CMPDNL90M24H501U'], ret_df=False)

        # deve essere un dizionario di dizionari
        self.assertIsInstance(res, type(defaultdict(dict)))
        
        # i valori al secondo livello devono essere dei set
        self.assertIsInstance(type(res['CMPDNL90M24H501U']['ramo_gestione']), type(set))
        self.assertIsInstance(type(res['CMPDNL90M24H501U']['possessi_vita']), type(set))

class test_get_possessi_clienti_wide(unittest.TestCase):
    def test_result_cliente_con_vita(self):
        # testiamo un cliente con solo prodotti vita
        res = getdb.get_possessi_clienti_wide(['MLNNDR82S12E463V'])
        col_prod_vita = ['possessi_vita_investimento', 'possessi_vita_previdenza', 'possessi_vita_protezione', 'possessi_vita_risparmio']
        
        # controlla se ci sono le colonne dei possessi vita
        col_vita_effettive = set(res.columns).intersection(set(col_prod_vita))
        self.assertTrue(len(col_vita_effettive)>0)
        # controlla che, qualora ci siano, non siano tutte zero
        self.assertTrue(res[res[list(col_vita_effettive)]==1].sum().sum()>0)

    def test_result_cliente_con_auto(self):
        # testiamo un cliente con solo prodotti auto
        res = getdb.get_possessi_clienti_wide(['CMPDNL90M24H501U'])
        col_prod_auto = ['ramo_gestione_1001', 'ramo_gestione_1002']
        
        # controlla se ci sono le colonne dei possessi auto
        col_auto_effettive = set(res.columns).intersection(set(col_prod_auto))
        self.assertTrue(len(col_auto_effettive)>0)
        # controlla che, qualora ci siano, non siano tutte zero
        self.assertTrue(res[res[list(col_auto_effettive)]==1].sum().sum()>0)

    def test_result_cliente_con_danno(self):
        # testiamo un cliente con solo prodotti auto
        res = getdb.get_possessi_clienti_wide(['RVTGNI26H70H883Y'])
        col_prod_danno = [c for c in res.columns if c.startswith('ramo_gestione_') and c[-4:] not in {'1001', '1002'}]
        
        # controlla se ci sono le colonne dei possessi danno
        self.assertTrue(len(set(res.columns).intersection(set(col_prod_danno)))>0)
        # controlla che, qualora ci siano, non siano tutte zero
        self.assertTrue(res[res[col_prod_danno]==1].sum().sum()>0)

if __name__ == '__main__':
    pass
    # unittest.main()