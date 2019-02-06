import string

diz_table = {
    'CUSTOMER': {'columns': ['CF', 'CITTA']},
    'SHOP': {'columns': ['ID', 'CFOWNER', 'TIPO']}
}

diz_joinz = [
    {'tables':('CUSTOMER','SHOP'),'ON':[('CF','CFOWNER')]}
]

diz_filterz = {
    'CITTA': 'ROMA'
}

class Query():

    '''
    Oggetto per creare una query dando appositi input
    
    Parameters:
        columns: dizionario tabelle -> field
        joinz: lista di dizionari dove ogni dizioanrio è {'tables': {$table1: {($table2: ($field1,$field2))}}}
        wherez = dizionario per le where {$table:{$field:$valore}}

    esempi
        self.columns = {'CUSTOMER':['CF','CITTA'], 'SHOP':['CFOWNER', 'TIPO']}

        self.joinz = [
                {
                    'tables':
                    {
                        'CUSTOMER':[
                            ('SHOP', ('CF','CFOWNER'))
                            ]
                        }
                }
            ]

        self.wherez = {'CUSTOMER':{'CITTA':'ROMA'}}
    '''

    def __init__(self, columns, joinz, wherez):
        self.columns = columns
        self.tables = [t for t in self.columns.keys()]
        self.joinz = joinz
        self.wherez = wherez
        self.aliases = {table:string.ascii_lowercase[idx] for idx, table in enumerate(set(self.tables).union(set(wherez.keys())))}

        self.SELECT = ''
        self.FROM = ''
        self.JOIN = ''
        self.WHERE = ''

        self.res = ''

    @classmethod
    def wrap(self, w,sym):
        return sym+w+sym

    def compose_select(self):
        self.SELECT = 'SELECT {}'.format(', '.join([self.aliases[table]+'.'+column for table in self.tables for column in self.columns[table]]))

    def compose_from(self):
        self.FROM = 'FROM ' + self.tables[0] + ' AS ' + self.aliases[self.tables[0]]

    def compose_join(self):
        ls = list()
        for j in self.joinz:
            for k in j['tables'].keys():
                for e in j['tables'][k]:
                    ls.append('JOIN ' + e[0] + ' AS ' + self.aliases[e[0]])
                    conds = list()
                    for i in range(1, len(e)):
                        # print('i {}'.format(i))
                        conds.append(self.aliases[k] + '.' + e[i][0] + ' = ' + self.aliases[e[0]] + '.' + e[i][1])
                    
                    ls.extend(['ON ' + ' AND '.join(conds)])
                    # print(conds)
        self.JOIN = '\n'.join(ls)

    def compose_where(self):
        ls = list()
        for t in self.wherez.keys():
            for field, val in self.wherez[t].items():
                operatore, valore = val.split(' ')
                valore = self.wrap(valore, "'") if operatore != 'IN' and not valore.isdigit() else valore
                ls.append(self.aliases[t] + '.' + field + self.wrap(operatore, ' ') + valore)

        self.WHERE =  'WHERE ' + ' AND '.join(ls)


    def compose(self):
        
        self.compose_select()
        self.compose_from()
        if self.joinz != '':
            self.compose_join()
        self.compose_where()

        self.res = '\n'.join([self.SELECT, self.FROM, self.JOIN, self.WHERE])
    


# def compose(diz_table, filterz, joinz, limit=None):

#     for idx, table in enumerate(diz_table.keys()):
#         print(type(idx),idx)
#         diz_table[table]['alias'] = string.ascii_lowercase[idx]
#         diz_table[table]['columns'] = ['.'.join([diz_table[table]['alias'], e]) for e in diz_table[table]['columns']]

#     qry = "SELECT {} FROM {} WHERE {}"

#     qry_select = ', '.join([col for table in diz_table.keys() for col in diz_table[table]['columns']])
#     qry_tables = list()
#     for table in diz_table.keys():
#         qry_tables.append(table+' AS '+diz_table[table]['alias'])
#         if len(joinz) > 0:
#             for j in joinz:
#                 qry_tables.append('ON '+diz_table[j['t1']]['alias']+'.'+j['ON'][0][0]+'='+diz_table[j['t2']]['alias']+'.'+j['ON'][0][1])

#     qry_tables = ' '.join(qry_tables)

#     return qry.format(qry_select, qry_tables, '')

# print(compose(diz_table, diz_filterz, diz_joinz))


# def assign_alias(ls_table):
#     return {table:string.ascii_lowercase[idx] for idx, table in enumerate(ls_table)}

# def build_on(t1, t2, c1, c2):
#     return t1+'='+t2

# def build_from(joinz, diz_alias):
#     for j in joinz:
#         for t in j['tables']:
#             t+' AS'+diz_alias[t]
#             if len(j['ON']) > 0:
#                 for couple in j['ON']:
