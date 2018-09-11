Il file wrap_model contiene una funzione principale che fa da wrap ad un modello GBM su h2o, in modo da conservare e recuperare velocemente tutti i dati usati e generati dal modello. È presente inoltre una classe per recuperare i dati su un modello passato per poterlo confrontare.

#### train_model_GBM ####

Dopo aver controllato se usare il dataset completo oppure le prime 10k righe, si assegna come nome la variabile model_id + il timestamp al lancio.
Viene inoltre definita la struttura delle directory per conservare le info sul modello:

rootDir
|-pathModel (cartella di un dato modello)
	|-PathData (contiene il csv del dataset)
		|-PathDataTrain (contiene il csv del train set con i predict)
		|-PathDataTest (contiene il csv del test set con i predict)


Lanciato il server e splittato il dataset, aggiungiamo il model_id completo al modello e il dataset di scaling creato or ora, instanziamo la classe h2o per il GBM e lanciamo il train. Il tempo impiegato verrà registrato a fine modello (idea, crea una versione per lanciare una griglia)

Vengono registrate le perfomance del modello, il modello viene salvato in pathModel con il model_id creato, lista e dizionario di predittori e column_types sono picklati anch'essi in pathModel.

Si creano i testfull e trainfull, train e test set con predizioni annesse, e salvati in pathDataTrain e pathDataTest.

Generiamo e salviamo  in pathModel un .txt che illustra una sintesi del modello.

La funzione restituisce il modello, test e train con predict, e l'oggetto per ottenere le performance.

#### ModelWrap ####

Classe per estrarre informazioni su un modello specificando una rootDir e un model_id (e il target che non mi porto dietro nel modello), in modo da poter confrontare agilmente vecchi modelli.

Nell'__init__ rifoma i path a partire dalla rootDir e recupera le informazioni ivi contenute.

Il metodo get_info()