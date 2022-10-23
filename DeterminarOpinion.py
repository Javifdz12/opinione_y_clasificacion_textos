import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import GridSearchCV


class determina_opinion:
    def __init__(self, dataset_name):
        self.mensajesTwitter = pd.read_csv(dataset_name, delimiter = ";")

    def cargar_archivo(self):
        '''Generamos nuevas propiedades a la clase que no están en el constructor:
                - Stopwords: Estas son palabras que no tienen relevancia en el lenguaje
                - Stemmer: 
        '''
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stopWords = stopwords.words('english')
        self.stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()


    def normalizacion(self):
        for mensaje in self.mensajesTwitter["TWEET"]:
            mensaje = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', mensaje)
            mensaje = re.sub('@[^\s]+','USER', mensaje)
            mensaje = mensaje.lower().replace("ё", "е")
            mensaje = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', mensaje)
            mensaje = re.sub(' +',' ', mensaje)
            mensaje.strip()
        print(self.mensajesTwitter.head(10))

    def normalizacion_frase(self, mensaje):
        mensaje = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', mensaje)
        mensaje = re.sub('@[^\s]+','USER', mensaje)
        mensaje = mensaje.lower().replace("ё", "е")
        mensaje = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', mensaje)
        mensaje = re.sub(' +',' ', mensaje)
        return mensaje.strip()

    def transformar_columna(self,columna):
        self.mensajesTwitter[columna] = (self.mensajesTwitter[columna]=='Yes').astype(int)
        print(self.mensajesTwitter.head(100))

    def eliminar_stopwords(self):
        self.mensajesTwitter['TWEET'] = self.mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([palabra for palabra in mensaje.split() if palabra not in (self.stopWords)]))
        print(self.mensajesTwitter.head(10))

    def stemming(self):
        self.mensajesTwitter['TWEET'] = self.mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([self.stemmer.stem(palabra) for palabra in mensaje.split(' ')]))
        print(self.mensajesTwitter.head(10))

    def preparacion(self):
        self.cargar_archivo()
        print(determina_opinion)
        self.transformar_columna('CREENCIA')
        self.normalizacion()
        self.eliminar_stopwords()
        self.stemming()
        print('')
        print('################ FIN DE LA PREPARACION ################')
        print('')

    def Aprendizaje_multinomial(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.mensajesTwitter['TWEET'].values,  self.mensajesTwitter['CREENCIA'].values,test_size=0.2)
        self.etapas_aprendizaje = Pipeline([('frequencia', CountVectorizer()),('tfidf', TfidfTransformer()),('algoritmo', MultinomialNB())])
        self.modelo = self.etapas_aprendizaje.fit(self.X_train,self.y_train)
        print(classification_report(self.y_test, self.modelo.predict(self.X_test), digits=4))

    def predecir_frase(self, frase):
        #Normalización
        frase = self.normalizacion_frase(frase)

        #Eliminación de las stops words
        frase = ' '.join([palabra for palabra in frase.split() if palabra not in (self.stopWords)])

        #Aplicación de stemming
        frase =  ' '.join([self.stemmer.stem(palabra) for palabra in frase.split(' ')])

        #Lematización
        frase = ' '.join([self.lemmatizer.lemmatize(palabra) for palabra in frase.split(' ')])
        print (frase)

        prediccion = self.modelo.predict([frase])
        print(prediccion)
        if(prediccion[0]==0):
            print(">> No cree en el calentamiento climático...")
        else:
            print(">> Cree en el calentamiento climático...")

    def aprendizaje_svm(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.mensajesTwitter['TWEET'].values,  self.mensajesTwitter['CREENCIA'].values,test_size=0.2)
        self.etapas_aprendizaje = Pipeline([('frequencia', CountVectorizer()),('tfidf', TfidfTransformer()),('algoritmo', svm.SVC(kernel='linear', C=2))])
        self.modelo = self.etapas_aprendizaje.fit(self.X_train,self.y_train)
        print(classification_report(self.y_test, self.modelo.predict(self.X_test), digits=4))

    def aprendizaje_svm_avanzado(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.mensajesTwitter['TWEET'].values,  self.mensajesTwitter['CREENCIA'].values,test_size=0.2)
        self.etapas_aprendizaje = Pipeline([('frequencia', CountVectorizer()),('tfidf', TfidfTransformer()),('algoritmo', svm.SVC(kernel='linear', C=2))])
        self.modelo = self.etapas_aprendizaje.fit(self.X_train,self.y_train)

        parametrosC = {'algoritmo__C':(1,2,4,5,6,7,8,9,10,11,12)}

        self.busquedaCOptimo = GridSearchCV(self.etapas_aprendizaje, parametrosC,cv=2)
        self.busquedaCOptimo.fit(self.X_train,self.y_train)
        print(self.busquedaCOptimo.best_params_['algoritmo__C'])

        #Parámetro nuevo C=1
        etapas_aprendizaje = Pipeline([('frequencia', CountVectorizer()),
                                        ('tfidf', TfidfTransformer()),
                                        ('algoritmo', svm.SVC(kernel='linear', C=self.busquedaCOptimo.best_params_['algoritmo__C']))])

        modelo = etapas_aprendizaje.fit(self.X_train,self.y_train)
        print(classification_report(self.y_test, modelo.predict(self.X_test), digits=4))

    def __str__(self):
        print('Nuestro dataset:\n')
        print(self.mensajesTwitter.head(10))
        return f'Tiene un tamaño de {self.mensajesTwitter.shape[0]}'



op=determina_opinion('datas\calentamientoClimatico.csv')
op.preparacion()
op.Aprendizaje_multinomial()
