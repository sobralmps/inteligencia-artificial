#-*- coding: utf-8 -*-

#imports
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

#definições iniciais
nameColumns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
dataFrame = pd.read_csv('iris.csv', names=nameColumns)
print("Linhas: %d, Colunas: %d\n" % (len(dataFrame), len(dataFrame.columns)))

#criação de novas colunas a partir de outras já existentes
dataFrame['SepalArea'] = dataFrame['SepalLength'] * dataFrame['SepalWidth']
dataFrame['PetalArea'] = dataFrame['PetalLength'] * dataFrame['PetalWidth']

#criação de outras colunas de valores booleanos
dataFrame['SepalLengthAboveMean'] = dataFrame['SepalLength'] > dataFrame['SepalLength'].mean()
dataFrame['SepalWidthAboveMean'] = dataFrame['SepalWidth'] > dataFrame['SepalWidth'].mean()
dataFrame['PetalLengthAboveMean'] = dataFrame['PetalLength'] > dataFrame['PetalLength'].mean()
dataFrame['PetalWidthAboveMean'] = dataFrame['PetalWidth'] > dataFrame['PetalWidth'].mean()

#cria uma variável sem a coluna "Class"
features = dataFrame.columns.difference(['Class'])

#print do cabeçalho e features
#print(f"Features: {features}\n")
#print(f"{dataFrame.head()}\n")

#define formato para o gráfico
dataFrame['Class'].value_counts().plot(title='Iris', kind='pie', figsize=(5, 5), autopct='%1.1f%%', startangle=90)
plt.show() #mostra o gráfico

#criação de variáveis para análise
X = dataFrame[features].values
y = dataFrame['Class'].values

#dados para serem treinados
sample1 = [1.0, 2.0, 3.5, 1.0, 10.0, 3.5, False, False, False, False]#exemplo de Iris-setosa
sample2 = [5.0, 3.5, 1.3, 0.2, 17.8, 0.2, False, True, False, False]#exemplo de Iris-versicolor
sample3 = [7.9, 5.0, 2.0, 1.8, 19.7, 9.1, True, False, True, True]#exemplo de Iris-virginica

#criação da árvore de decisão
#classifier_dt = DecisionTreeClassifier(max_depth=3)
classifier_dt = DecisionTreeClassifier(random_state=10, criterion='gini', max_depth=3)

#efetuar treinamento
classifier_dt.fit(X, y)

#a partir dos dados que foram apresentados, tenta-se advinhar a 'Class'
print(classifier_dt.predict([sample1, sample2, sample3]))
