# -*- coding: utf-8 -*-

# imports
import pandas as pd  # biblioteca pandas
import matplotlib.pyplot as plt  # biblioteca para criação do gráfico
from sklearn.tree import DecisionTreeClassifier  # biblioteca que permite a criação da árvore de decisão
from sklearn.model_selection import train_test_split  # biblioteca que divide em treino e teste
from sklearn.metrics import accuracy_score  # biblioteca que permite calcular a eficiência do teste
# from sklearn.model_selection import cross_val_score  # nova biblioteca

# definições iniciais
nameColumns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
dataFrame = pd.read_csv('iris.csv', names=nameColumns)
print("Linhas: %d, Colunas: %d" % (len(dataFrame), len(dataFrame.columns)))

# criação de novas colunas a partir de outras já existentes
dataFrame['SepalArea'] = dataFrame['SepalLength'] * dataFrame['SepalWidth']
dataFrame['PetalArea'] = dataFrame['PetalLength'] * dataFrame['PetalWidth']

# criação de outras colunas de valores booleanos
dataFrame['SepalLengthAboveMean'] = dataFrame['SepalLength'] > dataFrame['SepalLength'].mean()
dataFrame['SepalWidthAboveMean'] = dataFrame['SepalWidth'] > dataFrame['SepalWidth'].mean()
dataFrame['PetalLengthAboveMean'] = dataFrame['PetalLength'] > dataFrame['PetalLength'].mean()
dataFrame['PetalWidthAboveMean'] = dataFrame['PetalWidth'] > dataFrame['PetalWidth'].mean()

# cria uma variável sem a coluna "Class"
features = dataFrame.columns.difference(['Class'])

# print do cabeçalho e features
# print(f"Features: {features}\n")
# print(f"{dataFrame.head()}\n")

# define formato para o gráfico
dataFrame['Class'].value_counts().plot(title='Iris', kind='pie', figsize=(5, 5), autopct='%1.1f%%', startangle=90)
# plt.show() #mostra o gráfico

# criação de variáveis para análise
X = dataFrame[features].values
y = dataFrame['Class'].values

# dados para serem comparados
sample1 = [1.0, 2.0, 3.5, 1.0, 10.0, 3.5, False, False, False, False]  # exemplo de Iris-setosa
sample2 = [5.0, 3.5, 1.3, 0.2, 17.8, 0.2, False, True, False, False]  # exemplo de Iris-versicolor
sample3 = [7.9, 5.0, 2.0, 1.8, 19.7, 9.1, True, False, True, True]  # exemplo de Iris-virginica

# colocandos os dados que serão comparados em um dataframe
sample = pd.DataFrame([sample1, sample2, sample3], columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth',
                                                            'SepalArea', 'PetalArea', 'SepalLengthAboveMean',
                                                            'SepalWidthAboveMean', 'PetalLengthAboveMean',
                                                            'PetalWidthAboveMean'])
sample = sample[features].values

# criação da árvore de decisão para treino
classifier_dt = DecisionTreeClassifier(max_depth=2)
# classifier_dt = DecisionTreeClassifier(random_state=10, criterion='gini', max_depth=3)

# efetuar treinamento
classifier_dt.fit(X, y)

# a partir dos dados que foram apresentados, tenta-se advinhar a 'Class'
print(classifier_dt.predict([sample1, sample2, sample3]))

# '''
# dividindo entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# uma nova árvore de decisão com um novo classificador
classifier_dt2 = DecisionTreeClassifier(max_depth=2)
classifier_dt2.fit(X_train, y_train)

# tenta dar o predict e calcula a eficiência
y_pred = classifier_dt2.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nLinhas treinamento: %d\nLinhas teste: %d" % (len(X_train), len(X_test)))

print(f"\nEficiência do teste foi de: {accuracy}")
# '''

'''
classifier_dt3 = DecisionTreeClassifier(max_depth=2)
scores_dt = cross_val_score(classifier_dt3, X, y, scoring='accuracy', cv=5)
scores_dt.mean()
'''
