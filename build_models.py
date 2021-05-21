#!/usr/bin/env python
# coding: utf-8

"""
Данная программа содержит ряд экспериментов по построению моделей машинного обучения.

Разработчик: студент группы А-08-17 "НИУ МЭИ" 
Челышев Э.А.
e-mail: ChelyshevEA@mpei.ru
Место разработки: каф. ВМСС НИУ "МЭИ"
Дата разработки модуля: 17 мая 2021 г.

Входными данными является заранее подготовленный набор данных (файл dataset.csv).
Результатом работы являются обученные модели машинного обучения для задачи рубрикации текстов.
"""

# отчет о классификации
from sklearn.metrics import classification_report
# функция разделения на тренировочный и тестовый набор
from sklearn.model_selection import train_test_split
# импорт пакета для работы с датафреймами
import pandas as pd
import numpy as np  # для работы с массивами
import pickle  # для сохранения моделей
# логистическая регрессия
from sklearn.linear_model import LogisticRegression
# решетчатый поиск
from sklearn.model_selection import GridSearchCV
# для отображения матрицы путаницы
import confusion_matrix_pretty_print as cmpp
# гауссовский наивный байесовский классификатор
from sklearn.naive_bayes import GaussianNB
# предобработка данных
from sklearn import preprocessing
# случайный лес решающих деревьев
from sklearn.ensemble import RandomForestClassifier

"""
Данная функция выводит метрики качества модели классификации: отчет о классификации и матрицу путаницы.
На вход функция получает два массива: достоверный массив меток и предсказание модели.
Функция выводит отчет о классификации и отображает матрицу путаницы.
"""
def evaluate_quality(y_test, predicted):
    # отчет о классификации
    print(classification_report(y_test, lr_predicted, digits=5))
    # матрица путаницы
    cmpp.plot_confusion_matrix_from_data(y_test,
                                         lr_predicted,
                                         columns=range(0, 9),
                                         figsize=[7, 7])
    pass


# чтение данных
path = 'dataset.csv'
df = pd.read_csv(path)
# выделение меток классов в отдельный датафрейм
targets = df.filter(['topic'], axis=1)
df.drop(['topic'], axis='columns', inplace=True)
vectors = df  # содержит только векторы статей
# разделение на тренировочную и тестовую выборки

x_train, x_test, y_train, y_test = train_test_split(vectors,
                                                    targets,
                                                    test_size=0.25,
                                                    stratify=targets['topic'],
                                                    random_state=1)
# сжатие массивов меток до одной оси и приведение их к целочисленному типу
y_train = np.ravel(y_train)
y_train = np.array(y_train, dtype='int8')
y_test = np.ravel(y_test)
y_test = np.array(y_test, dtype='int8')

# приведение к типу float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

""" Наивный байесовский классификатор """
# нормализация данных
x_train_norm = preprocessing.normalize(x_train)
x_test_norm = preprocessing.normalize(x_test)

# обучение гауссовского наивного байесовского классификатора
bayes = GaussianNB().fit(x_train_norm, y_train)
bayes_predicted = bayes.predict(x_test_norm)
# оценка качества
evaluate_quality(y_test, bayes_predicted)
# Сохранить модель в файле
pickle.dump(bayes, open('bayes.pkl', 'wb'))

""" Логистическая регрессия """
# создание объекта логистической регрессии
lr = LogisticRegression(random_state=0, max_iter=10000)
# поиск гиперпараметров методом решетчатого поиска
log_params = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]}
log_grid = GridSearchCV(lr, log_params, cv=3)
log_grid.fit(x_train, y_train)
# вывод оптимальных гиперпараметров
print(log_grid.best_params_)
print(log_grid.cv_results_)

# обучение регрессии при оптимальных параметрах
lr_best = LogisticRegression(C=50, max_iter=10000, random_state=0)
lr_best.fit(x_train, y_train)

# оценка качества
lr_predicted = lr_best.predict(x_test)
evaluate_quality(y_test, lr_predicted)
# Сохранить модель в файле
pickle.dump(lr_best, open('log_reg.pkl', 'wb'))

""" Случайный лес решающих деревьев """
# поиск оптимальных параметров
rfc = RandomForestClassifier(random_state=42)
forest_params = {
    'n_estimators': [50, 100, 150, 200],
    'max_features': [5, 10, 15, 20, 25, 30, 35, 40]
}
forest_grid = GridSearchCV(rfc, forest_params, cv=3)
forest_grid.fit(x_train, y_train)
print(forest_grid.best_params_)
print(forest_grid.cv_results_)

# обучение для оптимальных параметров
rfc = RandomForestClassifier(n_estimators=150, max_features=30)
rfc.fit(x_train, y_train)
rfc_predicted = rfc.predict(x_test)

# оценка качества
evaluate_quality(y_test, rfc_predicted)
# Сохранить модель в файле
pickle.dump(rfc, open('forest.pkl', 'wb'))

