#!/usr/bin/env python
# coding: utf-8


"""
Программный модуль, содержащий функции для
обработки текстов на естественном языкe.

Разработчик: студент группы А-08-17 "НИУ МЭИ" 
Челышев Э.А.
e-mail: ChelyshevEA@mpei.ru
Место разработки: каф. ВМСС НИУ "МЭИ"
Дата разработки модуля: 15 мая 2021 г.

Входные и выходные данные описаны для каждой функции в отдельности.
См. поясняющую информацию далее.
Для корректной и полнофункциональной работы необходимо наличие 
файла предобученной модели векторизации FastText (файл model.model)
"""

import numpy as np  # импорт пакета для работы с массивами
import re  # импорт пакета для работы с регулярными выражениями
import gensim  # импорт пакета для работы с моделями векторизации
import nltk  # импорт пакета для работы с естественным языком
from nltk import word_tokenize  # импорт функции токенизации
from nltk.corpus import stopwords  # импорт стоп-слов
import pymorphy2  # импорт морфологического анализатора

# создание объекта класса морфологического анализатора
morph = pymorphy2.MorphAnalyzer()
# список стоп-слов русского языка
stop_words = stopwords.words("russian")
# объявление переменной для модели векторизации
w2v_model = None

"""
Данная функция устанавливает путь к внешему файлу модели векторизации
и загружает ее.
На вход функция получает строку - путь к файлу. По умолчанию путь 'model.model'
Каких-либо результатов функция не возвращает.
"""
def load_model(path='model.model'):
    try:
        # загрузка модели векторизации
        global w2v_model
        w2v_model = gensim.models.KeyedVectors.load(path)
    except:  # если не удалось загрузить модель векторизации
        print('Не удалось загрузить модель векторизации. Проверьте правильность пути')
    pass

"""
Данная функция осуществляет приведение строки текста к общему регистру (строчные буквы), а также 
производит удаление нерелевантных символов и встречающиеся в тексте URL-ссылки.
На вход функция получает text - строку текста, подлежащую обработке.
Результатом является обработанная строка
"""
def preprocess_text(text):
    # приведение к общему регистру и удаление буквы "ё"
    text = text.lower().replace("ё", "е")
    # удаление URL-ссылок
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL',
                  text)
    # удаление небуквенных символов
    text = re.sub('[^a-zA-Zа-яА-Я]+', ' ', text)
    # несколько подряд идущих пробелов заменяются на один
    text = re.sub(' +', ' ', text)
    # удаление пробелов с обоих концов строки, если таковые имеются
    preprocessed_text = text.strip()
    return preprocessed_text

"""
Данная функция осуществляет приведение токенов к начальной форме путем их лемматизации.
На вход функция получает text_list - список строк, каждая из которых
явлется отдельным токеном, подлежащим лемматизации.
Результатом функции является список токенов, прошедших лемматизацию
"""
def lemmatize(text_list):
    result = list()  # подготовка списка для хранения результата
    for token in text_list:  # цикл по токенам списка
        # получение начальной формы
        modified_token = morph.parse(token)[0].normal_form
        result.append(modified_token)  # добавление результата к списку
    return result

"""
Данная функция осущестляет удаление стоп-слов.
На вход функция получает список строк, являющихся отдельными токенами.
Результатом является список токенов, откуда удалены стоп-слова.
Для корректной работы данной функции токены должны быть приведены
к начальной форме с использованием функции lemmatize()
"""
def delete_stop_words(text_list):
    # подготовка списка для хранения результата
    result = list()
    # цикл по токенам входного списка
    for token in text_list:
        # если токен не является стоп-словом
        if token not in stop_words:
            # он добавляется к результирующему списку
            result.append(token)
    return result

"""
Данная функция осуществляет построение вектора статьи.
На вход функция получает список токенов.
Результатом выполнения функции является вектор размерности 300,
в формате массива NumPy.
Для корректной работы функции входной список должен быть обработан
с использованием функции prepare_text
"""
def build_article_vector(t):
    # счетчик числа векторизованных токенов
    count = 0
    # подготовка списка для хранения результата
    res = np.zeros(300)
    # цикл по токенам
    for word in t:
        # если токен содержится в модели
        if (word in w2v_model.vocab):
            # прибавляем полученный для токена вектор к результирующему
            res += w2v_model.get_vector(word)
            # увеличиваем счетчик векторизованных токенов
            count += 1
    # определяем среднее по статье
    res /= count
    return res

"""
Данная функция осуществляет полную подготовку текста на естественном языке,
а именно: токенизацию, лемматизацию и удаление стоп-слов.
На вход функция получает строку текста, подлежащую обработке.
Результатом выполнения функции является список токенов, прошедший этапы подготовки
"""
def prepare_text(text):
    # приведение к одному регистру и удаление нерелевантных символов
    text = preprocess_text(text)
    #  токенизация с использованием пакета nltk
    text_list = word_tokenize(text, language = "russian")
    # лемматизация с использованием функции lemmatize
    text_list = lemmatize(text_list)
    # удаление стоп-слов с использованием функции delete_stop_words
    text_list = delete_stop_words(text_list)
    # векторизация
    vector = build_article_vector(text_list)
    return vector



