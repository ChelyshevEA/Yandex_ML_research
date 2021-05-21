#!/usr/bin/env python
# coding: utf-8

"""
Данная программа подготавливает корпус для экспериментов по построению моделей машинного обучения.

Разработчик: студент группы А-08-17 "НИУ МЭИ" 
Челышев Э.А.
e-mail: ChelyshevEA@mpei.ru
Место разработки: каф. ВМСС НИУ "МЭИ"
Дата разработки модуля: 16 мая 2021 г.

Входными данными является корпус статей новостного портала lenta.ru (файл lentanews.csv).
Результатом выполнения данной программы являются подготовленный для построения моделей машинного обучения
корпус в виде файла dataset.csv.
Для работы необходимо подключить модуль prepare.py
"""

import pandas as pd  # импорт пакета для работы с датафреймами
# импорт пакета для отслеживания времени выполнения обработки датафрейма
from tqdm import tqdm
import prepare  # импорт модуля с функциями подготовки данных

tqdm.pandas()  # отслеживание времени выполнения для датафреймов pandas

# Считываем данные
data_path = 'C:/Users/Эдуард Челышев/Desktop/мл/Dip/diplom/lentanews.csv'
data = pd.read_csv(data_path,
                   sep=',',
                   error_bad_lines=False,
                   usecols=['text', 'topic', 'tags'])

# подсчет количества статей в рубриках
print(data.groupby(['topic']).count())
print(data.groupby(['tags']).count())

# формирование рубрики Политика
data['topic'] = np.where((data.tags == 'Политика'), 'Политика', data.topic)
# проверим наличие рубрики Политика
print(data.groupby(['topic']).count())

# удаление из корпуса статей, принадлежащих некоторым рубрикам
data = data.drop(
    data[(data['topic'] == '69-я параллель') | (data['topic'] == 'Библиотека')
         | (data['topic'] == 'Крым') | (data['topic'] == 'МедНовости') |
         (data['topic'] == 'Оружие') | (data['topic'] == 'Сочи') |
         (data['topic'] == 'ЧМ-2014') | (data['topic'] == 'Мир') |
         (data['topic'] == 'Культпросвет ') | (data['topic'] == 'Легпром') |
         (data['topic'] == 'Россия') | (data['topic'] == 'Бывший СССР') |
         (data['topic'] == 'Из жизни') | (data['topic'] == 'Ценности')].index)
# объединение рубрик Бизнес и Экономика
data = data.replace({'topic': {'Бизнес': 'Экономика'}})
# изучение оставшихся рубрик
print(data.groupby(['topic']).count())
# удаление строк, содержащих пропуски
data = data.dropna(axis='index', how='any')

# Построение списка из названий имеющихся рубрик
topics = np.sort(data['topic'].unique())
print(topics)
# построение словаря соответствия названий рубрик и их цифровых эквивалентов
dct = dict()  # объявление пустого словаря
label = 0  # нумерация рубрик начинается с 0
for item in topics:
    dct[item] = label  # добавляем новую рубрику
    label += 1
# замена названий рубрик на цифровые эквиваленты
data = data.replace({'topic': dct})

# рассмотрим структуру имещихся данных
print('Размерность датафрейма: {}'.format(data.shape))
print('Столбцы датафрейма: {}'.format(data.columns))
print('Информация: {}'.format(data.info()))
print('Статистические характеристики:\n')
data.describe()

# указываем модулю prepare адрес к модели векторизации
model_path = 'model/model.model'
prepare.load_model(model_path)

# произведем подготовку с использованием функции
# prepare_text модуля prepare
# обработка датафрейма осуществляется частями
chunk_list = []
for chunk in data:
    # для каждой части датафрейма осуществляем обработку
    chunk['text'] = chunk['text'].progress_apply(prepare.prepare_text)
    chunk_list.append(chunk)
# объединяем обработанные части датафрейма
data = pd.concat(chunk_list)

# создаем копию датафрейма
vectors = data.copy()
# разбиваем список по колонкам
vectors = pd.DataFrame(vectors['text'].to_list(), columns=range(0, 300))
# объединяем со столбцом рубрик
df = pd.concat([vectors, data['topic']], axis=1)
df = df.dropna(axis='index', how='any')  # удаляем пропуски
df.to_csv('dataset.csv', sep=',', header=True,
          index=False)  # сохрянем итоговый файл

