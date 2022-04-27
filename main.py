from logging import Handler
import random
import nltk
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from telegram import Update #Кусок новой информации
from telegram.ext import Updater #Инстурмент который получает апдейты с серверов
from telegram.ext import MessageHandler, Filters
from sklearn.model_selection import GridSearchCV
from config import api_token

#Загатовленные фразы и загатовленные реакции
# BOT_CONFIG = {
#     'hello': {
#         'examples':['Хеллоу', 'Привет', 'Салам', 'Здравствуйте'],
#         'responces':['Приветик','Калайсын', 'Салютик', 'Хай', 'Мхм'],
#     },
#     'how-do-you-do': {
#         'examples':['Как дела', 'Что как', 'Что по чем', 'Нестеп'],
#         'responces':['Класс','Пойдет', 'Валяюсь', 'Бегу', 'кушаю'],
#     },
#     'location': {
#         'examples':['Ты где', 'Где находишься', 'Откуда родом', 'Где был вчера'],
#         'responces':['Дома','На работе', 'В пути', 'В отпуске', 'На пляже'],
#     },
#     'hobbies': {
#         'examples':['Чем увлекаешься', 'Какие хобби', 'Чем занимаешься в свободное время',],
#         'responces':['Играю настолный теннис','Читаю', 'Играю в видеоигры', 'Бегаю по утрам', 'Хожу в горы'],
#     },
#     'programming_languages': {
#         'examples':['Какими ЯП владеешь', 'Что изучаешь в данный момент', 'Какой базовый ЯП',],
#         'responces':['Python','JS', 'Go', 'C++', 'java'],
#     },
#     'sports': {
#         'examples':['Любимы вид спорта', 'Каким спортом увлекаешься', 'Что смотришь из спорта',],
#         'responces':['Футбол','Хоккей', 'Лыжный спорт', 'Баскетбол', 'Теннис'],
#     },
#     'cars': {
#         'examples':['Какую машину предпочитаешь', 'Любимая марка машины', 'На чем катаешься',],
#         'responces':['Toyota','Subaru', 'Tesla', 'Mersedes', 'BMW'],
#     },
#     'jobs': {
#         'examples':['Где работаешь', 'Место работы', 'Чем занимаешься',],
#         'responces':['Наемный работник','ИП', 'Фрилансер'],
#     },
#     'education': {
#         'examples':['Где учился', 'Что закончил',],
#         'responces':['ВУЗ','Коллежд', '11 классов'],
#     },
#     'languages': {
#         'examples':['Какими языками владешь', 'На каких языках общаещься',],
#         'responces':['Казахский','Русский', 'английский'],
#     },
#     'porfolio': {
#         'examples':['Какие проекты есть', 'Что написал', 'Что есть в портфоио', 'Чем похвастать можешь'],
#         'responces':['Сайт на Django','Телеграмм бот на aiogram', 'редактор фото на Pillow'],
#     },
#     'marital status': {
#         'examples':['Семейный статус', 'Женат', 'Есть семья'],
#         'responces':['Женат','Не женат', 'в отношениях'],
#     },
#     'bye': {
#         'examples':['Пока', 'Бай', 'Давай', 'Чао', "все давай иди гуляй"],
#         'responces':['Попока','Досвидания', 'Аривидерчи', 'Чао какао', 'Сау бол'],
#     },
# }








# 1. собрать и загрузить данные
config_file = open('big_bot_config.json', 'r')
BOT_CONFIG = json.load(config_file)
# print(len(BOT_CONFIG['intents'].keys()))

# 2. Подготовить/обработать данные

# X - тексты (примеры)
X = []

# Y - названия интентов (классы)
y = []

for name, data in BOT_CONFIG['intents'].items():
    for example in data['examples']:
        X.append(example)
        y.append(name)

# Векторайзер превращает тексты в наборы чисел (векторы)

vectorizer = CountVectorizer()
vectorizer.fit(X) #Учит тесты преобразовывать в вектора
vecX = vectorizer.transform(X)

# 3. Обучть модель (алгоритм,настройки)

# model = LogisticRegression()
# model.fit(vecX, y)

# test = vectorizer.transform(['меньше чем за миллион я не согласился бы'])
# model.predict(test)

# 4. Проверить качество модели

# y_pred = model.predict(vecX)

model = RandomForestClassifier(n_estimators = 500, min_samples_split = 3)
model.fit(vecX, y)

# ideal_model = RandomForestClassifier()
# param = {
#     'n_estimator' : [60, 140],
#     'criterion' : ['gini', 'entropy'],
# }

# cv = GridSearchCV(ideal_model, param, scroing = 'jaccard')
# cv.fit(X,y)


# Соединяем все вместе
def filter(text):
    alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя -'
    result = [c for c in text if c in alphabet]  # фильтрует символы не входящие в список
    return ''.join(result)

# Если текст похож на example
def match(text, example):
    text = filter(text.lower())
    example = example.lower()
    distance = nltk.edit_distance(text, example) / len(example)
    return distance < 0.5

def get_intent(text):
    for intent in BOT_CONFIG['intents']:
        for example in BOT_CONFIG["intents"][intent]["examples"]:
            if match(text, example):
                return intent

def bot(text):
    intent = get_intent(text) #Пытаемя сходу понять намерение
    if not intent: # Если не найдено привлекаем модель
        transformed_text = vectorizer.transform([text])
        intent = model.predict(transformed_text)[0]
    
    if intent: # Если намерение найдено выдаем случайные ответ
        return random.choice(BOT_CONFIG['intents'][intent]['examples'])
    
    return random.choice(BOT_CONFIG['failure_phrases'])

question = ''
while question != 'Выйти':
    question = input('Начнем:')
    answer = bot(question)
    print(f'[Юзер]: {question}')
    print(f'[Бот]: {answer}')
        
# text = input('Enter something: ') 
# answer = get_intent(text)
# print(answer)

# Мама круто мыла раму
# круто мама мыла раму
# мыла рму круто мама

# Класификация текстов

BOT_KEY = api_token

def BotReactOnMessage(update : Update, context):
    text = update.message.text # То что пользователь написал
    print(f'[user]: {text}')
    reply = bot(text)
    update.message.reply_text(reply)



upd = Updater(BOT_KEY)


#Хандлер=обработчик
#На все текстовые сообщения реагирет BotReactOnMessage
handler = MessageHandler(Filters.text, BotReactOnMessage)
upd.dispatcher.add_handler(handler)


upd.start_polling()
upd.idle()


