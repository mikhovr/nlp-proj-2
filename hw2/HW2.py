
# coding: utf-8

# Для обработки выбран текст Дагласа Хофштадтера "Глаз разума". Необходимо произвести извлечение устойчивых биграмм (коллокаций).
# 
# Для морфологического анализа пользуюсь pymorphy2.
# 
# NLTK для предобработки, быстрого выделения биграмм, быстрого подсчёта различных характеристик.

# In[92]:

import pymorphy2
from pprint import pprint
import nltk
from tabulate import tabulate
from nltk.corpus import stopwords
import math

morph = pymorphy2.MorphAnalyzer()


# In[3]:

""" Dice's measure """
def mydice(Fa1, Fa2, Fa12):
    return 2*Fa12/(Fa1+Fa2)


# In[4]:

""" Mutual Information """
def myMI(N, Fa1, Fa2, Fa12):
    return math.log2((N*Fa12)/(Fa1*Fa2))


# In[5]:

text = open('mindeye.txt', 'r')
tokens = []
for line in text.readlines():
    tokens+=nltk.word_tokenize(line)


# Здесь производим предобработку текста. Удаляем токены, являющиеся символами, знаками препинания, числами.
# Также избавляемся от служебных частей речи (биграммы, где одна из частей речи -- служебная, практически никогда не являются коллокациями), местоимений и местоимённых прилагательных.

# In[6]:

badPOS = {'NPRO','PREP', 'CONJ', 'PRCL', 'INTJ','PNCT', 'NUMB', 'ROMN', 'UNKN', 'Apro'}
sw = stopwords.words("russian") # список стоп-слов из NLTK

# Списковое включение получается слишком длинным, поэтому вынесем в отдельные лямбда-функции 
# transform(token) -- то, что заносится в список
# condition(token) -- предикат на токен

transform = lambda x: morph.parse(x)[0]
condition = lambda x: not [item for item in badPOS if item in str(transform(x).tag)] and transform(x).word not in sw

words = [transform(token) for token in tokens if condition(token)]


# Подсчитаем количество словоупотреблений после чистки и количество словоформ.
# 
# Далее производится лемматизация с помощью приведения каждого слова к его нормальному виду.

# In[7]:

print("Количество словоупотреблений после чистки:", len(words))
print("Количество словоформ:", len(set(words)))
lemmatized = [word.normal_form for word in words]

print(lemmatized[:20], end=' ') # выведем первые 20 лемм, просто посмотреть


# In[97]:

lem_fd = nltk.FreqDist(lemmatized)
ranks = {}

for (i,w) in enumerate(lem_fd.most_common()):
    ranks[w[0]] = i+1

W = lem_fd.max() # Самое частотное слово (мочь)
Fw = lem_fd[W] # Абсолютная частота


# Как видим, самой частотной леммой оказалось "мочь".
# 
# С помощью NLTK создадим список биграмм. Затем посчитаем их абсолютные частоты. Выведем слова, чаще всего встречаемые вместе с "мочь".

# In[99]:

bgs = nltk.bigrams(lemmatized)

bgs_fd = nltk.ConditionalFreqDist(bgs)
bgs_fd[W].most_common(15)


# А теперь извлечём список самых устойчивых биграмм по мере Дайса.

# In[106]:

dicelist = sorted([(k,round(mydice(Fw, lem_fd[k], v),4)) for k,v in bgs_fd[W].items()], key= lambda x: x[1], reverse=True)
dicelist_top = dicelist[:20]

dicetable = [[W+' '+lemW, dice_val] for (lemW, dice_val) in dicelist_top]

print(tabulate(dicetable, headers=["Коллокация", "Dice"]))


# То же самое по мере Mutual Information:

# In[108]:

milist = sorted([(k,myMI(len(lemmatized), Fw, lem_fd[k], v)) for k,v in bgs_fd[W].items()], key= lambda x: x[1], reverse=True)
milist_top = milist[:20]

mitable = [[W+' '+lemW, mi_val] for (lemW, mi_val) in milist_top]

print(tabulate(mitable, headers=["Коллокация", "MI"]))


# Результат по мере Дайса получился весьма предсказуемым. Устойчивее те биграммы, в которых с нашим основным "мочь" второе слово встречается чаще. По мере "общей информации" (по крайней мере, по той формуле, которая представлена на слайде) результат получился странный и неочевидный. Большое количество результатов с одинаковым значением меры легко объясняется тем, что в тексте после чистки осталось не так много словоупотреблений ~80k, а словоформ и того меньше. Всё-таки даже книга не сравнится с большим корпусом. Однако всё было сделано правильно, поэтому странный результат для меры общей информации странен лишь по природе текста.
