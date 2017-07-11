
# coding: utf-8

# # 0 Загружаем библиотеки

# In[1]:

import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import cross_validation, datasets

import traceback

import numpy as np

import BasicFunc_for_gb2fs as bf
import os
import re


# # 1 Загружаем исходные данные

# In[2]:

trainFile = u'D:\\Python\\Хорошие плохие смс\\clientSMS_v2.csv'
#для загрузки файла используем магию ниже
pwd = os.getcwd()
os.chdir(os.path.dirname(trainFile))
df_start = pd.read_csv(os.path.basename(trainFile) , header = None, sep = '\t')

os.chdir(pwd)
#задаем имена столбцов
column_names = ['text','GoB']
df_start.columns = column_names


# In[3]:

#основной справочник сформированных фич
allRules = {
    'rule_1' : ([]),
    'rule_2' : ([]),
    'rule_3' : ([]),
    'rule_4' : ([]),
    'rule_5' : ([]),
    'rule_6' : ([]),
    'rule_7' : ([]),
    'rule_8' : ([]),
    'rule_9' : ([]),
    'rule_10' : ([]),
    'rule_11' : ([]),
    'rule_12' : ([]),
    'rule_13' : ([]),
    'rule_14' : ([]),
    'rule_15' : ([]),
    'rule_16' : ([]),
    'rule_17' : ([]),
    'rule_18' : ([]),
    'rule_19' : ([]),
    'rule_20' : ([]),
    'rule_21' : ([]),
    'rule_22' : ([]),
    'rule_23' : ([]),
    'rule_24' : ([]),
    'rule_25' : ([]),
    'rule_26' : ([]),
    'rule_27' : ([]),
    'rule_28' : ([]),
    'rule_29' : ([]),
    'rule_30' : ([]),
    'rule_31' : ([]),
    'rule_32' : ([]),
    'rule_33' : ([]),
    'rule_34' : ([]),
    'rule_35' : ([]),
    'rule_36' : ([]),
    'rule_37' : ([]),
    'rule_38' : ([]),
    'rule_39' : ([]),
    'rule_40' : ([]),
    'rule_41' : ([]),
    'rule_42' : ([]),
    'rule_43' : ([]),
    'rule_44' : ([]),
    'rule_45' : ([]),
    'rule_46' : ([]),
    'rule_47' : ([]),
    'rule_48' : ([]),
    'rule_49' : ([]),
    'rule_50' : ([]),
    'rule_51' : ([]),
    'rule_52' : ([]),
    'rule_53' : ([]),
    'rule_54' : ([]),
    'rule_55' : ([]),
    'final_rule' : ([]),

}


# In[4]:

%%time
df_start.head()

list_text = ([])
iii = 0



for sms in df_start['text'].values:
    #убираем лишние символы
    sms = sms.replace(':','')
    sms = sms.replace('-','')
    
    #разделяем слова в смс
    WiS = re.split('(\W+)', sms)
    #удаляем пробелы в смс
    while ' ' in WiS:
        WiS.remove(' ')
    #создаем стандартный справочник с итоговыми значениями сработавших правил по сообщению
    rules = {
        'rule_1' : 0,
        'rule_2' : 0,
        'rule_3' : 0,
        'rule_4' : 0,
        'rule_5' : 0,
        'rule_6' : 0,
        'rule_7' : 0,
        'rule_8' : 0,
        'rule_9' : 0,
        'rule_10' : 0,
        'rule_11' : 0,
        'rule_12' : 0,
        'rule_13' : 0,
        'rule_14' : 0,
        'rule_15' : 0,
        'rule_16' : 0,
        'rule_17' : 0,
        'rule_18' : 0,
        'rule_19' : 0,
        'rule_20' : 0,
        'rule_21' : 0,
        'rule_22' : 0,
        'rule_23' : 0,
        'rule_24' : 0,
        'rule_25' : 0,
        'rule_26' : 0,
        'rule_27' : 0,
        'rule_28' : 0,
        'rule_29' : 0,
        'rule_30' : 0,
        'rule_31' : 0,
        'rule_32' : 0,
        'rule_33' : 0,
        'rule_34' : 0,
        'rule_35' : 0,
        'rule_36' : 0,
        'rule_37' : 0,
        'rule_38' : 0,
        'rule_39' : 0,
        'rule_40' : 0,
        'rule_41' : 0,
        'rule_42' : 0,
        'rule_43' : 0,
        'rule_44' : 0,
        'rule_45' : 0,
        'rule_46' : 0,
        'rule_47' : 0,
        'rule_48' : 0,
        'rule_49' : 0,
        'rule_50' : 0,
        'rule_51' : 0,
        'rule_52' : 0,
        'rule_53' : 0,
        'rule_54' : 0,
        'rule_55' : 0,
        'final_rule' : 0,
        
    }
    #задаем начальные данные
    prevWord = 'EndLine'
    result = 0
    
    #определяем количество объектов в смс и затем начинаем работу по 61 правилам Толмачева
    Step = len(WiS)-1
    #идем с конца строки к началу прорабатывая каждое правило
    #с конца на начало нужно двигаться, т.к. существуют предшествующие объекты, которые отменяют либо оборачивают правило
    while Step >= 0:
        lookObject = WiS[Step].lower()
        
        #1 - смайлики
        rule_1 = 0
        rule_1 = lookObject.count(')') + lookObject.count('-P') + lookObject.count('*') + lookObject.count('^') - lookObject.count('(')
        rules['rule_1'] = rules['rule_1'] + rule_1
        
        #2 - знаки
        rule_2 = 0
        condition_2_1 = lookObject.count('?!') + lookObject.count('!?')
        if condition_2_1 == 0:
            condition_2_2 = lookObject.count('!')
            condition_2_3 = lookObject.count('?')
            if condition_2_2 > 1 :
                rule_2 = rule_2 + condition_2_2
            if condition_2_3 > 1 :
                rule_2 = rule_2 - condition_2_3
        rules['rule_2'] = rules['rule_2'] + rule_2
        
        #3 - уменьшительно ласкательные суффиксы
        rule_3 = 0
        rule_3_dict = ['чка', 'уля', 'йка', 'чка',  'тик', 'шка', 'шко', 'нький',
                       'нько', 'кни', 'юша', 'ська', 'дки', 'ньку', 'тка', 'сики',
                       'нька', 'хин', 'юхе', 'шка', 'чик', 'нки', 'нка', 'ться', 
                       'юха', 'юша', 'уля', 'лькин', 'жку', 'чик', 'мка']
        if len(lookObject) > 3:
            if lookObject[-3:] in rule_3_dict:
                rule_3 = 1
        elif len(lookObject) > 4:
            if lookObject[-4:] in rule_3_dict:
                rule_3 = 1
        elif len(lookObject) > 5:
            if lookObject[-5:] in rule_3_dict:
                rule_3 = 1
        rules['rule_3'] = rules['rule_3'] + rule_3
        
        #4 - явные слова
        rule_4 = 0
        rule_4_1_dict = ['хорошо', 'хороший', 'хорошие', 'отлично', 'плохо', 'нормально',
                       'любимая', 'понравится', 'люблю', 'рада', 'любимый', 'любимая',
                       'лучше', 'поцелую', 'поцелуй', 'целую', 'обнимаю', 'желаю', 'молодец',
                       'целую', 'лучший', 'любящий', 'надежный', 'счастье', 'счастлив',]
        rule_4_2_dict = ['испортил', 'плохо', 'отстой']
        if lookObject in rule_4_1_dict:
            rule_4 = 1
        if lookObject in rule_4_2_dict:
            rule_4 = -1
        rules['rule_4'] = rules['rule_4'] + rule_4
        
        #5 - матерные слова
        rule_5 = 0
        rule_5_dict = ['говно', 'сука', 'хуево', 'хуй', 'мутишь', 'хуле', 'охуел', 'ебал',
                       'блять', 'бля', 'ебу', 'нах', 'пизда', 'пиздеть', 'пиздец', 'пизданутый',
                       'пизданутая', 'пизданутые', 'ахуе', 'ахренеть', 'капец', 'ебанутый', 'ебан',
                       'заебал', 'нахуй', 'нах', 'сука', 'дохуя', 'пизди', 'нефига', 'минет', 'ебошишь',
                       'еб', 'ебу', 'мразь', 'мрази', 'мразота']
        if lookObject in rule_5_dict:
            rule_5 = -1
        rules['rule_5'] = rules['rule_5'] + rule_5
        
        #6 - увеличение и уменьшение
        rule_6 = 0
        rule_6_1_dict = ['пришло', 'прибыль','увеличилось','увеличение']
        rule_6_2_dict = ['ушло', 'уменьшилось' , 'упало']
        if lookObject in rule_6_1_dict:
            rule_6 = 1
        if lookObject in rule_6_2_dict:
            rule_6 = -1
        rules['rule_6'] = rules['rule_6'] + rule_6
        
        #7 - стандарты автоответов
        rule_7 = 0
        rule_7_dict = ['перезвонить', 'абонент' , 'услуга', 'пропущенный', 'просьба', 'получена', 'доставлено', 'перезвоните', 'звонил', 'пропущен', 'едет']
        if lookObject in rule_7_dict:
            rule_7 = 1
        rules['rule_7'] = rules['rule_7'] + rule_7
        
        #8 - жизнь хорошо, смерть плохо
        rule_8 = 0
        rule_8_dict = ['убью', 'пожалеешь']
        if lookObject in rule_8_dict:
            rule_8 = -1
        rules['rule_8'] = rules['rule_8'] + rule_8
        
        #9 - друг хорошо, враг плохо
        rule_9 = 0
        rule_9_1_dict = ['друг' , 'дружба', 'друзья', 'дружить']
        rule_9_2_dict = ['враг' , 'вражда', 'война', 'воюем','воевать','вражина']
        if lookObject in rule_9_1_dict:
            rule_9 = 1
        if lookObject in rule_9_2_dict:
            rule_9 = -1
        rules['rule_9'] = rules['rule_9'] + rule_9
        
        #10 - ужас, страх это плохо
        rule_10 = 0
        rule_10_dict = ['страшно','страх','боюсь','переживаю','волнуюсь','ужас','жестоко']
        if lookObject in rule_10_dict:
            rule_10 = -1
        rules['rule_10'] = rules['rule_10'] + rule_10
        
        #11 - троеточие или две точки чаще плохо, больше 3 точек чаще хорошо
        rule_11 = 0
        if lookObject.count('.') <=3 and lookObject.count('.') > 1:
            rule_11 = -1
        elif lookObject.count('.') > 3:
            rule_11 = 1
        rules['rule_11'] = rules['rule_11'] + rule_11
        
        #12 - стандарты магазинов это позитивчик
        rule_12 = 0
        rule_12_dict =['распродажа', 'акция']
        if lookObject in rule_12_dict:
            rule_12 = 1
        rules['rule_12'] = rules['rule_12'] + rule_12
        
        #13 - когда скучают это хорошо
        rule_13 = 0
        rule_13_dict = ['скучаем', 'соскучилась','ждем','жду','дождусь','ожидание','встречу']
        if lookObject in rule_13_dict:
            rule_13 = 1
        rules['rule_13'] = rules['rule_13'] + rule_13
        
        #14 - приветствие это хорошо
        rule_14 = 0
        rule_14_dict = ['привет', 'здравствуй', 'здарова', 'добрый', 'здорова',
                        'здравие', 'здравствуйте', 'добрым', 'утром', 'доброго',
                        'добро', 'сладкого', 'сладкий', 'сладких', 'доброе']
        if lookObject in rule_14_dict:
            rule_14 = 1
        rules['rule_14'] = rules['rule_14'] + rule_14
        
        #15 - тянущиеся гласные это хорошо
        rule_15 = 0
        symbols = re.split('(\w)', lookObject)
        while '' in symbols:
            symbols.remove('')
        len_symb = len(symbols)
        i = 1
        wooord = ''

        if len_symb > 1:
            while i < len_symb:
                if symbols[i-1]  == symbols[i] and symbols[i] in ['а', 'о', 'и', 'е', 'ё', 'э', 'ы', 'у', 'ю', 'я']:
                    word = '1'
                else:
                    word = '0'
                i += 1
                wooord = wooord+word
            symbols = re.split('0', wooord)
        for i in symbols:
            if len(i) > 2:
                rule_15 = 1
        rules['rule_15'] = rules['rule_15'] + rule_15
        
        #16 - эмоции
        rule_16 = 0
        rule_16_2_dict = ['плохо', 'истерика','истеричка', 'разочарование', 'разочаровал', 'разочарую', 'разочарован']
        if lookObject in rule_16_2_dict:
            rule_16 = -1
        rules['rule_16'] = rules['rule_16'] + rule_16
        
        #17 - срочность это плохо, отрицание срочности хорошо
        rule_17 = 0
        rule_17_2_dict = ['срочно']
        if lookObject in rule_17_2_dict:
            rule_17 = -1
        rules['rule_17'] = rules['rule_17'] + rule_17
        
        #18 - количество слов в смс
        rule_18 = 1
        rules['rule_18'] = rules['rule_18'] + rule_18
        
        #19 - предлог "бы" чаще плохо
        rule_19 = 0
        if lookObject == 'бы':
            rule_19 = -1
        rules['rule_19'] = rules['rule_19'] + rule_19
        
        #20 - предлог "не" чаще плохо
        rule_20 = 0
        if lookObject == 'бы':
            rule_20 = -1
        rules['rule_20'] = rules['rule_20'] + rule_20
        
        #21 - отсутствие "никто" чаще плохо
        rule_21 = 0
        if lookObject == 'никто':
            rule_21 = -1
        rules['rule_21'] = rules['rule_21'] + rule_21
        
        #22 - интерес это хорошо
        rule_22 = 0
        rule_22_dict = ['интерес', ' интересно', 'вау', 'ого','расскажи','очень']
        if lookObject in rule_22_dict:
            rule_22 = 1
        rules['rule_22'] = rules['rule_22'] + rule_22
        
        #23 - нужен это хорошо
        rule_23 = 0
        rule_23_dict = ['нужен','нужна','нужны']
        if lookObject in rule_23_dict:
            rule_23 = 1
        rules['rule_23'] = rules['rule_23'] + rule_23
        
        #24 - действие это хорошо
        rule_24 = 0
        rule_24_dict = ['гоу' , 'го' , 'пошли' , 'поход' , 'путешествие' , 'путешествовать' , 'пойдем']
        if lookObject in rule_24_dict:
            rule_24 = 1
        rules['rule_24'] = rules['rule_24'] + rule_24
        
        #25 - отсутствие действия это плохо
        rule_25 = 0
        rule_25_dict = ['стой','остановись','сядь']
        if lookObject in rule_24_dict:
            rule_25 = -1
        rules['rule_25'] = rules['rule_25'] + rule_25
        
        #26 - суффикс повелительный это плохо
        rule_26 = 0
        rule_26_0_dict = ['рди', 'тят' , 'нишь']
        rule_26_dict = ['скинь', 'сделай', 'ставь', 'посмотри', 'подтверди',  'ответят',  'звони', 'старайся', 'возьми'
                        ,'купи', 'отнеси', 'забери', 'скажешь', 'передашь', 'сделаешь', 'следи', 'ответь', 'назовешь'
                        ,'ответь', 'позвони',  'переведешь', 'скинь', 'ищи', 'явиться', 'перезвони', 'едь'
                        ,'спроси', 'пометь', 'напиши', 'пиши', 'вырывай', 'скажи', 'вызови', 'выйди', 'ответь', 'отвечай'
                        ,'позвони', 'забудь', 'наберешь', 'будь', 'сходи', 'разберись', 'отвези', 'пополни', 'принеси', 'подай' 
                        ,'позвони', 'спеши', 'переведи', 'попроси']
        if lookObject in rule_26_dict:
            rule_26 = -1

        elif len(lookObject) > 3:
            if lookObject[-3:] in rule_26_0_dict:
                rule_26 = -1
        rules['rule_26'] = rules['rule_26'] + rule_26
        
        #27 - быстрое действие это хорошо перед нейтральным словом это плохо
        rule_27 = 0
        rule_27_dict = ['быстро' , 'сразу' , 'мигом' , 'срочно']
        if lookObject in rule_27_dict:
            rule_27 = -1
        rules['rule_27'] = rules['rule_27'] + rule_27
        
        #28 - согласие это хорошо
        rule_28 = 0
        rule_28_dict = ['ладно', 'хорошо', 'договорились', 'успешно', 'приму', 'да', 'ок', 'понятно', 'круто', 'давай', 'окей']
        if lookObject in rule_28_dict:
            rule_28 = 1
        rules['rule_28'] = rules['rule_28'] + rule_28
        
        #29 - наказание это плохо
        rule_29 = 0
        rule_29_dict = ['издеваешься', 'наказание', 'наказан' ]
        if lookObject in rule_29_dict:
            rule_29 = -1
        rules['rule_29'] = rules['rule_29'] + rule_29
        
        #30 - благодарность это хорошо
        rule_30 = 0
        rule_30_dict = ['спасибо', 'благодарю', 'молодец', 'спс']
        if lookObject in rule_30_dict:
            rule_30 = 1
        rules['rule_30'] = rules['rule_30'] + rule_30
        
        #31 - жаргон это плохо
        rule_31 = 0
        rule_31_dict = ['нал', 'бабки', 'тварь', 'фига', 'трубу', 'охренел']
        if lookObject in rule_31_dict:
            rule_31 = -1
        rules['rule_31'] = rules['rule_31'] + rule_31
        
        #32 - отдых это хорошо
        rule_32 = 0
        rule_32_dict = ['отдых' , 'отдыхаем' ,'расслаблены','расслаблен','кайф','курорт','отпуск'
                       ,'выходные', 'путешествие',  'туризм' ]
        if lookObject in rule_32_dict:
            rule_32 = 1
        rules['rule_32'] = rules['rule_32'] + rule_32
        
        #33 - пожелания это хорошо
        rule_33 = 0
        rule_33_dict = ['спокойной', 'доброй', 'хорошей', 'замечательной', 'сладких', 'целую'
                        , 'поправляйся', 'выздаравливай', 'крепись', 'держись', 'постарайся' ]
        if lookObject in rule_33_dict:
            rule_33 = 1
        rules['rule_33'] = rules['rule_33'] + rule_33
        
        #34 - мольба это хорошо
        rule_34 = 0
        rule_34_dict = ['пожалуйста', 'помоги', 'спаси', 'прости', 'извини', 'помочь', 'вина', 'прошу', 'плиз']
        if lookObject in rule_34_dict:
            rule_34 = 1
        rules['rule_34'] = rules['rule_34'] + rule_34
        
        #35 - радостные слова
        rule_35 = 0
        rule_35_dict = ['приколюх', 'ржем', 'смешно']
        if lookObject in rule_35_dict:
            rule_35 = 1
        rules['rule_35'] = rules['rule_35'] + rule_35
        
        #36 - Уменьшительные имена это хорошо
        rule_36 = 0
        rule_36_dict = ['димон', 'саня', 'аленка', 'натуль', 'дашка', 'братан', 'ань', 'наташ', 'ната'
                        , 'миха', 'жека', 'женька', 'катин', 'сашь', 'катька', 'сашка', 'сашуля', 'сашуль'
                        ,'мих', 'денчик', 'леха', 'лех', 'алеш', 'дашка', 'дашкой', 'илюха', 'илюх','саня'
                        ,'санек', 'гришуля', 'кирюша',  'илюха', 'дань', 'саш', 'марин', 'даня', 'данилка'
                        ,'дань', 'толик', 'толян', 'жень', 'женька', 'ден', 'наташ','наташа', 'бро', 'макс'
                        , 'сынок', 'сереж', 'сережа', 'женек', 'димка', 'димон','братан', 'братиш', 'братишка', 'серега']
        if lookObject in rule_36_dict:
            rule_36 = 1
        rules['rule_36'] = rules['rule_36'] + rule_36
        
        #37 - скука это хорошо
        rule_37 = 0
        rule_37_dict = ['скучаю', 'жалею', 'волнуюсь', 'желаю']
        if lookObject in rule_37_dict:
            rule_37 = 1
        rules['rule_37'] = rules['rule_37'] + rule_37
        
        #38 - волнение это плохо
        rule_38 = 0
        rule_38_dict = ['переживай', 'жалей', 'волнуйся', 'страдай', 'болей']
        if lookObject in rule_38_dict:
            rule_38 = -1
        rules['rule_38'] = rules['rule_38'] + rule_38
        
        #39 - предлог некогда плохой
        rule_39 = 0
        if lookObject == 'некогда':
            rule_39 = -1
        rules['rule_39'] = rules['rule_39'] + rule_39
        
        #40 - наименование животным это хорошо
        rule_40 = 0
        rule_40_dict = ['зай', 'кролик', 'котик', 'тигр', 'зайка', 'зайчонок', 'пушистик'
                        ,'рыбка', 'зайчик', 'киса', 'заяц', 'котенок']
        if lookObject in rule_40_dict:
            rule_40 = 1
        
        rules['rule_40'] = rules['rule_40'] + rule_40
        
        #41 - отказ это плохо
        rule_41 = 0
        rule_41_dict = ['нет', 'отрицаю', 'неа']
        if lookObject in rule_41_dict:
            rule_41 = -1
        rules['rule_41'] = rules['rule_41'] + rule_41
        
        #42 - сочетание знаков !? или ?! это плохо
        rule_42 = 0
        rule_42 = -1 * (lookObject.count('?!') + lookObject.count('!?'))
        rules['rule_42'] = rules['rule_42'] + rule_42
        
        #43 - обязательства это плохо
        rule_43 = 0
        rule_43_dict = ['нужно', 'обязан', 'должен', 'должна', 'долг']
        if lookObject in rule_43_dict:
            rule_43 = -1
        rules['rule_43'] = rules['rule_43'] + rule_43
        
        
        #44 - злось это плохо
        rule_44 = 0
        rule_44_dict = ['злись', 'жестоко', 'плохо', 'скуп', 'тягость', 'бред', 'бесят', 'раздражают', 'стресс'
                        ,  'пофиг', 'переживать' ]
        if lookObject in rule_44_dict:
            rule_44 = -1
        rules['rule_44'] = rules['rule_44'] + rule_44
        
        #45 - добрые пожелания это хорошо
        rule_45 = 0
        rule_45_dict = ['удачи', 'счастья', 'поздравляю', 'здоровья', 'любви', 'страсти', 'здоровья', 'сердечно'
                        , 'желаю', 'дорог', 'подравляем', 'поздравление']
        if lookObject in rule_45_dict:
            rule_45 = 1
        rules['rule_45'] = rules['rule_45'] + rule_45
        
        #46 - оскорбления это плохо
        rule_46 = 0
        rule_46_dict = ['тупой', 'дурак', 'идиот', 'сука', 'проститутка', 'позор']
        if lookObject in rule_46_dict:
            rule_46 = -1
        rules['rule_46'] = rules['rule_46'] + rule_46
        
        #47 - начало предложения с "И" или "АУ" это плохо
        rule_47 = 0
        rule_47_dict = ['и' ,'ау']
        
        if Step == 0 and lookObject in rule_47_dict:
            rule_47 = -1
        
        rules['rule_47'] = rules['rule_47'] + rule_47
        
        #48 - траты это плохо
        rule_48 = 0
        rule_48_dict = ['траты', 'трать', 'растраты', 'потеря', 'утрата', 'убыток', 'минус']
        if lookObject in rule_48_dict:
            rule_48 = -1
        rules['rule_48'] = rules['rule_48'] + rule_48
        
        #49 - обман это плохо
        rule_49 = 0
        rule_49_dict = ['жулик', 'обман', 'шулер', 'мошенник', 'плевать', 'обманул', 'обманщик'
                        ,'обманщица', 'наплевал', 'заблевал']
        if lookObject in rule_49_dict:
            rule_49 = -1
        rules['rule_49'] = rules['rule_49'] + rule_49
        
        #50 - свобода это хорошо
        rule_50 = 0
        rule_50_dict = ['свободна', 'освободилась']
        if lookObject in rule_50_dict:
            rule_50 = 1
        rules['rule_50'] = rules['rule_50'] + rule_50
        
        #51 - упоминание стран курорта это хорошо
        rule_51 = 0
        rule_51_dict = ['турция', 'корсика', 'монако', 'милан', 'тунис', 'вьетам', 'гоа', 'кипр', 'египет', 'китай', 'япония']
        if lookObject in rule_51_dict:
            rule_51 = 1
        rules['rule_51'] = rules['rule_51'] + rule_51
        
        #52 - комбинация ахахаха это хорошо
        rule_52 = 0
        if lookObject.count('ха') >= 2:
            rule_52 = 1
        rules['rule_52'] = rules['rule_52'] + rule_52
        
        #53 - не это плохо
        rule_53 = 0
        if lookObject == 'не':
            rule_53 = -1
        rules['rule_53'] = rules['rule_53'] + rule_53
        
        #53.1 - ничего перед не убирает отрицание не
        if lookObject == 'ничего' and prevWord == 'не':
            rules['rule_53'] = rules['rule_53'] + 1
            
        #53.2 - или перед не и нет убирает плохое состояние
        if lookObject == 'или' and prevWord == 'не':
            rules['rule_53'] = rules['rule_53'] + 1
            
        #54 - радость это хорошо
        rule_54 = 0
        rule_54_dict = ['супер', 'класс', 'прекрасно', 'клас', 'великолепно', 'радость', 'вау', 'волшебно'
                        ,'чудесно', 'оболденно', 'обалденно', 'ачуметь',  'ура', 'бомба', 'праздник', 'праздником']
        if lookObject in rule_54_dict:
            rule_54 = 1
        rules['rule_54'] = rules['rule_54'] + rule_54
        
        #55 - когда что-либо нравится это хорошо
        rule_55 = 0
        rule_55_dict = ['понравилось', 'нравится', 'понравится', 'нравилось', 'классно', 'класно', 'понравились', 'нравились' ]
        if lookObject in rule_55_dict:
            rule_55 = 1
        rules['rule_55'] = rules['rule_55'] + rule_55
        

        
        
        
        #запоминаем информацию по текущему слову
        prevWord = lookObject
        result = 0
        for value in rules.values():
            result = result + value

        
        #переходим к следующему элементу
        Step = Step - 1
    
    final_rule = 0
    for key in rules.keys():
        if key != 'rule_18':
            final_rule = final_rule + rules[key]
    rules['final_rule'] = final_rule
    

    for key in rules.keys():
        allRules[key] = np.append(allRules[key] , rules[key])
    #для теста
    #iii += 1
    #if iii == 2:
    #    break
    


# In[5]:

params = pd.DataFrame.from_dict(allRules)
result = pd.concat([df_start, params], axis=1, join='inner')


# In[6]:


result['GoB'] = [1 if x > 0 else 0 if x == 0 else -1 for x in result['final_rule']]


# In[17]:

columns = ['GoB' , 'text' , 'final_rule']
X = result.drop(columns,axis = 1)
y = result['GoB'].values


# # Создаем модельки

# In[53]:

from sklearn import cross_validation, grid_search, linear_model, metrics
from sklearn.linear_model import SGDClassifier as SGD

classifier = SGD()
classifier

parameters_grid = {
    'loss' : ['log'],
    'penalty' : ['l1'],
    'n_iter' : [2000],
    'alpha' : [0.000775],
    'learning_rate' : ['optimal'],
    'eta0' : [0.0001],
}


# In[54]:

train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(X, y, 
                                                                                     test_size = 0.3)


# In[55]:

cv = cross_validation.StratifiedShuffleSplit(train_labels, n_iter = 10, test_size = 0.2, random_state = 15)


# In[56]:

grid_cv = grid_search.GridSearchCV(classifier, parameters_grid, scoring = 'accuracy', cv = cv)


# In[57]:

get_ipython().run_cell_magic('time', '', '#в процессе обучения оценивается качество в каждой из точек и в итоге выбирается оптимальное\ngrid_cv.fit(train_data, train_labels)')


# In[58]:

#самый лучший классификатор можно отобразить с помощью команды best_estimator
#нам выводиться модель с наилучшими параметрами
best_model = grid_cv.best_estimator_
best_model


# In[59]:

#выведем наилучший результат нашей оценки и наилучшие параметры
print ('точность: '+str(round(grid_cv.best_score_,4)*100)+'%')
print (grid_cv.best_params_)


# In[81]:

best_model.coef_


# # Тестирование

# In[107]:

i = 9
print ('-1:' , sum(test_data.iloc[i].values * best_model.coef_[0]) + best_model.intercept_[0])
print ('0:' , sum(test_data.iloc[i].values * best_model.coef_[1]) + best_model.intercept_[1])
print ('1:' , sum(test_data.iloc[i].values * best_model.coef_[2]) + best_model.intercept_[2])


# In[100]:

best_model.predict_proba(test_data)[9:10]


# In[61]:

print (test_labels[:100])


# In[94]:

print (best_model.predict(test_data)[:100])


# In[119]:

#итоговая система уравнений:
formula1 = ''
formula2 = ''
formula3 = ''
i = 0
for key in X.columns:
    formula1 = formula1 + key + '*' + str(round(best_model.coef_[0][i],2)) + ' + '
    i += 1
formula1 = formula1 + str(round(best_model.intercept_[0],2))
i = 0
for key in X.columns:
    formula2 = formula2 + key + '*' + str(round(best_model.coef_[1][i],2)) + ' + '
    i += 1
formula2 = formula2 + str(round(best_model.intercept_[1],2))
i = 0
for key in X.columns:
    formula3 = formula3 + key + '*' + str(round(best_model.coef_[2][i],2)) + ' + '
    i += 1
formula3 = formula3 + str(round(best_model.intercept_[2],2))


print ('bad =' + formula1)
print ('neutral =' + formula2)
print ('good =' + formula3)
#максимальное значение из 3-х есть значение линейной модели одного из трех классов
#в текущей реализации не нормированно для максимальной простоты работы без питона
#итог
#bad =rule_1*-30.02 + rule_10*-6.77 + rule_11*-15.74 + rule_12*0.0 + rule_13*-9.77 + rule_14*-22.49 + rule_15*-21.84 + rule_16*0.0 + rule_17*-24.07 + rule_18*0.0 + rule_19*-7.4 + rule_2*-63.19 + rule_20*-7.4 + rule_21*0.0 + rule_22*0.0 + rule_23*-13.81 + rule_24*0.0 + rule_25*0.0 + rule_26*-16.25 + rule_27*-28.04 + rule_28*-26.14 + rule_29*0.0 + rule_3*-11.02 + rule_30*-10.66 + rule_31*0.0 + rule_32*0.0 + rule_33*0.0 + rule_34*-21.37 + rule_35*0.0 + rule_36*-11.4 + rule_37*0.0 + rule_38*0.0 + rule_39*0.0 + rule_4*-26.55 + rule_40*-1.93 + rule_41*-24.08 + rule_42*-16.02 + rule_43*-15.82 + rule_44*0.0 + rule_45*-0.91 + rule_46*-6.95 + rule_47*-9.36 + rule_48*0.0 + rule_49*0.0 + rule_5*-17.3 + rule_50*0.0 + rule_51*0.0 + rule_52*0.0 + rule_53*-16.45 + rule_54*0.0 + rule_55*0.0 + rule_6*0.0 + rule_7*-32.67 + rule_8*0.0 + rule_9*0.0 + -6.79
#neutral =rule_1*-1.69 + rule_10*2.68 + rule_11*-1.88 + rule_12*0.0 + rule_13*0.0 + rule_14*-2.21 + rule_15*-2.6 + rule_16*0.0 + rule_17*0.29 + rule_18*-0.02 + rule_19*0.59 + rule_2*0.24 + rule_20*0.59 + rule_21*0.0 + rule_22*-3.9 + rule_23*0.0 + rule_24*0.0 + rule_25*0.0 + rule_26*2.09 + rule_27*1.32 + rule_28*-2.43 + rule_29*0.0 + rule_3*-3.13 + rule_30*-2.84 + rule_31*0.0 + rule_32*-0.61 + rule_33*-2.35 + rule_34*0.0 + rule_35*0.0 + rule_36*-2.89 + rule_37*0.0 + rule_38*0.0 + rule_39*0.0 + rule_4*-2.5 + rule_40*-1.94 + rule_41*2.36 + rule_42*0.0 + rule_43*1.58 + rule_44*0.0 + rule_45*-10.52 + rule_46*1.76 + rule_47*1.16 + rule_48*0.0 + rule_49*0.0 + rule_5*2.72 + rule_50*0.0 + rule_51*0.0 + rule_52*-1.46 + rule_53*2.33 + rule_54*0.0 + rule_55*0.0 + rule_6*0.0 + rule_7*-81.91 + rule_8*0.0 + rule_9*0.0 + 1.66
#good =rule_1*33.36 + rule_10*0.0 + rule_11*28.51 + rule_12*0.0 + rule_13*9.93 + rule_14*35.53 + rule_15*19.94 + rule_16*0.0 + rule_17*21.97 + rule_18*0.2 + rule_19*13.71 + rule_2*46.51 + rule_20*13.71 + rule_21*0.0 + rule_22*11.16 + rule_23*6.95 + rule_24*0.0 + rule_25*0.0 + rule_26*29.46 + rule_27*26.89 + rule_28*33.41 + rule_29*0.0 + rule_3*34.74 + rule_30*25.63 + rule_31*0.0 + rule_32*4.55 + rule_33*14.36 + rule_34*17.18 + rule_35*0.0 + rule_36*36.41 + rule_37*0.0 + rule_38*0.0 + rule_39*0.0 + rule_4*46.08 + rule_40*17.68 + rule_41*29.07 + rule_42*11.21 + rule_43*4.61 + rule_44*0.0 + rule_45*15.56 + rule_46*0.0 + rule_47*2.56 + rule_48*0.0 + rule_49*0.0 + rule_5*2.68 + rule_50*0.0 + rule_51*0.0 + rule_52*8.18 + rule_53*40.48 + rule_54*0.0 + rule_55*2.83 + rule_6*0.0 + rule_7*54.61 + rule_8*0.0 + rule_9*0.0 + -10.99



# # Мысли
# 
# 1. смотреть не справочник а косинусное расстояние до каждого слова справочника
# 2. расширить справочники с помощью библиотек гугла
# 3. раскладывать слово по частям корня, суффикса, приставки и т.д. и анализировать корни и усилители
# 4. вынести все справочники в библиотеку
# 5. вынести функцию расчета косинусного расстояния в библиотеку
# 6. Вынести функцию разложения на части слова в библиотеку

# In[ ]:



