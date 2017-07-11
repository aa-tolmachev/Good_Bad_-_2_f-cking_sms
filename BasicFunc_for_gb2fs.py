

import boto
import boto.dynamodb
import codecs
import json
import re
import traceback
import pandas as pd
import time
from datetime import datetime
import sys
import base64
import requests
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import numpy as np
import ast
import boto.dynamodb2
from boto.dynamodb2.table import Table
from time import sleep


def prepare_clear_json(input_json_str):
    print ('START -> prepare_clear_json')
    data_json = json.loads(input_json_str)

    user_id = -1

    
    # print 'input json = ' + input_json_str

    new_json = []

    for dt in data_json['data']:
        new_elem = {}
        #for key, value in dt.iteritems(): <-устарел метод можно юзать для 2.7
        for key, value in iter(dt.items()):
            new_value = ''
            if value.get('S', 0) != 0:
                new_value = value['S']
            else:
                new_value = value['N']
            new_value = new_value.replace('"', '')
            new_value = new_value.replace('\r', '')
            new_value = new_value.replace('\n', '')
            new_value = new_value.replace('\'', '')
            new_value = new_value.replace('+', '')
            new_elem[key] = new_value

        new_json.append(new_elem)

    for dt in new_json:
        dt[u'sim_id1'] = data_json['sim']['sim_id1']
        dt[u'sim_id2'] = data_json['sim']['sim_id2']
        dt[u'sim_id3'] = data_json['sim']['sim_id3']
        # dt[u'user_id'] = u'None' if data_json['user_id'] == None else data_json['user_id']
        dt[u'user_id'] = user_id
        dt[u'device_id'] = data_json['device_id']
        dt[u'mobile_id'] = data_json['mobile_id']


    json_str_clear = str(new_json).decode('unicode_escape') 

    json_str_clear = re.sub("(?<=[\{ ,])u'|'(?=[:,\}])", '"', json_str_clear)

    done = False
    while not done:
        try:
            data_json_clear = json.loads(json_str_clear)
            done = True
        except ValueError:
            err = traceback.format_exc()
            symbol = err.find('char ')
            number = err[symbol+5:-2]
            print(json_str_clear[int(number)])
            json_str_clear = json_str_clear[:int(number)] + json_str_clear[int(number)+1:]

    print ('END -> prepare_clear_json')
    return json_str_clear

