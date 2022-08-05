# coding=utf-8
import json
import random

import torch

import pandas as pd

print(torch.__version__)
print(torch.cuda.is_available())

o = json.loads('{"question": "用户遵循景点的规则的下一个子功能或任务是什么？","type": "semantic-role",'
               '"id": "f388cc7597def765543c1a6d00b19313","answers": [{"text": "","answer_start": -1}],'
               '"is_impossible": true}')
print((o['answers']))

data = pd.read_csv('./datasets/rd2.csv')

paragraph = {}
paragraphs = []
context = ''
temp = []
for i in range(len(data)):
    if context == data['context'][i]:
        m = {}
        temp.append(m)
        m['question'] = data['question'][i]
        m['type'] = 'vocab_noun'
        m['id'] = str(random.randint(1000000000000, 999999999999999))
        answers = []
        m['answers'] = answers
        m2 = {}
        if not type(data['answer'][i]) is float:
            print('here1')
            print(data['answer'][i])
            m2['text'] = data['answer'][i]
            m2['answer_start'] = context.find(data['answer'][i])
        else:
            m2['text'] = ''
            m2['answer_start'] = '-1'
        answers.append(m2)
        m['is_impossible'] = True
    else:
        paragraph1 = {}
        context = data['context'][i]
        paragraph1['context'] = context
        qas = []
        temp = qas
        m = {'question': data['question'][i], 'type': 'semantic-role',
             'id': str(random.randint(1000000000000, 999999999999999))}
        qas.append(m)
        answers = []
        m['answers'] = answers
        m2 = {}
        if not type(data['answer'][i]) is float:
            print('here2')
            print(data['answer'][i])
            m2['text'] = data['answer'][i]
            m2['answer_start'] = context.find(data['answer'][i])
        else:
            m2['text'] = ''
            m2['answer_start'] = '-1'
        answers.append(m2)
        m['is_impossible'] = True
        paragraph1['qas'] = qas
        paragraph1['title'] = 'title'
        paragraphs.append(paragraph1)
data = [{'paragraphs': paragraphs}]
res = {'data': data}


print(json.dumps(res, ensure_ascii=False, indent=4))

