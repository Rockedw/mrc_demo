# -*- coding: utf-8 -*-
# coding=utf-8
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import csv
from fastapi import FastAPI
from typing import Optional

parser = argparse.ArgumentParser(
        description="you should add those parameter")
    parser.add_argument('--config', type=str, default="./config.json", help="The path of config file")
    parser.add_argument('--input', type=str, help='Please input data which you want to predict', required=True)
    arguments = parser.parse_args()

print(arguments.config)
    models = []
    with open(arguments.config, encoding='utf-8', mode='r') as f:
        models = json.load(f)
    model_path = models['model_path']
    print(model_path)
    print('hellow mlflow')


model_names = [
    # 'hfl/chinese-pert-large',
    # 'hfl/chinese-pert-base',
    # 'hfl/chinese-roberta-wwm-ext-large',
    # 'hfl/chinese-roberta-wwm-ext',
    # 'hfl/chinese-bert-wwm-ext',
    # 'hfl/chinese-bert-wwm',
    # 'hfl/rbt3',
    # 'hfl/rbtl3',
    # 'uer/roberta-base-chinese-extractive-qa',
    'luhua/chinese_pretrain_mrc_roberta_wwm_ext_large',
    # 'luhua/chinese_pretrain_mrc_macbert_large',
    # 'wptoux/albert-chinese-large-qa',
    # 'yechen/question-answering-chinese',
    # 'liam168/qa-roberta-base-chinese-extractive'
]

# tokenizer = AutoTokenizer.from_pretrained(
#     "luhua/chinese_pretrain_mrc_roberta_wwm_ext_large")
#
# model = AutoModelForQuestionAnswering.from_pretrained(
#     "luhua/chinese_pretrain_mrc_roberta_wwm_ext_large")

tokenizer = AutoTokenizer.from_pretrained(
    model_path['mrc_roberta_wwm_ext_large'])

model = AutoModelForQuestionAnswering.from_pretrained(
   model_path['mrc_roberta_wwm_ext_large'])

QA = pipeline('question-answering', model=model, tokenizer=tokenizer, device=0)


def question_to_context(context):
    if not len(context)>0:
        return
    appeared = []
    answer_dict = {}
    print(context)
    parent_dict = QA({'question': '这是什么功能得描述?', 'context': context})
    parent = ''
    if parent_dict['score'] >= 0:
        parent = parent_dict['answer']
    if len(parent) > 0:
        appeared.append(parent)
        answer_dict['parent'] = parent
        children_list = []
        flag = True
        child_question = "满足'"+parent + "'有什么前提？"
        while flag:
            print(child_question)
            child_dict = QA({'question': child_question, 'context': context})
            if child_dict['score'] >= 0 and len(child_dict['answer']) > 0 and child_dict['answer'] not in appeared:
                appeared.append(child_dict['answer'])
                child = {'context': '', 'value': child_dict['answer']}
                child_question = '满足'+parent+'中除了'+child_dict['answer'] + '的下一个前提是什么?'
                print(child_dict['answer'] + '需要满足什么条件?')
                context_dict = QA({'question': child_dict['answer'] + '需要满足什么条件?', 'context': context})
                if context_dict['score'] >= 0:
                    child['context'] = context_dict['answer']
                children_list.append(child)
            else:
                flag = False
        answer_dict['children_and_context'] = children_list
        answer_dict['relation'] = ''
        if len(children_list) >= 2:
            print(children_list[0]['value'] + '和' + children_list[1]['value'] + '是并且还是或者的关系?')
            temp = QA({'question': children_list[0]['value'] + '和' + children_list[1]['value'] + '是并且还是或者的关系?',
                       'context': context})
            if temp['score'] >= 0:
                answer_dict['relation'] = temp['answer']
    print(answer_dict)
    return answer_dict


if __name__ == '__main__':
    # print(question_to_context('帮助游客需要包括游客遵循景点的规则并且在满足游客对某个景点感兴趣的前提下游客得到某处景物的介绍。'))

    paragraph = '游客子系统主要的目的是帮助游客。帮助游客的条件有：用户遵循景点的规则；游客得到了某处景物的介绍（前提是用对某个景点感兴趣）。用户需要遵循的景点规则有：在规定时间内游客进入景点；通知用户登记（在判断满足用户未登记的前提下）。在规定时间内用户进入景点需要在满足C2的前提下判断是否在景点关闭前通知用户。在景点关闭前通知游客需要判断：1.在满足了游客尚未进入景点的前提下，是否通知游客不许进入景点；2.在用户已经进入景点的前提下，是否通知用户已该离开。游客得到某处景物的介绍需要至少满足三种情况之一：1.在C4条件下游客通过电子向导获得景物介绍；2.在C5条件下游客通过手机应用获得景物介绍；3.在C6条件下游客通过导游获得景物介绍。游客通过电子向导获得景物介绍的前提是用户已经到达电子向导处并且游客知道如何使用电子向导。游客通过手机应用获得景物介绍的前提是游客手机中有向导应用并且向游客提供合适的信息。用户到达电子向导处要判断在C9条件下是否有地标指引游客到电子游客处。游客知道如何使用电子向导要判断在C15条件下电子向导使用demo是否得到展示。游客手机中有向导应用要判断在C16条件下是否粘贴下载了应用的二维码。向游客提供合适的信息要判断景物信息是否准备好。准备景物信息要判断景物的详细信息或者简短信息是否准备好。 '
    # paragraph = '在游客子系统下，主要的目的使帮助游客。帮助游客需要包括游客遵循景点的规则并且在满足游客对某个景点感兴趣的前提下游客得到某处景物的介绍。游客遵守景点规则包括游客在规定时间内进入景点并且判断是否在满足游客未登记的前提下通知游客登记。游客在规定时间内进入景点需要判断在满足C2的前提下是否在景点关闭前通知游客。在景点关闭前通知游客包括判断在满足游客为进入景点的前提下是否通知游客不许进入景点并且判断在满足游客已进入景点的前提下是否通知游客该离开了。游客得到某处景物的介绍需要满足在C4条件下游客通过电子向导获得景物介绍或者在满足C5条件下游客通过手机应用获得景物介绍或者在满足C6前提下游客通过导游获得景物介绍3种情况中的至少一种。游客通过电子向导获得景物介绍需要满足游客到达电子向导处并且游客知道如何使用电子向导。游客通过手机应用获得景物介绍需要满足游客手机中有向导应用而且向游客提供合适的信息。游客到达电子向导处需要判断在满足C9的前提下是否有地表主因游客到电子向导处。游客知道如何使用电子向导需要判断在C15前提下是否展示电子向导使用demo。游客手机中有向导应用需要判断在满足C16的前提下是否粘贴下载应用的二维码。向游客提供适合的信息需要判断是否准备景物信息。准备景物信息需要判断是否准备景物的详细信息或者准备景物的简短信息。 '
    with open(r'./test2.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        header = ['parent', 'child', 'child_type', 'relation', 'context']
        writer.writerow(header)
        for sentence in paragraph.split('。'):
            print(sentence)
            data = question_to_context(sentence)
            if 'children_and_context' in data:
                for child in data['children_and_context']:
                    row = [data['parent'], child['value'], 'ChildTarget', data['relation'], child['context']]
                    writer.writerow(row)

    f.close()
