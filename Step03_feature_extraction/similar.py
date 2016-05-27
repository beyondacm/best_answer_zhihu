import graphlab
import numpy as np
import pandas as pd

# Use graphlab to read question_content
question_content = graphlab.SFrame.read_csv('./Question_Content.csv')

question_content['text'] = question_content['tag'] +  question_content['title'] 
question_content['word_count'] = graphlab.text_analytics.count_words(question_content['text'])
tfidf = graphlab.text_analytics.tf_idf(question_content['word_count'])
question_content['tfidf'] = tfidf

knn_model = graphlab.nearest_neighbors.create(question_content, features=['tfidf'], label='question_id')


def writeDict(Dict, filename, sep) :
    with open(filename, 'w') as f:
        for key,value in sorted( Dict.items() ) :
            f.write( str(key) + sep + str(value) + '\n')



Similar_Qid_List = {}
# for i in range(10) :

for i in range( len(question_content) ) :

    key = 'Q' + str(i) + '_Similar_List'
    Q_i = question_content[i:i+1]
    value = knn_model.query(Q_i, k=3)['reference_label']
    Similar_Qid_List[key] = value

writeDict(Similar_Qid_List, './Similar_Qid.csv', ':')
