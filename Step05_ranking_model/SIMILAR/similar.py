import graphlab
import numpy as np
import pandas as pd

def writeDict(dict, filename, sep):
    with open(filename, "w") as f:
#        for key in dict.keys():            
#            f.write(str(key) + ":" + str(dict[key]) + "\n")
        for key, value in sorted( dict.items() ):
            f.write( str(key) + sep + ','.join(str(x) for x in value) + '\n' )
            
def readDict(filename, sep):
    with open(filename, "r") as f:
        dict = {}
        for line in f:
            values = line.split(sep)
            key = values[0]
            value = values[1].split(',')
            #dict[values[0]] = {int(x) for x in values[1:len(values)]}
            dict[key] = value
        return(dict)

def Get_Similar_SelectedID(selected_qid, similar_num=5) :
    Similar_Qid_List = {}
    for i in range( len(selected_qid) ) :
        key = 'Q' + str(i) + '_Similar_List'
        Q_i = selected_qid[i:i+1]
        value = knn_model.query(Q_i, k = similar_num)['reference_label']
        Similar_Qid_List[key] = value
    return Similar_Qid_List
    
    
question_content = graphlab.SFrame.read_csv('./Question_Content.csv')

question_content['text'] = question_content['tag'] + question_content['title']
question_content['word_count'] = graphlab.text_analytics.count_words(question_content['text'])
tfidf = graphlab.text_analytics.tf_idf(question_content['word_count'])
question_content['tfidf'] = tfidf

# Build KNN MODEL
knn_model = graphlab.nearest_neighbors.create(question_content, features=['tfidf'], label='question_id')

selected = graphlab.SFrame.read_csv('./Selected.csv')

# k is the num of similar questions
def Get_Similar_SelectedID(selected, similar_num=5) :
    Similar_Qid_List = {}
    for i in range( len(selected) ) :
        key = 'Q' + str(i) + '_Similar_List'
        Q_i = selected[i:i+1]
        value = knn_model.query(Q_i, k = similar_num)['reference_label']
        Similar_Qid_List[key] = value
    return Similar_Qid_List

Similar_Qid_List = Get_Similar_SelectedID(selected, similar_num = 10)

writeDict(Similar_Qid_List, './Similar_Qid.csv', ':')









