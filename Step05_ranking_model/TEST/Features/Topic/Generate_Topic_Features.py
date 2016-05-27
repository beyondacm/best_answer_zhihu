import graphlab
import pandas as pd

topic_model = graphlab.load_model('/home/zpgao/ML/Best_Answer/Zhihu/Step05_ranking_model/TEST/Features/Topic/zh_topic_model')

predict_docs = graphlab.SFrame.read_csv('/home/zpgao/ML/Best_Answer/Zhihu/Step05_ranking_model/TEST/Features/Topic/TEST_LDA_SOURCE.txt', header=False)
predict_docs = graphlab.text_analytics.count_words(predict_docs['X1'])
predict_docs = predict_docs.dict_trim_by_keys(graphlab.text_analytics.stopwords(), exclude=True)

pred = topic_model.predict(predict_docs, output_type='probability')
pred.save('/home/zpgao/ML/Best_Answer/Zhihu/Step05_ranking_model/TEST/Features/Topic/Raw_Topic_Features.csv', format='csv')

tf = pd.read_table('/home/zpgao/ML/Best_Answer/Zhihu/Step05_ranking_model/TEST/Features/Topic/Raw_Topic_Features.csv', sep = ' |"|\[|\]', header=None)
tf = tf.drop(tf.columns[[0, 1, 52, 53]], axis=1)
tf.to_csv('/home/zpgao/ML/Best_Answer/Zhihu/Step05_ranking_model/TEST/Features/Topic/Raw_Topic_Features.csv', encoding='utf-8', index = False, header=None)
