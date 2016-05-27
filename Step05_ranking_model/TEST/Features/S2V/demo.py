#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""

"""

import logging
import sys
import os
from word2vec import Word2Vec, Sent2Vec, LineSentence

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

# input_file = 'zh_text_source.csv'
# model = Word2Vec(LineSentence(input_file), size=50, window=5, sg=0, min_count=5, workers=8)
# model.save('S2V.model')
# model.save_word2vec_format(input_file + '.vec')

sent_file = '/home/zpgao/ML/Best_Answer/Zhihu/Step05_ranking_model/TEST/Features/S2V/TEST_S2V_SOURCE.txt'
model = Sent2Vec(LineSentence(sent_file), model_file='/home/zpgao/ML/Best_Answer/Zhihu/Step05_ranking_model/TEST/Features/S2V/S2V.model')
model.save_sent2vec_format(sent_file + '.vec')

program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)
