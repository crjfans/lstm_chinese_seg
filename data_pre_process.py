# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import time

# 以字符串的形式读入所有数据
with open('msr_train.txt', 'rb') as inp:
    texts = inp.read().decode('gbk')
sentences = texts.split('\r\n')  # 根据换行切分


# 将不规范的内容（如每行的开头）去掉
def clean(s):
    if u'“/s' not in s:  #
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s


texts = u''.join(map(clean, sentences))  # 把所有的词拼接起来
print('Length of texts is %d' % len(texts))
print('Example of texts: \n', texts[:300])
sentences = re.split(u'[，。！？、‘’“”]/[bems]', texts)
print('Sentences number:', len(sentences))
print('Sentence Example:\n', sentences[1])

def get_Xy(sentence):
    """将 sentence 处理成 [word1, w2, ..wn], [tag1, t2, ...tn]"""
    words_tags = re.findall('(.)/(.)', sentence)
    if words_tags:
        words_tags = np.asarray(words_tags)
        words = words_tags[:, 0]
        tags = words_tags[:, 1]
        return words, tags # 所有的字和tag分别存为 data / label
    return None

datas = list()
labels = list()
print('Start creating words and tags data ...')
for sentence in tqdm(iter(sentences)):
    result = get_Xy(sentence)
    if result:
        datas.append(result[0])
        labels.append(result[1])

print('Length of datas is %d' % len(datas))
print('Example of datas: ', datas[0])
print('Example of labels:', labels[0])

# 句子长度的分布
df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
#　句子长度
df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))
df_data.head(2)
'''
import matplotlib.pyplot as plt
df_data['sentence_len'].hist(bins=100)
plt.xlim(0, 100)
plt.xlabel('sentence_length')
plt.ylabel('sentence_num')
plt.title('Distribution of the Length of Sentence')
plt.show()
'''

from itertools import chain
all_words = list(chain(*df_data['words'].values))
# 2.统计所有 word
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
set_ids = range(1, len(set_words)+1) # 注意从1开始，因为我们准备把0作为填充值
tags = [ 'x', 's', 'b', 'm', 'e']
tag_ids = range(len(tags))

# 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)
tag2id = pd.Series(tag_ids, index=tags)
id2tag = pd.Series(tags, index=tag_ids)
vocab_size = len(set_words)
print('vocab_size={}'.format(vocab_size))

#把 words 和 tags 都转为数值 id
max_len = 32

def X_padding(words):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    return ids

def y_padding(tags):
    """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
    ids = list(tag2id[tags])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    return ids

df_data['X'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)

# 最后得到了所有的数据
X = np.asarray(list(df_data['X'].values))
y = np.asarray(list(df_data['y'].values))
print('X.shape={}, y.shape={}'.format(X.shape, y.shape))
print('Example of words: ', df_data['words'].values[0])
print('Example of X: ', X[0])
print('Example of tags: ', df_data['tags'].values[0])
print('Example of y: ', y[0])
# 保存数据
import pickle
import os

if not os.path.exists('data/'):
    os.makedirs('data/')

with open('data/data.pkl', 'wb') as outp:
    pickle.dump(X, outp)
    pickle.dump(y, outp)
    pickle.dump(word2id, outp)
    pickle.dump(id2word, outp)
    pickle.dump(tag2id, outp)
    pickle.dump(id2tag, outp)
print('** Finished saving the data.')
