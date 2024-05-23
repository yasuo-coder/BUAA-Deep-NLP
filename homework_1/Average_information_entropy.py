import os
import math
import jieba
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def read_multiple_txt_files(directory_path):
    all_texts = []

    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory_path):
        # 检查文件是否是txt文件
        if filename.endswith('.txt'):
            # 构建文件的完整路径
            file_path = os.path.join(directory_path, filename)

            # 读取文件内容
            with open(file_path, 'r', encoding='gb18030') as file:
                all_texts += file.read()
            #     text = str(all_texts)
            #
            # calculate_model_entropies_base_character(filename, text)
            # calculate_model_entropies_base_word(filename, text)

    return all_texts

# 得到单个词的词频表
def get_unigram_tf(word):
    unigram_tf = {}
    for w in word:
        unigram_tf[w] = unigram_tf.get(w, 0) + 1
    return unigram_tf

# 得到二元词的词频表
def get_bigram_tf(word):
    bigram_tf = {}
    for i in range(len(word) - 1):
        bigram = (word[i], word[i + 1])
        bigram_tf[bigram] = bigram_tf.get(bigram, 0) + 1
    return bigram_tf

# 得到三元词的词频表
def get_trigram_tf(word):
    trigram_tf = {}
    for i in range(len(word) - 2):
        trigram = (word[i], word[i + 1], word[i + 2])
        trigram_tf[trigram] = trigram_tf.get(trigram, 0) + 1
    return trigram_tf

# 计算一元模型的信息熵
def calc_entropy_unigram(word, is_ci):
        word_tf = get_unigram_tf(word)
        word_len = sum(word_tf.values())
        entropy = sum(-freq / word_len * math.log(freq / word_len, 2) for freq in word_tf.values())
        if is_ci:
            print("基于词的一元模型的中文信息熵为：{}比特/词".format(entropy))
        else:
            print("基于字的一元模型的中文信息熵为：{}比特/字".format(entropy))
        return entropy

# 计算二元模型的信息熵
def calc_entropy_bigram(word, is_ci):
        if is_ci:
            word_tf = get_bigram_tf(word)
            last_word_tf = get_unigram_tf(word[:-1])
            bigram_len = sum(word_tf.values())
            entropy = sum(
                -freq / bigram_len * math.log(freq / last_word_tf[bigram[0]], 2) for bigram, freq in word_tf.items())
            print("基于词的二元模型的中文信息熵为：{}比特/词".format(entropy))
        else:
            word_tf = get_bigram_tf(word)
            last_word_tf = get_unigram_tf(word[:-1])
            bigram_len = sum(word_tf.values())
            entropy = sum(
                -freq / bigram_len * math.log(freq / last_word_tf[bigram[0]], 2) for bigram, freq in word_tf.items())
            print("基于字的二元模型的中文信息熵为：{}比特/字".format(entropy))
        return entropy

# 计算三元模型的信息熵
def calc_entropy_trigram(word, is_ci):
        word_tf = get_trigram_tf(word)
        last_word_tf = get_bigram_tf(word[:-1])
        trigram_len = sum(word_tf.values())
        entropy = sum(-freq / trigram_len * math.log(freq / last_word_tf[(trigram[0], trigram[1])], 2) for trigram, freq in word_tf.items())
        if is_ci:
            print("基于词的三元模型的中文信息熵为：{}比特/词".format(entropy))
        else:
            print("基于字的三元模型的中文信息熵为：{}比特/字".format(entropy))
        return entropy

# 调用函数
texts_directory_path = 'D:\\PyCharmProject\\BUAA\\NLP\\小作业1\\中文语料库' 
txt = read_multiple_txt_files(texts_directory_path)
text = str(txt) # 类型转换

# 使用jieba进行中文分词
words = jieba.lcut(text)
# 计算词频
word_counts = Counter(words)
# 加入要去除的标点符号
extra_characters = {"，", "。", '\n', "“", "”", "：", "；", "？", "（", "）", "！", "…", '》', '《'}
# 去除标点符号
for word in extra_characters:
    del word_counts[word]

# 计算基于词的
calc_entropy_unigram(words, True)
calc_entropy_bigram(words, True)
calc_entropy_trigram(words, True)

# 计算基于字的
calc_entropy_unigram(text, False)
calc_entropy_bigram(text, False)
calc_entropy_trigram(text, False)

