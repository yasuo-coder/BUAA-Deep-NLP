import os
import math
import jieba
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter



def read_multiple_txt_files(directory_path):
    # 存储所有文件内容的字典，键是文件名，值是文件内容
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

    return all_texts

# 定义一个函数来验证Zipf定律
def verify_zipfs_law(text):
    # 使用jieba进行中文分词
    words = jieba.lcut(text)
    # 计算词频
    word_counts = Counter(words)
    # 加入要去除的标点符号
    extra_characters = {"，", "。", '\n', "“", "”", "：", "；", "？", "（", "）", "！", "…", '》', '《'}
    # 去除标点符号
    for word in extra_characters:
        del word_counts[word]
    # 获取最常见的词及其频率，按频率降序排序
    common_words = word_counts.most_common()
    # 准备绘图
    ranks = [i for i in range(1, len(common_words) + 1)]
    frequencies = [freq for word, freq in common_words]
    # 绘制Zipf定律图
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, marker="o")
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title("Zipf's Law Verification")
    plt.savefig('./Zipf_Law.jpg')
    plt.show()

# 调用函数
texts_directory_path = 'D:\\PyCharmProject\\BUAA\\NLP\\小作业1\\中文语料库'  
txt = read_multiple_txt_files(texts_directory_path)
text = str(txt)     # 类型转换

# 验证Zipf定律
verify_zipfs_law(text)
