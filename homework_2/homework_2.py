import os
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import openpyxl

def content_deal(content):  # 语料预处理，进行断句，去除一些广告和无意义内容
    content = pd.Series(content)[pd.Series(content).apply(len) > 0]  # 去除长度为0的词
    stopwords = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '新语丝电子文库', '：', '“', '！', '”', "\n", ",", "，", "。", "？", "、", "；", "（", "）", "《", "》", "…", "「", "」", "—", "～",
                 "【", "】", "……", "-", " ", "　"]
    content = content[~content.isin(stopwords)]
    final_words = content.tolist()
    return final_words

# 从指定文件夹读取文本数据，并抽取固定长度的段落
def load_data_and_sample(folder_path, sample_size=1000, paragraph_length=100):
    corpus = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='ANSI') as file:
                text = file.read().replace('\n', ' ')
                words = jieba.lcut(text)
                words = content_deal(words)
                for i in range(0, len(words), paragraph_length):
                    if i + paragraph_length <= len(words):
                        corpus.append(' '.join(words[i:i+paragraph_length]))
                        labels.append(filename[:-4])
    sampled_indices = np.random.choice(range(len(corpus)), sample_size, replace=False)
    sampled_corpus = [corpus[i] for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]
    return sampled_corpus, sampled_labels

# 执行LDA和使用随机森林进行分类和评估
def perform_lda_and_classification(corpus, labels, T, unit, n_splits=10):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, analyzer=unit, max_features=5000)
    lda = LatentDirichletAllocation(n_components=T, random_state=0)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    X = vectorizer.fit_transform(corpus)
    X_topics = lda.fit_transform(X)

    # 初始化十折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies_train = []
    accuracies_test = []

    # 执行交叉验证
    for train_index, test_index in kf.split(X_topics):
        X_train, X_test = X_topics[train_index], X_topics[test_index]
        y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]
        clf.fit(X_train, y_train)

        # 计算训练准确率
        y_pred_train = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        accuracies_train.append(train_accuracy)

        # 计算测试准确率
        y_pred_test = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        accuracies_test.append(test_accuracy)

    return np.mean(accuracies_train), np.mean(accuracies_test)


if __name__ == '__main__':
    path = 'D:/PyCharmProject/BUAA/NLP/小作业2/中文语料库'
    K = [20, 100, 500, 1000, 3000]
    T = [10, 25, 50, 100, 200, 500, 1000]

    Unit = ['word', 'char']
    # 初始化列表以存储所有结果
    results = []

    for unit in Unit:
        for k in K:
            for t in T:
                corpus, labels = load_data_and_sample(path, sample_size=1000, paragraph_length=k)
                # 处理数据并获取结果
                accuracies_train, accuracies_test = perform_lda_and_classification(corpus, labels, t, unit)
                print("when Unit = {:}, K = {:}, T = {:}, accuracies_train: {:.2f}, accuracies_test: {:.2f}".format(unit, k, t, accuracies_train, accuracies_test))
                # 将结果存储到列表中
                results.append({
                    'Unit': unit,
                    'K': k,
                    'T': t,
                    'Train Accuracy': accuracies_train,
                    'Test Accuracy': accuracies_test
                })

    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)

    # 保存结果到Excel文件
    results_df.to_excel('LDA_classification_results.xlsx', index=False)



