import jieba
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from gensim.models import Word2Vec

matplotlib.rc("font", family='YouYuan')
from scipy.spatial.distance import cosine


# 读取多个txt文件
def read_novels(directory_path):
    novels = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            with open(os.path.join(directory_path, filename), 'r', encoding='ANSI') as file:
                text = file.read()
                novels.append(text)
    return novels


# 预处理内容
def content_deal(content):  # 语料预处理，进行分词并去除一些广告和无意义内容
    stopwords = ['‘', ']', '：', '“', '！', '”', "\n", ",", "，", "。", "？", "、", "；", "（", "）", "《", "》", "…", "「", "」", "“", "：", "。", "》", "，",
                 "—", "～", "\\", ", ", "'", "n", "一", "u3000", "【", "】", "……", "-", " ", "　", "w", ".", "c", "r", "1", "7", "3", "o", "m", "t", "x"]
    words = jieba.lcut(content)  # 使用jieba进行分词
    words = [word for word in words if word not in stopwords]  # 去除停用词
    return words


directory_path = 'E:\\PyCharmProject\\BUAA\\NLP\\homework_3\\中文语料库'

# 读取小说内容
novels = read_novels(directory_path)


# 对每部小说进行预处理并分词
tokenized_sentences = []
for novel in novels:
    processed_content = content_deal(novel)
    tokenized_sentences.append(processed_content)


# 训练Word2Vec模型
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=5, workers=16, epochs=30)
model.save("word2vec.model")


# 加载训练好的Word2Vec模型
model = Word2Vec.load("word2vec.model")


# def calculate_similarity(word1, word2, model):
#     similarity_score = model.wv.similarity(word1, word2)
#     print(f"词语 '{word1}' 和 '{word2}' 的相似度得分为：{similarity_score}")
#     return  similarity_score
#
# calculate_similarity('皇上', '皇帝', model)
# calculate_similarity('武林', '江湖', model)
# calculate_similarity('冰霜', '酒杯', model)


# def plot_word_clusters(model, sample_size=1000, n_clusters=10):
#     words = list(model.wv.index_to_key)
#     word_vectors = model.wv[words]
#
#     # 抽样部分数据点进行可视化
#     if len(words) > sample_size:
#         indices = np.random.choice(len(words), sample_size, replace=False)
#         words = [words[i] for i in indices]
#         word_vectors = word_vectors[indices]
#
#     # 使用 t-SNE 降维到2D
#     tsne = TSNE(n_components=2, random_state=42)
#     word_vectors_tsne = tsne.fit_transform(word_vectors)
#
#     # 使用K-means聚类
#     kmeans = KMeans(n_clusters=n_clusters)
#     clusters = kmeans.fit_predict(word_vectors_tsne)
#
#     # 计算轮廓系数
#     silhouette_avg = silhouette_score(word_vectors_tsne, clusters)
#     print(f'Silhouette Score for {n_clusters} clusters: {silhouette_avg:.4f}')
#
#     # 可视化
#     plt.figure(figsize=(12, 12))
#     scatter = plt.scatter(word_vectors_tsne[:, 0], word_vectors_tsne[:, 1], c=clusters, cmap='viridis', s=10, alpha=0.5)
#     plt.colorbar(scatter)
#     plt.title(f'Word2Vec Word Clusters with {n_clusters} Clusters\nSilhouette Score: {silhouette_avg:.4f}')
#     for i, word in enumerate(words):
#         plt.annotate(word, xy=(word_vectors_tsne[i, 0], word_vectors_tsne[i, 1]), fontsize=8, alpha=0.75)
#     #保存图片
#     plt.savefig(f'word2vec_word_clusters_{n_clusters}_clusters.png')
#     plt.show()
#
#
# plot_word_clusters(model)
def plot_and_print_cluster_words(model, sample_size=1000, n_clusters=10):
    words = list(model.wv.index_to_key)
    word_vectors = model.wv[words]

    # 抽样部分数据点进行可视化
    if len(words) > sample_size:
        indices = np.random.choice(len(words), sample_size, replace=False)
        words = [words[i] for i in indices]
        word_vectors = word_vectors[indices]

    # 使用 t-SNE 降维到2D
    tsne = TSNE(n_components=2, random_state=42)
    word_vectors_tsne = tsne.fit_transform(word_vectors)

    # 使用K-means聚类
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(word_vectors_tsne)

    # 可视化
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(word_vectors_tsne[:, 0], word_vectors_tsne[:, 1], c=clusters, cmap='viridis', s=10, alpha=0.5)
    plt.colorbar(scatter)
    plt.title('Word2Vec Word Clusters')
    plt.savefig(f'word2vec_word_clusters_{n_clusters}_clusters.png')
    plt.show()

    # 计算整体轮廓系数
    silhouette_avg = silhouette_score(word_vectors_tsne, clusters)
    print(f'Overall Silhouette Score: {silhouette_avg:.4f}')


    # 计算各簇轮廓系数
    silhouette_values = silhouette_samples(word_vectors_tsne, clusters)

    for target_cluster in range(n_clusters):
        cluster_silhouette_vals = silhouette_values[clusters == target_cluster]
        print(f'Average silhouette score for cluster {target_cluster}: {np.mean(cluster_silhouette_vals):.4f}')

        # 筛选特定簇的词语
        selected_indices = [i for i, cluster_id in enumerate(clusters) if cluster_id == target_cluster]
        selected_words = [words[i] for i in selected_indices]
        selected_word_vectors = word_vectors_tsne[selected_indices]

        # 可视化特定簇
        plt.figure(figsize=(10, 10))
        plt.scatter(selected_word_vectors[:, 0], selected_word_vectors[:, 1], color='red', s=30,
                    label=f'Cluster {target_cluster}')
        for i, word in enumerate(selected_words):
            plt.annotate(word, (selected_word_vectors[i, 0], selected_word_vectors[i, 1]), fontsize=12)
        plt.title(f'Visualization of Words in Cluster {target_cluster}')
        plt.legend()
        plt.grid(True)

        # 保存图像
        plt.savefig(f'cluster_{target_cluster}_visualization.png')
        plt.show()
        print(f"Visualization saved as 'cluster_{target_cluster}_visualization.png'.")
        # 输出指定簇的所有词语
        words_in_cluster = [words[i] for i in range(len(words)) if clusters[i] == target_cluster]
        print(f"Words in cluster {target_cluster}: {words_in_cluster}")


# 可视化词聚类并输出簇中的词语
plot_and_print_cluster_words(model)


# # 词语类比任务
# def word_analogy(model, positive, negative):
#     result = model.wv.most_similar(positive=positive, negative=negative, topn=1)
#     print(f"{positive} - {negative} = {result[0][0]} (相似度: {result[0][1]})")
#
#
# word_analogy(model, positive=['女人', '皇帝'], negative=['男人'])
# word_analogy(model, positive=['武林', '江湖'], negative=['侠客'])
# word_analogy(model, positive=['马蹄', '青石板'], negative=['黑衣'])


