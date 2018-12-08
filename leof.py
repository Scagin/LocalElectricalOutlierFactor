import numpy as np
import jieba
import pickle
from sklearn.metrics import pairwise_distances


def load_embedding(emb_path):
    word_to_id, embedding = {}, []
    with open(emb_path, encoding='utf-8') as lines:
        for i, line in enumerate(lines):
            line = line.strip()
            array = line.split(' ')
            word = array[0]
            vec = [float(val) for val in array[1:]]
            word_to_id[word] = i
            embedding.append(vec)
    return word_to_id, np.array(embedding)

def load_stopwords(path, encoding='utf-8'):
    stopwords = set()
    with open(path, encoding=encoding) as lines:
        for line in lines:
            stopwords.add(line.strip())
    return stopwords

# word_to_id, embedding = load_embedding('./data/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5')   # 维基百科预训练中文词向量
# pickle.dump(embedding, open('./data/embedding.pkl', 'wb'))
# pickle.dump(word_to_id, open('./data/word_to_id.pkl', 'wb'))

stop_words = load_stopwords('./data/stop_words.txt')
embedding = pickle.load(open('./data/embedding.pkl', 'rb'))
word_to_id = pickle.load(open('./data/word_to_id.pkl', 'rb'))


def cal_cosine_similarity(a, b):
    '''余弦相似度'''
    num = np.sum(a * b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = num / denom
    return cos


def cal_laplace_matrix(matrix):
    '''拉普拉斯矩阵'''
    degree_matrix = np.zeros_like(matrix)
    for i in range(len(matrix)):
        degree_matrix[i, i] = np.sum(matrix[i])
    laplace_matrix = degree_matrix - matrix
    return laplace_matrix


def cal_moore_penrose_inverse(matrix):
    '''计算MP逆矩阵'''
    mp_inverse = np.linalg.pinv(matrix)
    return mp_inverse


def cal_kirchhoff_index_graph(matrix):
    '''计算全图的基尔霍夫指数'''
    return len(matrix) * np.trace(matrix)


def cal_each_kirchhoff_index(matrix):
    '''计算去掉每个点之后的基尔霍夫指数'''
    n_size = len(matrix)
    kirchhoff_list = np.zeros(n_size, dtype=np.float64)
    for i in range(n_size):
        queen_val = matrix[i, i]
        tmp_matrix = np.eye(n_size) * (matrix + queen_val)
        tmp_matrix[i, i] = 0
        kf = (n_size - 1) * np.sum(tmp_matrix)
        kirchhoff_list[i] = kf
    return kirchhoff_list


def cal_electrical_outlier_factor(completed_kf, kirchhoff_list):
    '''计算电离群因子的值'''
    return completed_kf / kirchhoff_list


def cal_local_electrical_outlier_factor(eof, similarity_matrix, k_neighbour):
    '''
    取k个最接近的点，并计算leof值
    '''
    k_neighbour_indices = np.argsort(similarity_matrix, axis=-1)[:,:-k_neighbour-1:-1]  # 排序，返回index下标
    ld = np.sum(eof[:,np.newaxis] / eof[k_neighbour_indices], axis=-1)
    ld = k_neighbour / ld
    return ld


def transform_to_vec(text):
    '''
    文本 -> 向量
    '''
    words = jieba.lcut(text)
    words = [word for word in words if word not in stop_words]
    words_id = [word_to_id.get(word, word_to_id['None']) for word in words]
    emb = embedding[words_id]
    emb = np.mean(emb, axis=0)
    return emb


def cal_similarity_matrix_by_sklearn(data_set):
    '''
    调用sklearn计算相似度矩阵
    '''
    feature_matrix = []
    for text in data_set:
        vec = transform_to_vec(text)
        feature_matrix.append(vec)
    feature_matrix = np.array(feature_matrix, dtype=np.float64)
    # similarity_matrix = 1 / (1 + pairwise_distances(feature_matrix, metric="euclidean"))
    similarity_matrix = 1 - pairwise_distances(feature_matrix, metric="cosine")
    return similarity_matrix


def cal_similarity_matrix_by_rewrite(data_set):
    '''
    重写计算相似度矩阵的代码
    '''
    data_length = len(data_set)
    similarity_matrix = np.zeros((data_length, data_length), dtype=np.float64)
    for i in range(data_length):
        doc_i = data_set[i]
        vec_i = transform_to_vec(doc_i)
        for j in range(i + 1, data_length):
            doc_j = data_set[j]
            vec_j = transform_to_vec(doc_j)
            similarity = cal_cosine_similarity(vec_i, vec_j)
            similarity_matrix[i, j] = similarity
    similarity_matrix += np.copy(similarity_matrix).T
    # similarity_matrix += np.eye(data_length)
    return similarity_matrix


def leof(data_set, similarity_matrix=None, k_neighbour=8):

    if similarity_matrix is None:
        # 相似度矩阵
        print('calculating similarity...')
        similarity_matrix = cal_similarity_matrix_by_sklearn(data_set)
        # similarity_matrix = cal_similarity_matrix_by_rewrite(data_set)

    print(similarity_matrix)

    # 拉普拉斯矩阵
    print('calculating Laplace matrix...')
    laplace_matrix = cal_laplace_matrix(similarity_matrix)

    # MP逆矩阵
    print('calculating Moore-Penrose-Inverse matrix...')
    laplace_matrix_mpinverse = cal_moore_penrose_inverse(laplace_matrix)

    # 基尔霍夫指数
    print('calculating Kirchhoff index...')
    kirchhoff_ = cal_kirchhoff_index_graph(laplace_matrix_mpinverse)
    kirchhoff_list = cal_each_kirchhoff_index(laplace_matrix_mpinverse)

    # 电离群因子
    print('calculating Electrical Outlier Factor...')
    eof = cal_electrical_outlier_factor(kirchhoff_, kirchhoff_list)

    # 局部电离群因子
    print('calculating Local Electrical Outlier Factor...')
    leof = cal_local_electrical_outlier_factor(eof, similarity_matrix, k_neighbour)

    # 取大于1
    dirty_indices = np.where(leof > 1)[0]

    return dirty_indices



if __name__ == '__main__':
    texts = ['这电影不错'] * 4 + ['我很帅']
    dirty_indices = leof(texts)
    
    # a = np.array([[1, 0.5, 0.8], [0.5, 1, 0.93], [0.8, 0.93, 1]])
    # dirty_indices = leof(None, similarity_matrix=a)

    print(dirty_indices)

