import jieba
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# 语料
corpus = np.array([
    "自然语言处理是计算机科学领域的一个重要方向。",
    '据介绍，铁路部门对动车组列车（高铁、动车）服务备品有一整套规范的管理办法，确保备品齐全，干净整洁，方便旅客使用。座席套是动车组列车的重要备品之一，属于布制备品。',
    '为保证旅客乘车体验良好，根据《铁路旅客运输服务质量规范》，铁路部门对座席套实行规范管理：一是与其他布制备品一样定期换洗，一般座席套换洗周期为180'
    '天；二是在换洗期限内污损的，将及时换洗；三是在列车上发生突发污损情况的，工作人员将及时清理，提供备用座套（坐垫）。',
    '极目新闻记者了解到，按照相关规范，动车组列车出库前要进行整备，保证车厢内外各部位整洁，服务备品配备齐全，定位放置，定型统一。'
])

ll = []
for text in corpus:
    words = [word for word in jieba.cut(text)]
    ll.append(' '.join(words))

# 将文本中的词转换成词频矩阵
vectorizer = CountVectorizer()
bag = vectorizer.fit_transform(ll)

print('词汇表:\n', vectorizer.vocabulary_)

print('词向量矩阵:\n', bag.toarray())

tfidf = TfidfTransformer()

tfidf = tfidf.fit_transform(bag)

print('tfidf向量矩阵：\n', tfidf.toarray())

lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')

lda.fit_transform(tfidf)

print(lda.components_.shape)

n_top_words = 5
feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
