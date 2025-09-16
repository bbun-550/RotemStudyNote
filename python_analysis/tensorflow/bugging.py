# 영화 리뷰 이전 분류
# imdb dataset
from keras.datasets import imdb
(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words=10000)
# !ls -al /root/.keras/datasets
# print(train_data)
print(train_data[0], len(train_data[0]))

# 참고 : 리뷰 데이터 하나를 원래 영어 단어로 보기
word_index = imdb.get_word_index()
# print(word_index)
# print(word_index.items())

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# print(reverse_word_index)

for i, (k, v) in enumerate(sorted(reverse_word_index.items(), key=lambda x:x[0])):
    if i <= 10:
        print(k, " ", v)

decord_review = ' '.join([reverse_word_index.get(i) for i in train_data[0]])
print(decord_review)