from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from pyvi import ViTokenizer
import gensim
import joblib

def preprocessing_doc(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)
    return lines

def predict(test_doc):
    X_data = joblib.load(open('X_data.pkl', 'rb'))
    y_data = joblib.load(open('y_data.pkl', 'rb'))
    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=300000)
    X_data_tfidf = tfidf_vect.fit_transform(X_data)
    model = naive_bayes.MultinomialNB()
    model.fit(X_data_tfidf, y_data)
    test_doc = preprocessing_doc(test_doc)
    test_doc_tfidf = tfidf_vect.transform([test_doc])
    if model.predict(test_doc_tfidf) == 'Suc khoe':
        return True
    return False


"""
import time
start_time = time.time()
test_doc = '''Chỉ số BMI là gì?.'''
print(predict(test_doc))
end_time = time.time()
execution_time = end_time - start_time
print("Thời gian thực thi: ", execution_time, " giây")


đoạn này viết thêm để chạy tính thời gian khi bỏ vào project thì xoá đi 

cách chạy anh cứ gôm hết các import anh bỏ vào gg colab cái nào báo lỗi 
chưa pip install anh copy về vs code anh install là chạy được. 
xong rồi anh chạy file train.py để lưu X_data.pkl và y_data.pkl thật ra anh không 
cần chạy file train.py cũng được chỉ cần có X_data.pkl và y_data.pkl là chạy được file test.py rồi
cái train.py chỉ chạy để lưu 2 fiel kia thôi chứ không liên quan gì tới file test.py.
hàm dự đoán nếu là sức khoẻ thì true mà không thì false.
"""