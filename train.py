from pyvi import ViTokenizer
from tqdm import tqdm
import gensim
import joblib
import os

dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
data_dir = os.path.join(dir_path, 'Data')

def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = gensim.utils.simple_preprocess(lines)
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)
                X.append(lines)
                y.append(path)
    return X, y

train_path = r'C:\Users\mdnt5\Downloads\Filter\train'
X_data, y_data = get_data(train_path)

joblib.dump(X_data, open('X_data.pkl', 'wb'))
joblib.dump(y_data, open('y_data.pkl', 'wb'))