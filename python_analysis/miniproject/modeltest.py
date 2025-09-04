import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

read_model = pickle.load(open('model.pkl','rb'))

features = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

new_data = pd.DataFrame([
    [200, 50, 100, 30, 10, 20],   # 샘플1
    [500, 20, 400, 15, 25, 40],   # 샘플2
    [50,  10, 20,  5,  5,  10]    # 샘플3
], columns=features)

new_y = [0,0,1] # 실제 결과값

new_pred = read_model.predict(new_data)
print(new_pred)
# 정확도 평가
accuracy = accuracy_score(new_y, new_pred)
print(f"모델 정확도: {accuracy:.4f}\n")