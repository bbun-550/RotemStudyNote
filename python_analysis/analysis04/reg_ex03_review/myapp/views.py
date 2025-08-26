from django.shortcuts import render
from django.shortcuts import render
from django.db import connection
from django.http.response import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Create your views here.
def homeView(request):
    return render(request, 'home.html')

@csrf_exempt
def resultView(request):
    # 입력받은 근속년수 불러오기
    workyear = request.POST.get('workyear','').strip()
    
    # 입력 받은 데이터 숫자형으로 변환
    try:
        workyear = int(workyear)
    except ValueError:
        return JsonResponse({'error': 'workyear 값이 올바르지 않습니다.'})
    workyear = [[workyear]]
    
    sql = """
        SELECT jikwonibsail, jikwonpay, jikwonjik
        FROM jikwon
    """

    with connection.cursor() as cur:
        cur.execute(sql)
        jikwon_table = cur.fetchall()
        # cols = [c[0] for c in cur.description]
        # print(cols)

    data = pd.DataFrame(jikwon_table, columns=['jikwonibsail','jikwonpay'])    
    today = datetime.now().year
    
    # 근속년수 칼럼 생성
    data['workyear'] = today - pd.to_datetime(data['jikwonibsail']).dt.year
    
    # 데이터 분할
    train, test = train_test_split(data, test_size=0.3, random_state=1)
    
    x_train = train[['workyear']].values
    y_train = train['jikwonpay'].values
    x_test = test[['workyear']].values
    y_test = test['jikwonpay'].values
    
    # 모델 학습
    model = LinearRegression().fit(x_train,y_train)

    # print(f'slope : {lmodel.coef_[0]:.4f}') # 기울기 : 
    # print(f'intercept : {lmodel.intercept_:.4f}') # 절편 : 

    # 모델 예측    
    predict_result = model.predict(x_test)
    # print(f'예측값 : {predict_result[:5]:.4f}')
    # print(f'실제값 : {y_test[:5]:.4f}')
    r2 = r2_score(y_test, predict_result)
    
    # 입력받은 근속년수로 급여 예측
    expectedPay = model.predict(workyear)
    
    # 직급별 연봉평균 차트
    jikTable = pd.DataFrame(jikwon_table, columns=['jikwonibsail','jikwonpay','jikwonjik'])[['jikwonpay','jikwonjik']]    
    jik_mean = jikTable.groupby('jikwonjik', as_index=False).mean().sort_values(by='jikwonpay', ascending=False)
    # print(jikTable)
    # print(jik_mean)
    # jik_mean.columns = ['직급, ']

    context_dict = {
        'expectedPay_result':np.round(expectedPay[0],0),
        'r2_result':np.round(r2*100,2),
        # 'jobPays' : jik_mean.to_dict('records')
    }
    
    return JsonResponse(context_dict)