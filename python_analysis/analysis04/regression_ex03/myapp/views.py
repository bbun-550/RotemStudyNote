from django.shortcuts import render
from django.db import connection
from django.http.response import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create your views here.
def Index(request):
    return render(request, 'index.html')

@csrf_exempt
def Result(request):
    workyear = request.POST.get('workyear','').strip()
    try:
        workyear = int(workyear)
    except ValueError:
        return JsonResponse({'error': 'workyear 값이 올바르지 않습니다.'}, status=400)
    workyear = [[workyear]]

    sql = """
        SELECT jikwonibsail, jikwonpay
        FROM jikwon
    """

    with connection.cursor() as cur:
        cur.execute(sql)
        jikwon_table = cur.fetchall()
        cols = [c[0] for c in cur.description]
        # print(cols)

    data = pd.DataFrame(jikwon_table, columns=cols)
    today = datetime.now().year
    
    data['workyear'] = today - pd.to_datetime(data['jikwonibsail']).dt.year
    # print(data.head(2))

    x = data[['workyear']].values
    # print(x)
    y = data['jikwonpay'].values

    lmodel = LinearRegression().fit(x,y)
    # print(f'slope : {lmodel.coef_[0]:.4f}') # 기울기 : 583.2902
    # print(f'intercept : {lmodel.intercept_:.4f}') # 절편 : -1227.8497

    predict = lmodel.predict(workyear)
    # print(f'예측값 : {predict[0]:.4f}')
    r2 = r2_score(y, lmodel.predict(x))

    context_dict = {
        'expectpay_result':np.round(predict[0],0),
        'r2_result':np.round(r2*100,2),
    }
    return JsonResponse(context_dict)



# def Result(request):
#     return render(request, 'result.html')