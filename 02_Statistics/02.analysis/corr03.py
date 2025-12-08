## 외국인 대상 국내 주요 관광지 방문 관련 상관관계 분석
import json
import matplotlib.pyplot as plt
import matplotlib
plt.rc('font', family='malgun gothic')
import pandas as pd
import numpy as np

# 5. 상관계수를 구하고, 차트를 그린다.
def setScatterGraph(tour_table, all_table, tourPoint): # scatter 차트 작성 함수
    # print(tourPoint)

    # 6. 계산할 관광지명에 해당하는 자료만 뽑아 별도 저장하고, 외국인 관광 자료와 병합
    tour = tour_table[tour_table['resNm']==tourPoint]
    tour = tour.set_index('yyyymm') # 인덱스를 yyyymm
    # print(tour)
    merge_table = pd.merge(tour, all_table, left_index=True, right_index=True)
    # print(merge_table)

    # 7. 시각화 준비
    fig = plt.figure()
    fig.suptitle(tourPoint + ' 상관관계분석')
    
    # 8. 중국
    plt.subplot(1,3,1)
    plt.xlabel('중국인 입국수')
    plt.ylabel('외국인 입장객수')

    # 9. lambda로 상관계수 계산
    lamb1 = lambda p:merge_table['china'].corr(merge_table['ForNum'])
    r1 = lamb1(merge_table)
    # print(f'r1 : {r1:.5f}')

    # 10. 시각화
    plt.title('r={:.5f}'.format(r1))
    plt.scatter(merge_table['china'], merge_table['ForNum'], alpha=0.7, s=6, color='red')

    # 11. 일본
    plt.subplot(1,3,2)
    plt.xlabel('일본인 입국수')
    plt.ylabel('외국인 입장객수')
    lamb2 = lambda p:merge_table['japan'].corr(merge_table['ForNum'])
    r2 = lamb2(merge_table)
    # print(f'r2 : {r2:.5f}')
    plt.title('r={:.5f}'.format(r2))
    plt.scatter(merge_table['japan'], merge_table['ForNum'], alpha=0.7, s=6, color='blue')

    # 12. 미국
    plt.subplot(1,3,3)
    plt.xlabel('미국인 입국수')
    plt.ylabel('외국인 입장객수')
    lamb3 = lambda p:merge_table['usa'].corr(merge_table['ForNum'])
    r3 = lamb3(merge_table)
    # print(f'r3 : {r3:.5f}')
    plt.title('r = {:.5f}'.format(r3))
    plt.scatter(merge_table['usa'], merge_table['ForNum'], alpha=0.7, s=6, color='green')


    # plt.tight_layout()
    # plt.show()
    # plt.close()

    return [tourPoint, r1, r2, r3]

def chulbal():
    # 1. 서울시 관광지 정보 읽어서 DataFrame으로 저장
    # 강사님 방식
    # fname = '서울특별시_관광지입장정보_2011_2016.json'
    # jsonTP = json.loads(open(fname, 'r', encoding='utf-8').read())
    # tour_table = pd.DataFrame(jsonTP, columns=['yyyymm','resNm','ForNum']) # resNm - 관광지명, ForNum - 관광객수
    # tour_table = tour_table.set_index('yyyymm')
    # print(tour_table)

    # 작동하지 않아서 수정한 방식
    fname = 'python_analysis/analysis04/서울특별시_관광지입장정보_2011_2016.json'
    with open(fname, 'r', encoding='utf-8') as f:
        jsonTP = json.load(f)
    tour_table = pd.DataFrame(jsonTP, columns=['yyyymm','resNm','ForNum'])
    # print(tour_table)

    resNm = tour_table.resNm.unique() # 관광지 이름
    # print(f'resNm : {resNm[:5]}') # ['창덕궁' '운현궁' '경복궁' '창경궁' '종묘']
    # 관광지 다섯 곳만 분석한다.

    # 중국인 관광 정보
    cdf = r'python_analysis\analysis04\중국인방문객.json'
    with open(cdf, 'r', encoding='utf-8') as f:
        jdata = json.load(f)
    china_table = pd.DataFrame(jdata, columns=['yyyymm','visit_cnt'])
    china_table = china_table.rename(columns={'visit_cnt':'china'})
    china_table = china_table.set_index('yyyymm')
    # print(china_table[:2])

    # 미국인 관광 정보
    udf = r'python_analysis\analysis04\미국인방문객.json'
    with open(udf, 'r', encoding='utf-8') as f:
        jdata = json.load(f)
    usa_table = pd.DataFrame(jdata, columns=['yyyymm','visit_cnt'])
    usa_table = usa_table.rename(columns={'visit_cnt':'usa'})
    usa_table = usa_table.set_index('yyyymm')
    # print(usa_table[:2])

    # 일본인 관광 정보
    jdf = r'python_analysis\analysis04\일본인방문객.json'
    with open(jdf, 'r', encoding='utf-8') as f:
        jdata = json.load(f)
    japan_table = pd.DataFrame(jdata, columns=['yyyymm','visit_cnt'])
    japan_table = japan_table.rename(columns={'visit_cnt':'japan'})
    japan_table = japan_table.set_index('yyyymm')
    # print(japan_table[:2])

    # 2. 나라별 데이터 merge
    all_table = pd.merge(china_table, japan_table, left_index=True, right_index=True) # 인덱스를 기준으로 merge
    all_table = pd.merge(all_table, usa_table, left_index=True, right_index=True)
    # print(all_table.head(3))

    # 3. 각 관광지별 상관계수와 그래프 그리기
    r_list = [] # 각 관광지별 상관계수 기억
    for tourPoint in resNm[:5]:
        # print(tourPoint)
        r_list.append(setScatterGraph(tour_table, all_table, tourPoint))
    
    # 13. r_list로 DataFrame 작성
    r_df = pd.DataFrame(r_list, columns=['고궁명','중국','일본','미국'])
    r_df = r_df.set_index('고궁명')
    # print(r_df)

    # 14. DataFrame로 차트 출력
    r_df.plot(kind='bar', rot=50) # rot : rotation 인덱스 글자 회전
    plt.show()
    plt.close()

if __name__ == '__main__':
    chulbal()