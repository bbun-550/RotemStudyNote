# 미분계수 연습
# 평균변화율과 순간변화율을 코드로 작성
# 서울에서 부산까지 자동차로 이동하는 경우, 총거리 400km / 총소요시간 4시간
# - 중심차분공식으로 순간 속력을 근사한다
# - 중심차분 : 양 옆 두 지점의 차이를 이용해 중간 지점의 순간 기울기를 추정한다
import numpy as np
import matplotlib.pyplot as plt
# 데이터 준비
t = np.array([0,1,2,3,4], dtype=float) # 시간
s= np.array([0.0, 80.0, 180.0, 300.0, 400.0], dtype=float) # 누적 이동거리

# 시각화
plt.plot(t, s)
plt.xlabel('t')
plt.ylabel('s')
plt.grid()
plt.show()
plt.close

# 전체 주행 거리
s_tot = s[-1] # 400.0
s_half = s_tot / 2.0 # 전체 거리 절반, 중간 지점 = 200km

# 평균변화율 계산
# 평균속도 = 전체 거리 변화량 / 전체 시간 변화량
t_tot = t[-1] - t[0] # 총소요시간
v_avg = (s[-1] - s[0])/t_tot # 전체 평균 속도

# 보간함수 사용
t_mid = np.interp(s_half, s, t)
print(f't_mid : {t_mid}') # 약 2.17 실제 주행곡선은 속도의 변화 때문에 더 늦게 200km에 도달했다.

# 시간 간격의 중앙값을 구한다. => 평균적인 샘플 계산
dt_mid = np.median(np.diff(t))
h = dt_mid * 0.5 # 중심파분에서 사용할 작은 간격 0.5h
s_plus = np.interp(t_mid + h, t, s) # 중간 지점 약간 앞
s_minus = np.interp(t_mid - h, t, s) # 중간 지점 약간 뒤

# 중심차분으로 순간 속도 추정 (단위:km/h)
v_kph = (s_plus - s_minus) / (2.0 * h) # 기울기 근삿값

print(f'총 이동거리 = {s_tot:.1f}km, 총 소요시간 : {t_tot:.1f}h')
print(f'평균변화율(평균속력) = {v_avg:.1f}km/h')
print(f'중간지점을 지나는 시간 = {t_mid:.2f}시간')
print(f's_plus={s_plus:.1f}km, s_minus={s_minus:.1f}km')
print(f'중간지점의 순간 속력={v_kph:.1f}km/h') # 순간속도 = 순간변화율 = 접선기울기 = 미분계수
'''
총 이동거리 = 400.0km, 총 소요시간 : 4.0h
평균변화율(평균속력) = 100.0km/h
중간지점을 지나는 시간 = 2.17시간
s_plus=260.0km, s_minus=146.7km
중간지점의 순간 속력=113.3km/h
'''