import math, random, sys, time
import numpy as np

try:
    import pygame
except ImportError:
    print("pygame 미설치: pip install pygame")
    raise

# 전역 하이퍼파라미터
class G:
    SEED = 42
    WIDTH, HEIGHT = 960, 640
    FPS = 60
    HEADLESS = False          # True면 렌더링 없이 빠르게 진행
    SHOW_SENSORS = True

    # 트랙(도넛) 크기
    OUTER_MARGIN = 40
    INNER_MARGIN = 180
    CORNER_RADIUS = 200       # 둥근 직사각형처럼 보이도록 4코너 굴곡

    # 에피소드/진화 파라미터
    POP_SIZE = 12
    ELITE = 2
    N_GENERATIONS = 10
    EPISODE_STEPS = 1200      # 한 개체를 평가하는 최대 스텝 수 (FPS 가정)
    SENSOR_COUNT = 5
    SENSOR_FOV_DEG = 120      # 좌우로 펼친 총 각도
    SENSOR_MAX_DIST = 220.0

    # 차량 파라미터
    CAR_LENGTH = 26
    CAR_WIDTH = 14
    MAX_STEER = math.radians(25)
    MAX_ACCEL = 0.12
    FRICTION = 0.015
    TURN_GAIN = 0.045
    SPEED_CLAMP = 5.0

    # 신경망 구조: inputs(센서5 + speed + bias=1) -> hidden(8) -> outputs(2: steer, accel)
    N_IN = SENSOR_COUNT + 1 + 1
    N_H = 8
    N_OUT = 2

    # GA
    MUT_P = 0.15
    MUT_SIGMA = 0.25
    CROSSOVER_P = 0.8

random.seed(G.SEED)
np.random.seed(G.SEED)


# 기하 유틸
def line_intersection(p, r, q, s):
    """선분 p→p+r 와 q→q+s 의 교차점(t,u)을 구한다. (없으면 None)
       참고: 선분 교차 파라메트릭 폼
    """
    rxs = r[0]*s[1] - r[1]*s[0]
    if abs(rxs) < 1e-9:  # 평행
        return None
    qmp = (q[0]-p[0], q[1]-p[1])
    t = (qmp[0]*s[1]-qmp[1]*s[0]) / rxs
    u = (qmp[0]*r[1]-qmp[1]*r[0]) / rxs
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (p[0]+t*r[0], p[1]+t*r[1]), t
    return None

def point_in_polygon(pt, poly):
    """홀수 교차법: pt가 다각형 poly 내부에 있으면 True (단순 폴리곤 가정)"""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1)%n]
        cond1 = (y1 > y) != (y2 > y)
        if cond1:
            xints = (x2 - x1)*(y - y1)/(y2 - y1 + 1e-9) + x1
            if x < xints:
                inside = not inside
    return inside

def rounded_rect_polygon(w, h, r, cx, cy, n_corner=10):
    """중앙(cx,cy)에 폭 w, 높이 h, 코너반경 r 인 둥근 직사각형을 다각형으로 생성"""
    r = min(r, w/2 - 2, h/2 - 2)
    # 4코너 중심
    corners = [
        (cx - w/2 + r, cy - h/2 + r), # TL
        (cx + w/2 - r, cy - h/2 + r), # TR
        (cx + w/2 - r, cy + h/2 - r), # BR
        (cx - w/2 + r, cy + h/2 - r), # BL
    ]
    pts = []
    # 각 코너마다 90도 원호를 n_corner 분할
    for i,(cxk, cyk) in enumerate(corners):
        start = i * math.pi/2 + math.pi
        for j in range(n_corner+1):
            th = start + j*(math.pi/2)/n_corner
            pts.append((cxk + r*math.cos(th), cyk + r*math.sin(th)))
    return pts

def polygon_edges(poly):
    return list(zip(poly, poly[1:]+poly[:1]))


# 트랙 구성 (도넛 모양)
class Track:
    def __init__(self):
        W, H = G.WIDTH, G.HEIGHT
        outer_w = W - 2*G.OUTER_MARGIN
        outer_h = H - 2*G.OUTER_MARGIN
        inner_w = W - 2*G.INNER_MARGIN
        inner_h = H - 2*G.INNER_MARGIN

        self.outer = rounded_rect_polygon(outer_w, outer_h, G.CORNER_RADIUS, W/2, H/2, 18)
        self.inner = rounded_rect_polygon(inner_w, inner_h, G.CORNER_RADIUS*0.6, W/2, H/2, 18)
        self.outer_edges = polygon_edges(self.outer)
        self.inner_edges = polygon_edges(self.inner)

        # 시작 위치/방향
        self.start = (G.WIDTH/2 - inner_w/2 - 30, G.HEIGHT/2)  # 왼쪽 스트레이트
        self.start_angle = 0.0  # 오른쪽(+x)으로 바라보게

    def on_track(self, p):
        return point_in_polygon(p, self.outer) and (not point_in_polygon(p, self.inner))

    def raycast(self, p, ang, maxdist):
        """양쪽 다각형 경계에 대해 레이캐스트, 가장 가까운 교차거리 반환"""
        dx = math.cos(ang)*maxdist
        dy = math.sin(ang)*maxdist
        best = None
        for (a,b) in self.outer_edges + self.inner_edges:
            hit = line_intersection(p, (dx,dy), a, (b[0]-a[0], b[1]-a[1]))
            if hit is not None:
                pos, t = hit
                dist = t*maxdist
                if best is None or dist < best:
                    best = dist
        return best if best is not None else maxdist



# 차량 & 센서
class Car:
    def __init__(self, track):
        self.track = track
        self.reset()

    def reset(self):
        self.x, self.y = self.track.start
        self.angle = self.track.start_angle
        self.v = 0.0
        self.alive = True
        self.time_alive = 0
        self.forward_reward = 0.0

    def step(self, steer_cmd, accel_cmd):
        # 조향/가속 제한
        steer = max(-G.MAX_STEER, min(G.MAX_STEER, steer_cmd * G.MAX_STEER))
        accel = max(-G.MAX_ACCEL, min(G.MAX_ACCEL, accel_cmd * G.MAX_ACCEL))

        # 간단한 자전거 모델 흉내 (거칠게)
        self.v += accel
        self.v = max(-0.5, min(G.SPEED_CLAMP, self.v))
        self.angle += steer * (1.0 + 0.15*abs(self.v)) * G.TURN_GAIN

        # 마찰
        if abs(self.v) > 1e-6:
            self.v -= math.copysign(G.FRICTION, self.v)
        else:
            self.v = 0.0

        self.x += math.cos(self.angle)*self.v
        self.y += math.sin(self.angle)*self.v

        # 트랙 이탈 시 사망
        if not self.track.on_track((self.x, self.y)):
            self.alive = False

        # 전진 보상 (차량 방향과 속도 부호 일치 시 가점)
        self.forward_reward += max(0.0, self.v*math.cos(0.0))  # 전역 x축 기준 단순화
        self.time_alive += 1

    def sensor_readings(self):
        """레이 5개: -FOV/2 ~ +FOV/2 범위"""
        readings = []
        span = math.radians(G.SENSOR_FOV_DEG)
        for i in range(G.SENSOR_COUNT):
            a = -span/2 + span*(i/(G.SENSOR_COUNT-1))
            ang = self.angle + a
            d = self.track.raycast((self.x, self.y), ang, G.SENSOR_MAX_DIST)
            readings.append(d / G.SENSOR_MAX_DIST)  # 0~1
        return np.array(readings, dtype=np.float32)

    def rect_points(self):
        """차량 사각형 꼭짓점 (렌더용)"""
        L = G.CAR_LENGTH
        W = G.CAR_WIDTH
        pts = [ ( L/2,  0),
                (-L/2, -W/2),
                (-L/2,  W/2)]
        # 앞 삼각형처럼 그릴 것 (간단히)
        rot = lambda X,Y: (self.x + X*math.cos(self.angle) - Y*math.sin(self.angle),
                           self.y + X*math.sin(self.angle) + Y*math.cos(self.angle))
        return [rot(*p) for p in pts]



# 신경망 & 유전알고리즘
class Brain:
    """단층 MLP: in->hidden(8,tanh)->out(2,tanh)"""
    def __init__(self, w1=None, b1=None, w2=None, b2=None):
        if w1 is None:
            self.w1 = np.random.randn(G.N_IN, G.N_H)*0.7
            self.b1 = np.zeros((G.N_H,), dtype=np.float32)
            self.w2 = np.random.randn(G.N_H, G.N_OUT)*0.7
            self.b2 = np.zeros((G.N_OUT,), dtype=np.float32)
        else:
            self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2

    def clone(self):
        return Brain(self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy())

    def forward(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        y = np.tanh(h @ self.w2 + self.b2)  # [-1,1]
        # y[0]=steer, y[1]=accel
        return y

    def genome(self):
        return np.concatenate([self.w1.ravel(), self.b1, self.w2.ravel(), self.b2])

    @staticmethod
    def from_genome(g):
        s1 = G.N_IN*G.N_H
        s1b = s1 + G.N_H
        s2 = s1b + G.N_H*G.N_OUT
        # cut
        w1 = g[:s1].reshape(G.N_IN, G.N_H)
        b1 = g[s1:s1b]
        w2 = g[s1b:s2].reshape(G.N_H, G.N_OUT)
        b2 = g[s2:]
        return Brain(w1, b1, w2, b2)


def crossover(g1, g2):
    if random.random() > G.CROSSOVER_P:
        return g1.copy(), g2.copy()
    cut = random.randint(1, len(g1)-2)
    c1 = np.concatenate([g1[:cut], g2[cut:]])
    c2 = np.concatenate([g2[:cut], g1[cut:]])
    return c1, c2

def mutate(gn):
    mask = np.random.rand(len(gn)) < G.MUT_P
    noise = np.random.randn(len(gn))*G.MUT_SIGMA
    gn[mask] += noise[mask]
    return gn



# 에피소드 평가 / 학습 루프
def evaluate(brain, track):
    car = Car(track)
    total = 0.0
    for step in range(G.EPISODE_STEPS):
        if not car.alive:
            break
        sensors = car.sensor_readings()  # 5
        speed = np.array([car.v / G.SPEED_CLAMP], dtype=np.float32)
        x = np.concatenate([sensors, speed, np.array([1.0], dtype=np.float32)])  # + bias=1
        steer, accel = brain.forward(x)
        car.step(steer, accel)
    # 피트니스: 생존시간 + 전진보상
    total = car.time_alive + 3.5*car.forward_reward
    return total, car


def run_evolution(screen):
    clock = pygame.time.Clock()
    track = Track()

    # 초기 개체군
    brains = [Brain() for _ in range(G.POP_SIZE)]
    genomes = [b.genome() for b in brains]

    best_fit = -1e9
    best_brain = None

    for gen in range(1, G.N_GENERATIONS+1):
        fits = []
        starts = time.time()
        for i, gn in enumerate(genomes):
            br = Brain.from_genome(gn)
            fit, _ = evaluate(br, track)
            fits.append(fit)

        order = np.argsort(fits)[::-1]
        genomes = [genomes[i] for i in order]
        fits = [fits[i] for i in order]

        if fits[0] > best_fit:
            best_fit = fits[0]
            best_brain = Brain.from_genome(genomes[0].copy())

        # 로그
        print(f"[Gen {gen:02d}] best={fits[0]:.1f}  mean={np.mean(fits):.1f}  time={time.time()-starts:.1f}s")

        # 리플레이(상위 1~2개 시각화)
        if not G.HEADLESS:
            replay(screen, track, [Brain.from_genome(genomes[0])], title=f"Gen {gen} Top-1")

        # 다음 세대 생성 (엘리트 유지 + 교차/돌연변이)
        next_gen = genomes[:G.ELITE]  # 엘리트 보존
        while len(next_gen) < G.POP_SIZE:
            p1, p2 = random.sample(genomes[:max(4, G.POP_SIZE//2)], 2)  # 상위 절반에서 부모 선택
            c1, c2 = crossover(p1, p2)
            next_gen.append(mutate(c1))
            if len(next_gen) < G.POP_SIZE:
                next_gen.append(mutate(c2))
        genomes = next_gen

    print(f"\n=== Finished ===\nBest fitness: {best_fit:.1f}")
    if not G.HEADLESS:
        replay(screen, track, [best_brain], title="BEST")



# 렌더링
def draw_track(screen, track):
    screen.fill((18,18,22))
    pygame.draw.polygon(screen, (70,70,70), track.outer)  # 아스팔트
    pygame.draw.polygon(screen, (18,18,22), track.inner)  # 내부 구멍
    # 가장자리 라인
    pygame.draw.lines(screen, (220,220,220), True, track.outer, 2)
    pygame.draw.lines(screen, (220,220,220), True, track.inner, 2)

def draw_car(screen, car):
    pts = car.rect_points()
    pygame.draw.polygon(screen, (0,170,255), pts)
    # 센서
    if G.SHOW_SENSORS and car.alive:
        span = math.radians(G.SENSOR_FOV_DEG)
        for i in range(G.SENSOR_COUNT):
            a = -span/2 + span*(i/(G.SENSOR_COUNT-1))
            ang = car.angle + a
            d = car.track.raycast((car.x, car.y), ang, G.SENSOR_MAX_DIST)
            x2 = car.x + math.cos(ang)*d
            y2 = car.y + math.sin(ang)*d
            pygame.draw.line(screen, (255,200,50), (car.x,car.y), (x2,y2), 1)
            pygame.draw.circle(screen, (255,200,50), (int(x2),int(y2)), 2)

def replay(screen, track, brains, title="REPLAY"):
    clock = pygame.time.Clock()
    cars = [Car(track) for _ in brains]
    steps = 0
    while steps < G.EPISODE_STEPS and any(c.alive for c in cars):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)

        # 동시 시뮬
        for c, b in zip(cars, brains):
            if not c.alive: 
                continue
            sensors = c.sensor_readings()
            speed = np.array([c.v / G.SPEED_CLAMP], dtype=np.float32)
            x = np.concatenate([sensors, speed, np.array([1.0], dtype=np.float32)])
            steer, accel = b.forward(x)
            c.step(steer, accel)

        draw_track(screen, track)
        for c in cars:
            draw_car(screen, c)
        # 텍스트
        font = pygame.font.SysFont("consolas", 18)
        txt = font.render(f"{title}  step {steps}", True, (240,240,240))
        screen.blit(txt, (16, 12))

        pygame.display.flip()
        clock.tick(G.FPS)
        steps += 1


# 메인
def main():
    if G.HEADLESS:
        # 화면 없이 빠르게 GA 실행
        pygame.display.init()  # 일부 Surface 연산을 위해 최소 init
        screen = pygame.Surface((G.WIDTH, G.HEIGHT))
        run_evolution(screen)
        return

    pygame.init()
    screen = pygame.display.set_mode((G.WIDTH, G.HEIGHT))
    pygame.display.set_caption("Mini Evo Car (GA + Tiny MLP)")
    run_evolution(screen)
    # 종료 대기
    done = False
    while not done:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                done = True
        time.sleep(0.02)
    pygame.quit()

if __name__ == "__main__":
    main()


