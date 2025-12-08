# 자료 구조 중 Queue : FIFO 구조
from collections import deque    # 양쪽 끝에서 O(1)으로 삽입·삭제 가능한 덱 컨테이너

class Queue:
    def __init__(self, iterable=None):
        # queue는 양쪽 끝에서 삽입/삭제가 처리속도 O(1)로 빠르게 처리된다.
        self._data = deque()
        
        if iterable is not None:
            for x in iterable:
                self.enqueue(x)
    
    def enqueue(self, x):    # rear에 데이터 추가를 enqueue라고 부른다
        self._data.append(x)    # 뒤(rear, back)에 원소 추가
        return x
    
    def dequeue(self):    # 앞(front)의 원소를 제거
        if not self._data:
            raise IndexError('dequeue from empty Queue')
        return self._data.popleft()
    
    def front(self):    # Queue에서 맨 앞 원소 확인용 메서드
        if not self._data:
            raise IndexError('front from empty Queue')
        return self._data[0]
    
    def is_empty(self):
        return not self._data
    
    def size(self):
        return len(self._data)
    
    def clear(self):
        self._data.clear()
        
    def __repr__(self) -> str:  # (자동 호출되는 메서드) 객체를 문자열로 표현할 때 사용
        # 이쁘게 출력하는 용도
        return f"Queue(front -> rear) {list(self._data)}"
    
# FIFO
def demo_fifo():
    q = Queue()
    for item in ['a', 'b', 'c', 'd']:
        q.enqueue(item)
        print(f"enqueue {item} -> {q}")
    
    print("\nDequeue until empty (FIFO)")
    while not q.is_empty():
        print(f"dequeue -> {q.dequeue()} | now {q}")
        
demo_fifo()
