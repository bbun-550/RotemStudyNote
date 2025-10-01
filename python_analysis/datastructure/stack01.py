# 자료 구조 중 stack : LIFO 구조
class MyStack:
    def __init__(self, iterable=None):
        self._data = []    # 내부 전용 변수 ; 캡슐화
        if iterable is not None:
            for x in iterable:
                self.push(x)
                
                
    def push(self, x): # 맨 위(top)에 원소를 추가 ; 속도 O(1)
        self._data.append(x)
        return x
    
    def pop(self):  # 맨 위 원소를 제거 ; 속도 O(1)
        if not self._data:
            raise IndexError('pop from empty stack')
        return self._data.pop()
    
    def is_empty(self):
        return not self._data
    
    def clear(self):
        self._data.clear()
        
    def __repr__(self) -> str:  # (자동 호출되는 메서드) 객체를 문자열로 표현할 때 사용
        # 이쁘게 출력하는 용도
        top_to_bottom = list(reversed(self._data))
        return f"Stack(top -> bottom) {top_to_bottom}"


# LIFO 동작 확인
def demo_lifo():
    s = MyStack()
    for item in ['a', 'b', 'c', 'd']:
        s.push(item)     # d, c, b, a 형태
        print(f"push {item} -> {s}")    # __repr__이 자동 호출된다
        
    print("\nPop until empty (LIFO)")
    while not s.is_empty():
        print(f"pop -> {s.pop()} | now {s}")

demo_lifo()