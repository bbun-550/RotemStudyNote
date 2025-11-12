from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json

# 한 줄 버전 (리스트 컴프리헨션)
# todos = [{'id': i, 'title': f'할 일 {i}', 'done': False} for i in range(1, 11)]

# 메모리에서 todos 저장 (리로드하면 초기화 됨!)
todos = []

for i in range(1, 11):
    todos.append({
        'id': i,
        'title': f'할 일 {i}',
        'done': False
    })

print(todos)


def home(request):
    return HttpResponse('서버가 정상적으로 작동 중입니다!')

@csrf_exempt
def todo_list(request):
    if request.method == 'GET':
        return JsonResponse(todos, safe=False)
    elif request.method == 'POST':
        # CSRF 예외 처리
        return create_todo(request)


def create_todo(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        todos.append(data)
        return JsonResponse(data, status=201)
