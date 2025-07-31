from django.shortcuts import render, redirect
from .models import Post

# Create your views here.
# Create 글쓰기
def post_create(request):
    if request.method == 'POST':
        Post.objects.create(
            name=request.POST['name'],
            age=request.POST['age'],
            phone=request.POST['phone'],            
            email=request.POST['email']
        )
        return redirect('post_list')
    return render(request, 'board/post_form.html')

# Lists 목록페이지
def post_list(request):
    posts = Post.objects.all().order_by('-created_at')
    return render(request, 'board/post_list.html', {'posts': posts})


# Details 글보기
def post_detail(request, post_id):
    post = Post.objects.get(id=post_id)
    return render(request, 'board/post_detail.html', {'post': post})


