from django.shortcuts import render, redirect, get_object_or_404
from .models import Post

# Create your views here.
def post_list(request):
    posts = Post.objects.all().order_by('-created_at')
    return render(request, 'board/post_list.html', {'posts': posts})


def post_detail(request, post_id):
    post = Post.objects.get(id=post_id)
    return render(request, 'board/post_detail.html', {'post': post})


def post_create(request):
    if request.method == 'POST':
        Post.objects.create(
            author=request.POST['author'],
            title=request.POST['title'],
            content=request.POST['content'],
            url=request.POST['url'],
            email=request.POST['email']
        )
        return redirect('post_list')
    return render(request, 'board/post_form.html')

def post_delete(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    if request.method == 'POST':
        post.delete()
        return redirect('post_list')
    return render(request, 'board/post_conform_delete.html', {'post': post})


def post_update(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    if request.method == 'POST':
        post.title = request.POST['title']
        post.author = request.POST['author']
        post.content = request.POST['content']
        post.url = request.POST['url']
        post.email = request.POST['email']
        post.save()
        return redirect('post_detail', post_id=post.id)
    return render(request, 'board/post_form.html', {'form': form})