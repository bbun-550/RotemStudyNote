from django.db import models

# Create your models here.
class Survey(models.Model): # 클래스명은 바꿔도 되지만 아래 Meta 테이블명 바꾸면 안된다. 클래스는 장고에서 쓰는 것이다
    rnum = models.AutoField(primary_key=True)
    gender = models.CharField(max_length=4, blank=True, null=True)
    age = models.IntegerField(blank=True, null=True)
    co_survey = models.CharField(max_length=10, blank=True, null=True)

    class Meta:
        managed = False # 테이블 생성 X 기존 테이블 사용
        db_table = 'survey' # mariaDB의 테이블명