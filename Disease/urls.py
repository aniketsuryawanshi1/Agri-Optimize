from django.urls import path
from . import views

urlpatterns = [
    path('',views.home , name='home'),
    path('Registration/',views.register, name='register'  ),
    path('Login/', views.log_in, name='log_in'),
    path('Dashboard/', views.dashboard, name='dashboard'),
    path('Logout/', views.log_out, name='log_out'),
    path('Plant verification/', views.accept, name='accept'),
    path('Fertilizer/', views.fertilizer, name='fertilizer'),
    path('Crop Recommendation/', views.crop, name='crop'),

]
