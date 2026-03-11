from django.urls import path 
from .views import(
                    HomeView,chat_api
                    )
urlpatterns = [
                path('', HomeView.as_view(), name='home'),
                path("chat/", chat_api, name="chat_api"),
]