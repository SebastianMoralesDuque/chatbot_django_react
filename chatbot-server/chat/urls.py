from django.urls import path
from .views import send_message, process_data, add_qa_pairs

app_name = 'chat'

urlpatterns = [
    path('send-message/', send_message, name='send_message'),
    path('process-data/', process_data, name='process_data'),
    path('add-qa-pairs/', add_qa_pairs, name='add_qa_pairs'),
]
