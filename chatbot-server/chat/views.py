from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import time
import json
from django.http import JsonResponse
from .file_utils import append_qa_pairs
from .chat_logic import logic
from .train_red import train



@csrf_exempt
def process_data(request):
    if request.method == 'POST':
        train()
        return JsonResponse({'message': 'Datos procesados correctamente'})

    return JsonResponse({'message': 'Método no permitido'})


@csrf_exempt
def send_message(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body.decode('utf-8'))
            message = body.get('message', '')
            response = logic(message)

            return JsonResponse({'message': response})
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Datos JSON inválidos'}, status=400)
    else:
        return JsonResponse({'message': 'Método no permitido'})


@csrf_exempt
def add_qa_pairs(request):
    if request.method == 'POST':
        try:
            qa_pairs_json = request.body.decode('utf-8')
            append_qa_pairs(qa_pairs_json)
            return JsonResponse({'message': 'Pares de preguntas y respuestas agregados correctamente'})
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Datos JSON inválidos'}, status=400)
    else:
        return JsonResponse({'message': 'Método no permitido'})
