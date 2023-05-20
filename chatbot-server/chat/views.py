from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def send_message(request):
    if request.method == 'POST':
        message = request.POST.get('message', '')
        # Procesa el mensaje y genera la respuesta
        response = 'Hoasdxasdasdasdla, soy un chatbot.'

        return JsonResponse({'message': response})
