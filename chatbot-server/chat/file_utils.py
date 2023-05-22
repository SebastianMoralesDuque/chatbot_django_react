import json
import os
file_path = os.path.join(os.path.dirname(__file__), 'intents.json')

def append_qa_pairs(qa_pairs_json):
            qa_pairs = json.loads(qa_pairs_json)
            
            # Abrir el archivo existente en modo lectura y escritura
            with open(file_path, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                # Agregar el nuevo par de valores a la lista de intenciones
                data['intents'].append(qa_pairs)
                # Regresar al inicio del archivo para no sobreescribir el contenido
                f.seek(0)
                # Escribir el objeto JSON al archivo con la codificación UTF-8 y ensure_ascii como False
                json.dump(data, f, ensure_ascii=False, indent=2)
                # Truncar el contenido restante del archivo si lo hubiera
                f.truncate()