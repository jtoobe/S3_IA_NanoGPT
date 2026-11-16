"""
Generador de Dataset JSON Expandido para NanoGPT
Extrae pares únicos y crea variaciones para evitar overfitting.
"""
import re
import json

def limpiar_texto(texto):
    texto = texto.lower().strip()
    texto = re.sub(r'[¿?!.,;:"]', '', texto)
    for a, b in {'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u', 'ñ':'n'}.items():
        texto = texto.replace(a, b)
    return texto

# 1. Extraer pares únicos del archivo original
pares_unicos = {}
with open("mi_texto.txt", "r", encoding="utf-8") as f:
    for linea in f:
        linea = limpiar_texto(linea)
        if "<usuario>" in linea and "<asistente>" in linea and "<fin>" in linea:
            partes = linea.replace("<fin>", "").split("<asistente>")
            if len(partes) == 2:
                pregunta = partes[0].replace("<usuario>", "").strip()
                respuesta = partes[1].strip()
                # Usamos la respuesta como clave para agrupar variaciones de la misma pregunta
                if respuesta not in pares_unicos:
                    pares_unicos[respuesta] = []
                if pregunta not in pares_unicos[respuesta]:
                    pares_unicos[respuesta].append(pregunta)

print(f"Se encontraron {len(pares_unicos)} conceptos únicos.")

# 2. Plantillas de variación para enriquecer el dataset
plantillas_pregunta = [
    "que es {tema}",
    "explica {tema}",
    "define {tema}",
    "para que sirve {tema}",
    "como funciona {tema}",
    "cual es la funcion de {tema}",
    "dame informacion sobre {tema}"
]

# 3. Generar el dataset expandido
dataset_expandido = []
conceptos_procesados = 0

for respuesta, preguntas_originales in pares_unicos.items():
    # Tomamos la primera palabra o dos del tema original para las plantillas
    # Ej: "el esp32", "un gpio", "mqtt"
    tema_base = preguntas_originales[0].split()[-2:] if len(preguntas_originales[0].split()) > 1 else preguntas_originales[0].split()
    tema_str = " ".join(tema_base)
    
    # Agregamos las preguntas originales
    for p in preguntas_originales:
        dataset_expandido.append({"pregunta": p, "respuesta": respuesta})
    
    # Agregamos variaciones generadas por plantilla
    for plantilla in plantillas_pregunta:
        nueva_pregunta = plantilla.format(tema=tema_str)
        # Evitar duplicados exactos
        if nueva_pregunta not in preguntas_originales:
            dataset_expandido.append({"pregunta": nueva_pregunta, "respuesta": respuesta})
            
    conceptos_procesados += 1

# 4. Guardar como JSON
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset_expandido, f, indent=2, ensure_ascii=False)

print(f"✓ Dataset generado: dataset.json")
print(f"  Total de ejemplos únicos: {len(dataset_expandido)}")
print(f"  Conceptos base cubiertos: {conceptos_procesados}")
