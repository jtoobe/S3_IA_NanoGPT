Version 1: Funcional, el modelo es chico, hay que mejorar el dataset

Para generar el modelo (model.h & vocals.h) usar lo que esta en la carpeta modelo

Explicacion del funcionamiento:

NanoGPT en ESP32-S3
A continuación se expone una explicación estructurada del funcionamiento del sketch, para que se comprenda tanto la arquitectura del modelo como las decisiones de implementación en un microcontrolador.


Contexto y objetivo del proyecto
El programa ejecuta un modelo de lenguaje pequeño (NanoGPT) directamente sobre un microcontrolador ESP32-S3, sin depender de la nube.

El modelo fue entrenado previamente en Python con un archivo dataset.json y sus pesos se exportaron como arreglos de enteros de 8 bits (INT8) para que quepan en la memoria del microcontrolador.
El formato de conversación que entiende el modelo es:
<usuario> pregunta <asistente> respuesta <fin>
Cada una de esas etiquetas tiene un identificador numérico fijo (del 0 al 4), y el resto del vocabulario empieza en el índice 5.

Configuración del modelo
Al inicio del archivo se definen las hiperparámetros arquitectónicos que deben coincidir exactamente con los del entrenamiento en Python:
EMB_SIZE = 48: cada token se representa con un vector de 48 dimensiones.
BLOCK_SIZE = 24: la secuencia máxima que puede procesar.
NUM_HEADS = 4: cabezas de atención paralelas.
HEAD_SIZE = 12: dimensión de cada cabeza (48 / 4).
NUM_BLOCKS = 2: bloques Transformer apilados.
FF_HIDDEN = 96: dimensión oculta de la red feed-forward.
MAX_NEW_TOKENS = 15: límite de tokens generados por respuesta.
Estos valores están verificados en tiempo de arranque mediante aserciones estáticas, de modo que si alguien los cambia sin ajustar el resto, el compilador avisa antes de grabar el firmware.


Preprocesamiento del texto de entrada
Antes de que el modelo reciba la frase del usuario, el texto pasa por dos etapas:

Normalización léxica:
Se replica la misma limpieza que hace el script de entrenamiento en Python:
Se convierte todo a minúsculas.
Se reemplazan vocales con tilde y la "ñ" por sus equivalentes sin acento.
Se conservan únicamente letras, dígitos y espacios; todo lo demás se descarta.
Esto garantiza que lo que ve el modelo en inferencia sea idéntico a lo que vio en entrenamiento.

Tokenización
La frase normalizada se divide en palabras por espacios y cada palabra se busca en el vocabulario cargado desde vocab.h. Si una palabra no existe, se asigna el token especial <unk>. La secuencia resultante se construye así:
Token <usuario> al inicio.
Tokens de cada palabra de la pregunta.
Token <asistente> como marcador de que ahora el modelo debe generar.
Espacio reservado para los tokens que se irán añadiendo durante la generación.

El corazón del modelo: los bloques Transformer
El corazón del programa es la función forward, que replica paso a paso lo que haría NanoGPT.forward() en PyTorch. Se ejecuta sobre todas las posiciones de la secuencia a la vez (no solo la última), porque la atención necesita ver el contexto completo.

Embeddings de token y posición
Cada token se convierte en un vector sumando dos contribuciones:
El embedding de token, que aporta el significado de la palabra.
El embedding de posición, que indica en qué lugar de la secuencia está.
Ambos están almacenados como INT8 con su factor de escala, y se convierten a flotante al combinarse. El resultado se guarda en la matriz X, que representa el estado residual que irá fluyendo por los bloques.

Estructura de cada bloque Transformer
Cada uno de los dos bloques contiene:
LayerNorm 1 sobre el estado residual.
Atención multi-cabeza autoatentiva con máscara causal.
Proyección de salida y suma residual.
LayerNorm 2.
Red feed-forward (dos capas lineales con activación GELU).
Suma residual final.
Los pesos de cada bloque se cargan al inicio mediante macros que enlazan los arreglos globales de model.h con una estructura BlockWeights, facilitando el acceso durante la inferencia.

Operaciones matemáticas clave
LayerNorm: calcula media y varianza del vector, lo normaliza y aplica los pesos y sesgos aprendidos. Es fundamental para estabilizar el entrenamiento y la inferencia.
Multiplicación matriz-vector (matmul): operación central que proyecta el estado a los espacios Q, K, V, a la proyección de atención y a la red feed-forward. Trabaja con pesos INT8 y escala el resultado a flotante.
GELU exacta: función de activación no lineal usada en la feed-forward, calculada con la función de error (erff) para coincidir con PyTorch.
Atención causal con softmax numéricamente estable: para cada posición t, se calcula la similitud con todas las posiciones anteriores (nunca futuras), se aplica un truco de resta del máximo para evitar desbordamientos en la exponencial, se normaliza con softmax y se ponderan los vectores de valor. El resultado se concatena por cabezas y se proyecta.
Todas estas operaciones se realizan sobre buffers globales (no en la pila) para evitar desbordamientos de memoria, ya que el ESP32-S3 tiene RAM limitada.

Generación autoregresiva
Una vez que forward ha procesado toda la secuencia, se toma el último vector del estado residual, se le aplica la LayerNorm final (ln_f) y se multiplica por la matriz del cabezal de lenguaje (lm_head) para obtener una puntuación por cada palabra del vocabulario.
Se elige la palabra con mayor puntuación (argmax), se añade a la secuencia y se repite el proceso:
Se vuelve a llamar a forward con la secuencia extendida.
Se predice el siguiente token.
Se continúa hasta que aparece el token <fin>, se alcanza el máximo de tokens nuevos o se llena el tamaño de bloque.
Este esquema se llama generación autoregresiva: cada token generado depende de todos los anteriores.

Verificación de coherencia al arrancar
Antes de aceptar entradas del usuario, el programa comprueba que las dimensiones declaradas en los macros coincidan con los tamaños reales de los arreglos de pesos. Si hay discrepancia (por ejemplo, alguien cambió EMB_SIZE sin regenerar model.h), el programa se detiene y muestra un mensaje de error. Esto evita comportamientos extraños difíciles de depurar en el microcontrolador.

Consideraciones de eficiencia y hardware
Algunas decisiones de diseño merecen destacarse:
Pesos en INT8: reducen a la cuarta parte el consumo de memoria flash frente a flotante de 32 bits, con pérdida de precisión aceptable para un modelo pequeño.
Buffers globales: los tensores intermedios (X, Qb, Kb, Vb, ffh, etc.) son estáticos para no saturar la pila.
Llamadas a yield(): se insertan en bucles largos para que el watchdog y las tareas del sistema operativo del ESP32 no reinicien el dispositivo.
Replicación exacta de PyTorch: se usa GELU exacta (no la aproximación tanh), softmax con resta del máximo y el mismo orden de operaciones, para que los resultados en el microcontrolador sean numéricamente equivalentes a los del entrenamiento.


Flujo resumido
Para cerrar, el ciclo completo que ocurre cada vez que el usuario escribe una frase es:
Recibir la línea por el puerto serie.
Normalizar (minúsculas, sin tildes, solo alfanuméricos).
Tokenizar buscando cada palabra en el vocabulario.
Preparar la secuencia con las etiquetas <usuario> y <asistente>.
Iterar hasta MAX_NEW_TOKENS: ejecutar forward sobre la secuencia actual, predecir el siguiente token con argmax, añadirlo y mostrarlo.
Detenerse al generar <fin> o alcanzar el límite.

 Este ejemplo muestra cómo la misma arquitectura que usan los grandes modelos de lenguaje (Transformer) puede ejecutarse, en versión miniatura, sobre un microcontrolador ESP32-S3 Dev.Kit . Las limitaciones de memoria y potencia obligan a tomar decisiones interesantes: cuantización INT8, buffers estáticos, replicación numérica exacta… todas ellas técnicas que también se usan (a mayor escala) en la industria.

 La programacion fue asistida con Qwen3.7-Plus y tiene fines educativos.
