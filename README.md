# Neurito

Neurito es una mascota virtual (un ajolote) que decide qué hacer usando una red neuronal pequeña. La red la hicimos desde cero con NumPy y la parte gráfica con Pygame.

La mascota tiene tres necesidades que van de 0 a 1: hambre, energía y aburrimiento. Con eso la red escoge entre comer, dormir o jugar.

## Cómo correrlo

```bash
pip install numpy pygame
python main.py
```

Para ver solo los experimentos, sin abrir la ventana:

```bash
python experimentos.py
```

Teclas:

- H / J: subir o bajar el hambre
- E / D: subir o bajar la energía
- A / S: subir o bajar el aburrimiento
- R: valores aleatorios
- N: estado tranquilo
- F: estado feliz
- ESC: salir

## Cómo funciona la red

Tiene 3 entradas (las necesidades), una capa oculta de 6 neuronas con ReLU y 3 salidas con Softmax, una por cada acción.

Los datos de entrenamiento son 3000 estados aleatorios. La respuesta correcta de cada estado sale de una regla sencilla: se hace lo que más se necesita, comparando el hambre, el cansancio (1 - energía) y el aburrimiento. La red se entrena 1200 épocas con una tasa de aprendizaje de 0.1. Al probarla con 1000 estados nuevos obtuvo una pérdida de 0.1225 y una precisión de 99.4 %.

## Lo que le agregué

Además de las tres acciones de la red, la mascota tiene un estado **feliz** que se calcula con una regla aparte, no con la red. Aparece cuando todo está muy bien (hambre menor a 0.15, energía mayor a 0.85 y aburrimiento menor a 0.15). El ajolote rebota y le salen burbujas.

También le puse movimiento a la interfaz:

- Las barras ya no saltan de un valor a otro, se van llenando poco a poco. Lo mismo pasa con los porcentajes de cada acción.
- Cuando cambia el estado, la imagen anterior se desvanece y la nueva aparece con un pequeño rebote.
- Cada estado tiene su propio movimiento: rebota cuando juega o está feliz, "respira" despacio cuando duerme y se mueve un poco cuando come.

## Experimentos

Todos los resultados se pueden repetir con `experimentos.py` (usa la semilla 42). La salida completa está en `resultados_experimentos.txt`.

### 1. Dejar dos necesidades fijas y cambiar la tercera

Primero dejé la energía en 0.5 (cansancio 0.5) y el aburrimiento en 0.3, y fui subiendo el hambre:

| Hambre | Acción | Comer | Dormir | Jugar |
|---|---|---|---|---|
| 0.3 | Dormir | 2.3 % | 95.6 % | 2.0 % |
| 0.4 | Dormir | 13.3 % | 85.0 % | 1.7 % |
| 0.5 | Comer | 49.9 % | 49.1 % | 0.9 % |
| 0.6 | Comer | 86.6 % | 13.2 % | 0.2 % |
| 0.7 | Comer | 97.7 % | 2.3 % | 0.0 % |

Con el aburrimiento pasó lo mismo: con hambre 0.3 y cansancio 0.5, cambió de dormir a jugar justo en 0.5. Con la energía, dejando hambre y aburrimiento en 0.3, la mascota dejó de dormir cuando la energía llegó a 0.7, o sea cuando el cansancio bajó a 0.3 y quedó empatado con los otros.

Lo que vi es que la acción cambia justo cuando la necesidad que estoy moviendo pasa a la más alta de las otras dos. El cambio no es de golpe: cerca de ese punto la red no está tan segura y las probabilidades quedan casi a la mitad.

### 2. Prioridades casi iguales

| Hambre, energía, aburrimiento | Comer | Dormir | Jugar |
|---|---|---|---|
| 0.50, 0.50, 0.50 | 35.2 % | 33.4 % | 31.3 % |
| 0.51, 0.50, 0.50 | 39.7 % | 31.3 % | 29.1 % |
| 0.60, 0.41, 0.10 | 53.9 % | 46.1 % | 0.0 % |
| 0.90, 0.90, 0.10 | 100 % | 0.0 % | 0.0 % |
| 0.10, 0.90, 0.10 | 34.5 % | 33.7 % | 31.8 % |
| 0.90, 0.10, 0.90 | 35.9 % | 33.2 % | 30.9 % |

Cuando las prioridades son casi iguales, Softmax reparte las probabilidades casi parejo, más o menos 33 % cada una (o 50 y 50 si empatan dos). Con solo 0.01 de diferencia la acción ya cambia, pero con poca seguridad. Algo curioso es que "todo bajo" y "todo alto" dan casi lo mismo, porque la red mira cuál necesidad es mayor y no qué tan grande es. Por eso los estados tranquilo y feliz tienen que ir con reglas aparte.

### 3. Cambiar el tamaño de la capa oculta

| Neuronas | Pérdida | Precisión |
|---|---|---|
| 3 | 0.1618 | 96.3 % |
| 6 | 0.1225 | 99.4 % |
| 10 | 0.1262 | 99.3 % |
| 20 | 0.1220 | 98.8 % |

Con 3 neuronas la red queda corta y pierde unos 3 puntos de precisión. De 6 para arriba casi no cambia nada, porque el problema es sencillo. La pequeña baja con 20 es por la inicialización aleatoria, no porque sea peor. Con 6 está bien.

### 4. Cambiar las épocas y la tasa de aprendizaje

Épocas, con tasa 0.1:

| Épocas | Pérdida | Precisión |
|---|---|---|
| 50 | 0.9388 | 60.6 % |
| 200 | 0.4984 | 91.8 % |
| 1200 | 0.1225 | 99.4 % |
| 5000 | 0.0535 | 99.6 % |

Tasa de aprendizaje, con 1200 épocas:

| Tasa | Pérdida final | Precisión | Qué pasó |
|---|---|---|---|
| 0.001 | 1.0333 | 41.8 % | Se estanca, aprende demasiado lento |
| 0.01 | 0.7376 | 82.9 % | Mejora, pero no le alcanza el tiempo |
| 0.1 | 0.1225 | 99.4 % | Baja de forma estable |
| 1.0 | 0.0447 | 98.7 % | Aprende rápido pero la pérdida sube y baja |
| 5.0 | 1.0979 | 33.8 % | Inestable |
| 20.0 | 6.1621 | 33.8 % | Se dispara por completo |

Con pocas épocas no alcanza a aprender. Con más épocas la pérdida sigue bajando, pero la precisión ya casi no sube después de 1200. Con la tasa pasa algo parecido: si es muy pequeña se estanca, con 0.1 funciona bien y si es muy grande los pasos se pasan del mínimo y el entrenamiento se vuelve inestable. Con 5 o 20 la precisión queda en 33.8 %, que es lo mismo que adivinar entre tres opciones.

### 5. ¿Qué se necesitaría para una cuarta acción?

Por ejemplo, para que se bañe según una necesidad nueva de suciedad:

1. Agregar una cuarta columna a los datos: `X = np.random.rand(3000, 4)`.
2. Cambiar la regla de las etiquetas para que incluya la suciedad: `argmax(hambre, 1 - energía, aburrimiento, suciedad)`, y revisar que las cuatro acciones salgan más o menos la misma cantidad de veces.
3. Cambiar los tamaños de la red: `W1` de (4, ocultas) y `W2` y `b2` con 4 salidas. Seguramente toca subir un poco las neuronas ocultas.
4. En la interfaz, agregar la acción a la lista, sus textos, una barra nueva, teclas para moverla y sus imágenes.

Si la acción nueva no viene de una necesidad nueva (por ejemplo, ir al médico cuando tiene hambre y sueño al mismo tiempo), no basta con el `argmax`. Hay que inventar una regla para etiquetar esos casos y asegurarse de que salgan suficientes ejemplos en los datos.

## Preguntas

**1. ¿En qué se diferencian pesos y sesgos?**

Los pesos multiplican las entradas y dicen cuánto pesa cada una en cada neurona, y si suma o resta. Los sesgos se suman después y no dependen de la entrada, sirven para mover el punto en el que la neurona se activa. Sin sesgo, todas las fronteras de decisión tendrían que pasar por el origen. En Neurito hay 36 pesos (3×6 + 6×3) y 9 sesgos (6 + 3).

**2. ¿Por qué el cansancio es 1 - energía?**

Porque la energía va al revés de las otras dos necesidades. Mucha hambre significa que quiere comer y mucho aburrimiento que quiere jugar, pero mucha energía significa que no necesita dormir. Para poder comparar las tres con el mismo criterio se invierte: con energía 0.9 el cansancio es 0.1 y no duerme; con energía 0.2 el cansancio es 0.8 y sí duerme. Eso se vio en el experimento 1.

**3. ¿Qué hace ReLU y qué hace Softmax?**

ReLU está en la capa oculta: deja pasar los valores positivos y los negativos los vuelve 0. Sirve para que la red no sea solo lineal. Sin ReLU, las dos capas serían como una sola y no podría aprender algo como "escoger el mayor de tres".

Softmax está en la salida: convierte los tres resultados en probabilidades que suman 1. La acción es la de mayor probabilidad, y el número muestra qué tan segura está la red (100 % cuando es obvio, cerca de 33 % cuando hay empate).

**4. ¿Por qué una precisión alta puede ser engañosa?**

El 99.4 % es real, pero tiene trampa:

- Solo mide si la red copia nuestra regla, no si la mascota hace lo correcto de verdad. Si la regla estuviera mal, la red la copiaría igual de bien.
- Esconde dónde se equivoca. Probando con 100 000 estados, la precisión general fue 99.25 %, pero en los casos casi empatados (diferencia menor a 0.02) bajó a 87.5 %, mientras que en los casos claros fue 100 %.
- No dice qué tan segura estaba. Acertar con 50.1 % cuenta igual que acertar con 99.9 %.
- Si las clases no estuvieran balanceadas podría engañar todavía más. Aquí cada acción sale más o menos un 33 %, pero si el 95 % de los datos fuera "comer", una red que siempre diga comer tendría 95 % sin haber aprendido nada.

## Archivos

- `main.py`: la red, el entrenamiento y la interfaz con Pygame.
- `experimentos.py`: los experimentos sin interfaz.
- `resultados_experimentos.txt`: lo que imprime `experimentos.py`.
- `assets/`: las imágenes de la mascota, incluida `feliz_1.png`.
