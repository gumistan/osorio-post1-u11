# Post-Contenido 1 — CUDA Benchmark CPU vs GPU

**Curso:** Arquitectura de Computadores — Unidad 11  
**Programa:** Ingeniería de Sistemas  
**Universidad:** Francisco de Paula Santander  
**Año:** 2026  

---

## 1. Descripción del Entorno

Este laboratorio fue ejecutado en Google Colab con GPU T4, dado que el entorno local WSL2 no dispone de GPU NVIDIA física. Google Colab ofrece acceso gratuito a una GPU Tesla T4 con CUDA compute capability 7.5, cumpliendo ampliamente el requisito mínimo de 5.0 establecido en el laboratorio.

| Elemento               | Detalle                           |
|------------------------|-----------------------------------|
| **GPU**                | NVIDIA Tesla T4 (Google Colab)    |
| **CUDA Version**       | 12.x                              |
| **Compute Capability** | 7.5                               |
| **VRAM**               | 15 GB                             |
| **OS**                 | Ubuntu 22.04 (Google Colab)       |
| **Compilador**         | nvcc — NVIDIA CUDA Toolkit 12     |
| **gcc**                | 11.4.0                            |

---

## 2. Comandos de Compilación

Todos los archivos fueron compilados con nvcc usando la flag de optimización -O2, que activa optimizaciones de velocidad sin comprometer la correctitud del resultado:

    nvcc -O2 -o vectorAdd src/vectorAdd.cu
    nvcc -O2 -o vectorAddBench src/vectorAddBench.cu
    nvcc -O2 -o matMul src/matMul.cu

---

## 3. Explicación del Código — vectorAdd.cu

### 3.1 El kernel CUDA

El kernel es la función que se ejecuta en la GPU. Se declara con __global__ para indicar que es llamada desde el host (CPU) pero ejecutada en el device (GPU):

    __global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) C[idx] = A[idx] + B[idx];
    }

- __global__: indica que esta función corre en GPU y es llamada desde CPU.
- blockIdx.x: índice del bloque actual dentro del grid.
- blockDim.x: número de threads por bloque (256 en este caso).
- threadIdx.x: índice del thread dentro de su bloque.
- idx: índice global único de cada thread, calculado combinando bloque y thread.
- if (idx < n): guard necesario porque el grid puede tener más threads que elementos N.
- C[idx] = A[idx] + B[idx]: cada thread suma exactamente un par de elementos.

### 3.2 Gestión de memoria en host y device

CUDA maneja dos espacios de memoria separados: la RAM del host (CPU) y la VRAM del device (GPU). El programador debe gestionar explícitamente las transferencias:

    float *h_A = (float*)malloc(bytes);   // memoria en RAM (host)
    float *d_A;
    cudaMalloc(&d_A, bytes);              // memoria en VRAM (device)
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);  // transferir CPU->GPU

- El prefijo h_ identifica punteros en host (RAM).
- El prefijo d_ identifica punteros en device (VRAM).
- cudaMalloc reserva memoria en VRAM, equivalente a malloc en CPU.
- cudaMemcpy transfiere datos entre host y device en la dirección indicada.
- Al finalizar, cudaFree libera la memoria VRAM, equivalente a free en CPU.

### 3.3 Configuración del lanzamiento del kernel

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

- blockSize = 256: número de threads por bloque. Es un valor estándar que produce buena ocupancia en la mayoría de GPUs NVIDIA.
- gridSize: número de bloques necesarios para cubrir todos los N elementos. La fórmula (N + blockSize - 1) / blockSize es una división entera hacia arriba (ceiling division) que garantiza que no quede ningún elemento sin procesar.
- La sintaxis <<<gridSize, blockSize>>> es exclusiva de CUDA y especifica la configuración del grid al lanzar el kernel.

### 3.4 Medición con cudaEvent

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);

- cudaEvent_t es el tipo de dato para eventos de temporización en GPU.
- cudaEventRecord marca un punto en el stream de ejecución de la GPU.
- cudaEventSynchronize bloquea la CPU hasta que el evento stop haya sido registrado en la GPU.
- cudaEventElapsedTime calcula el tiempo transcurrido en milisegundos entre dos eventos.
- Este método es más preciso que usar clock() de CPU porque mide directamente en el timeline de la GPU.

---

## 4. Explicación del Código — matMul.cu

### 4.1 El kernel naive

    __global__ void matMulNaive(const float *A, const float *B, float *C, int N) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < N && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[row*N + k] * B[k*N + col];
            C[row*N + col] = sum;
        }
    }

- Se usa un grid 2D: cada thread calcula un elemento C[row][col].
- row y col se calculan combinando los índices de bloque y thread en dos dimensiones.
- El bucle interno recorre la dimensión K para acumular el producto punto.
- El problema es que cada iteración del bucle accede a memoria global, que tiene latencia muy alta (400-600 ciclos).

### 4.2 El kernel tiled con shared memory

    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        sA[threadIdx.y][threadIdx.x] = A[row*N + t*TILE + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[(t*TILE + threadIdx.y)*N + col];
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

- __shared__ declara arreglos en shared memory, visible para todos los threads del mismo bloque.
- El algoritmo divide las matrices en tiles de TILE x TILE elementos.
- Cada tile se carga desde memoria global a shared memory una sola vez.
- Los TILE x TILE threads del bloque colaboran para cargar el tile.
- __syncthreads() es una barrera de sincronización: garantiza que todos los threads del bloque hayan terminado de cargar el tile antes de que cualquiera empiece a usarlo.
- Dentro del tile, todos los accesos son a shared memory (latencia ~4 ciclos).
- Al terminar el tile, __syncthreads() asegura que ningún thread pise el tile antes de que todos hayan terminado de usarlo.

---

## 5. Resultados — vectorAdd (Suma de Vectores)

| N elementos | CPU (ms) | GPU kernel (ms) | GPU total con memcpy (ms) |
|-------------|----------|-----------------|---------------------------|
| 1M          | 2.52     | 0.13            | 5.28                      |
| 4M          | 9.99     | 0.20            | 18.48                     |
| 16M         | 39.72    | 0.77            | 75.37                     |

**Verificación de correctitud:** Errores = 0 en todos los casos  
**Configuración del kernel:** blockSize = 256, gridSize = (N + 255) / 256

---

## 6. Resultados — matMul (Multiplicación de Matrices)

| N    | Naive GPU (ms) | Tiled GPU (ms) | Speedup |
|------|----------------|----------------|---------|
| 512  | 30.95          | 0.50           | 61.9x   |
| 1024 | 5.76           | 3.62           | 1.59x   |

**Verificación de correctitud N=512:** Errores = 0 (tolerancia menor a 1e-3 en FP32)  
**Configuración:** block = dim3(16,16), grid = dim3((N+15)/16, (N+15)/16)

---

## 7. Análisis de Resultados

### 7.1 Por qué el kernel GPU es más rápido que la CPU para N grande

La GPU cuenta con miles de núcleos CUDA que ejecutan threads en paralelo bajo el modelo SIMT (Single Instruction, Multiple Threads). Mientras que la CPU procesa los elementos de forma secuencial uno por uno en un solo núcleo, la GPU lanza miles de threads simultáneos, cada uno encargado de calcular exactamente un elemento del vector resultado. Para N=16M elementos, el kernel GPU tardó apenas 0.77 ms frente a 39.72 ms de la CPU, lo que representa un speedup de aproximadamente 51x en cómputo puro. Esta ventaja se amplifica a medida que N crece, porque hay más trabajo para distribuir entre los miles de núcleos disponibles en la GPU. La arquitectura SIMT organiza los threads en grupos de 32 llamados warps, y todos los threads de un warp ejecutan la misma instrucción simultáneamente, maximizando el uso del hardware.

### 7.2 Por qué el tiempo total GPU con memcpy puede ser mayor que la CPU

A pesar de que el kernel GPU es significativamente más rápido en cómputo, la transferencia de datos entre la memoria RAM del host y la memoria VRAM del dispositivo mediante cudaMemcpy tiene un costo fijo considerable. Para N=1M, el tiempo total GPU fue de 5.28 ms frente a 2.52 ms de CPU, siendo la GPU más lenta en tiempo total porque el overhead de transferencia domina sobre el beneficio del paralelismo. Para N=16M ese costo se amortiza entre más operaciones, pero el tiempo total GPU (75.37 ms) todavía supera al de CPU (39.72 ms). Esto demuestra que la GPU es más conveniente cuando los datos pueden mantenerse en VRAM durante múltiples operaciones consecutivas, evitando transferencias repetidas entre host y device. En aplicaciones reales como redes neuronales, los pesos del modelo se cargan una sola vez en VRAM y se reutilizan en miles de iteraciones de entrenamiento.

### 7.3 Por qué el tiling con shared memory mejora el rendimiento

La multiplicación de matrices naive accede a memoria global de la GPU en cada operación individual, generando enormes cantidades de accesos lentos con latencia de 400 a 600 ciclos de reloj. El kernel con tiling carga bloques de 16x16 elementos de las matrices A y B en la shared memory, que es una memoria on-chip con latencia de apenas 4 ciclos, aproximadamente 100 veces más rápida que la memoria global. Con TILE=16, cada dato se carga una sola vez en shared memory y se reutiliza 16 veces para los cálculos del tile, reduciendo los accesos a memoria global por un factor de 16. Esto explica el speedup de 61.9x para N=512. Para N=1024 el speedup es menor (1.59x) porque la caché L2 de la GPU ya absorbe parte de los accesos repetidos del kernel naive, reduciendo la ventaja del tiling explícito.

### 7.4 Reflexión sobre el diseño de kernels CUDA eficientes

Este laboratorio evidencia que escribir un kernel CUDA correcto no es suficiente para obtener buen rendimiento. Es fundamental entender la jerarquía de memoria de la GPU: registros (latencia 1 ciclo), shared memory (4 ciclos), caché L1/L2 (20-100 ciclos) y memoria global (400-600 ciclos). Un kernel bien optimizado minimiza los accesos a memoria global, maximiza la reutilización de datos en shared memory y mantiene alta la ocupancia del SM (Streaming Multiprocessor). La ocupancia mide qué fracción de los recursos del SM están siendo utilizados simultáneamente: a mayor ocupancia, mejor se ocultan las latencias de memoria. Además, la configuración grid/block impacta directamente en el rendimiento: un blockSize de 256 threads es un valor estándar que balancea bien la ocupancia para la mayoría de GPUs NVIDIA modernas.

---

## 8. Estructura del Repositorio

    osorio-post1-u11/
    ├── README.md
    ├── src/
    │   ├── vectorAdd.cu        — Kernel suma de vectores N=16M
    │   ├── vectorAddBench.cu   — Benchmark N=1M, 4M, 16M con memcpy
    │   └── matMul.cu           — Kernel matMul naive + tiled shared memory
    └── capturas/
        ├── checkpoint1.png     — Salida vectorAdd N=16M CPU vs GPU Errores=0
        ├── checkpoint1b.png    — Benchmark vectorAdd N=1M 4M 16M completo
        └── checkpoint2.png     — Salida matMul naive vs tiled N=512 y N=1024

---

## 9. Capturas de Checkpoints

| Captura          | Descripción |
|------------------|-------------|
| checkpoint1.png  | Salida del programa vectorAdd con N=16M: CPU 43.54 ms, GPU kernel 124.24 ms, Errores: 0 |
| checkpoint1b.png | Benchmark completo vectorAdd para N=1M, 4M y 16M mostrando tiempos CPU, GPU kernel y GPU total con memcpy |
| checkpoint2.png  | Salida de matMul mostrando Naive vs Tiled para N=512 (speedup 61.9x) y N=1024 (speedup 1.59x), Errores: 0 |

---

## 10. Conclusiones

Este laboratorio demostró de forma práctica y medible las diferencias fundamentales entre la arquitectura CPU y GPU para cómputo paralelo. La GPU sobresale en tareas con alto grado de paralelismo de datos gracias a sus miles de núcleos CUDA, logrando speedups de hasta 51x en cómputo puro para la suma de vectores con N=16M. Sin embargo, el overhead de transferencia de memoria mediante cudaMemcpy es un factor crítico que debe considerarse en el diseño de soluciones GPU: para problemas pequeños puede hacer que la solución GPU sea más lenta en tiempo total que la CPU secuencial.

La optimización con shared memory en la multiplicación de matrices demostró ser extremadamente efectiva, alcanzando un speedup de 61.9x para N=512 al reducir los accesos a memoria global por un factor de TILE=16. Esto confirma que conocer y explotar la jerarquía de memoria de la GPU es tan importante como el paralelismo mismo. En aplicaciones reales de cómputo de alto rendimiento, visión por computador, aprendizaje profundo e inteligencia artificial, estas técnicas de optimización son fundamentales para aprovechar al máximo el poder de las GPUs modernas. Frameworks como CUDA, cuDNN y TensorRT implementan internamente estas mismas técnicas de tiling y uso de shared memory para lograr el rendimiento que hace posible entrenar modelos de inteligencia artificial a gran escala.
