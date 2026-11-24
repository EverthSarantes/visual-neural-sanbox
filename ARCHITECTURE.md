# Arquitectura del Visual Neural Sandbox

## Flujo de datos

1. **Carga CSV** (`utils/CsvParser.js` + `ui/CsvController.js`): se diagnostican columnas (tipo, rango, cardinalidad) y se mapean roles (entrada/salida/ignorar) y normalización.
2. **Compilación** (`CsvController.compileSelectedArchitecture`): se calcula `D_in` y `D_out`, se construye la red (`Network.buildArchitecture`) y se divide el dataset en train/test.
3. **Renderizado** (`ui/CanvasRenderer.js`): dibuja el grafo físico (capas, neuronas, conexiones) a partir del estado de `Network`.
4. **Entrenamiento** (`training/Engine.js`): por época mezcla el train set, ejecuta forward/backward/optimizer por lotes y evalúa métricas sobre el test set.
5. **Inferencia** (`ui/PredictionModal.js`): construye un vector de entrada y ejecuta `forwardPass`.

## Representación física de la red

- `Network` → `Layer[]` → `Neuron[]` → `Connection[]` (`inputs`/`outputs`).
- Cada `Neuron` encapsula su comportamiento físico:
  - `computeNetInput()`: `bias + Σ(w · value)`.
  - `applyActivation(x)` / `activate()`: aplica la función de activación configurada.
  - `updateBias(gradiente, network)`: regla de actualización (momentum + clipping).
- Cada `Connection` encapsula:
  - `updateWeight(gradiente, network)`: momentum + L2 + clipping.

La propagación (`propagation/*`) solo **orquesta** el grafo; no recalcula la matemática que ya vive en los objetos.

## Separación matemática

Toda la matemática pura vive en `src/math/`:

- `math/activations.js`: `identity`, `relu`, `sigmoid`, `tanh`, y las operaciones de capa de `softmax`.
- `math/losses.js`: `mse`, `bce`, `cce` (cálculo y término de gradiente).
- `math/initializers.js`: `random`, `xavier`, `he`.

Los registros (`activation_methods/`, `loss_methods/`, `initialization_methods/`) son adaptadores delgados que añaden metadatos (nombre, descripción) para la UI. Esto permite testear las funciones matemáticas de forma aislada.

## Convención de gradientes

Se usa la convención estándar: `errorSignal` almacena la **verdadera derivada parcial** `∂L/∂z`. El descenso se realiza restando (`peso -= velocidad`). No hay factores `multiplier` mágicos: cada pérdida devuelve su gradiente con el signo correcto.

## Módulos principales

| Módulo | Responsabilidad |
|---|---|
| `network/` | Grafo físico y estado |
| `math/` | Funciones matemáticas puras |
| `propagation/` | Orquestación forward/backward/optimizer |
| `training/` | Bucle de épocas y métricas |
| `ui/` | Renderizado, controles, inspector |
| `utils/` | Parseo CSV y persistencia |