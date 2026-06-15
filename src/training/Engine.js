import { forwardPass } from '../propagation/forward.js';
import { backwardPass } from '../propagation/backward.js';
import { updateParameters, accumulateBatchGradients } from '../propagation/optimizer.js';
import { getLossMethod } from '../loss_methods/registry.js';

export class TrainingEngine {
    /**
     * @param {Network} network - Instancia maestra de la red neuronal.
     * @param {Function} onEpochComplete - Callback para avisar a la UI que una época terminó (y actualizar gráficas).
     * @param {Function} onStepRender - Callback para redibujar el Canvas en cada iteración visual.
     */
    constructor(network, onEpochComplete, onStepRender) {
        this.network = network;
        this.onEpochComplete = onEpochComplete;
        this.onStepRender = onStepRender;
        
        this.trainSet = [];
        this.testSet = [];
        this.animationFrameId = null;
    }

    /**
     * Configura los sets de datos activos para la simulación
     */
    setDatasets(trainSet, testSet) {
        this.trainSet = trainSet;
        this.testSet = testSet;
    }

    /**
     * Enciende el bucle continuo de entrenamiento
     */
    start() {
        if (this.network.isTraining || this.trainSet.length === 0) return;
        this.network.isTraining = true;
        
        const loop = () => {
            if (!this.network.isTraining) return;
            
            this.executeEpoch();
            this.onStepRender(); // Redibuja pesos y brillos en el Canvas
            
            // Re-invocar recursivamente en el siguiente cuadro de animación del navegador
            this.animationFrameId = requestAnimationFrame(loop);
        };
        
        this.animationFrameId = requestAnimationFrame(loop);
    }

    /**
     * Pausa el entrenamiento manteniendo el estado intacto
     */
    pause() {
        this.network.isTraining = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    /**
     * Avanza estrictamente una sola época (Paso a paso)
     */
    step() {
        if (this.network.isTraining || this.trainSet.length === 0) return;
        this.executeEpoch();
        this.onStepRender();
    }

    /**
     * Ejecuta una pasada completa por el set de entrenamiento siguiendo las reglas del Batch Size
     */
    executeEpoch() {
        const size = this.network.batchSize;
        const currentBatchSize = size === -1 ? this.trainSet.length : size;
        
        let batchCounter = 0;

        this.shuffleArray(this.trainSet);

        this.trainSet.forEach((sample) => {
            forwardPass(this.network, sample.inputs);
            backwardPass(this.network, sample.targets);
            
            accumulateBatchGradients(this.network);
            
            batchCounter++;

            if (batchCounter >= currentBatchSize) {
                updateParameters(this.network); 
                batchCounter = 0;
            }
        });

        if (batchCounter > 0) {
            updateParameters(this.network);
        }

        this.evaluateMetrics();
        this.network.epoch++;
        this.onEpochComplete(this.network.epoch, this.network.currentLoss, this.network.currentAccuracy);
    }

    /**
     * Calcula la pérdida y el acierto global usando estrictamente el set de Prueba (Test Set)
     */
    evaluateMetrics() {
        if (this.testSet.length === 0) return;

        let totalLoss = 0;
        let correctPredictions = 0;
        const lossMethod = getLossMethod(this.network.lossTypeId);

        this.testSet.forEach(sample => {
            // apagar por completo el Dropout. Queremos evaluar la red al 100% de su capacidad física.
            const wasTraining = this.network.isTraining;
            this.network.isTraining = false;
            
            const outputs = forwardPass(this.network, sample.inputs);
            
            this.network.isTraining = wasTraining; // Restaurar estado original

            // Sumar pérdidas puntuales de cada neurona de salida
            outputs.forEach((pred, idx) => {
                totalLoss += lossMethod.calculate(pred, sample.targets[idx]);
            });

            // Evaluar Acierto (Accuracy) basado en Clasificación (ArgMax)
            const maxPredIdx = outputs.indexOf(Math.max(...outputs));
            const maxTargetIdx = sample.targets.indexOf(Math.max(...sample.targets));
            
            if (maxPredIdx === maxTargetIdx) {
                correctPredictions++;
            }
        });

        this.network.currentLoss = totalLoss / this.testSet.length;
        this.network.currentAccuracy = (correctPredictions / this.testSet.length) * 100;
    }

    /**
     * Auxiliar algoritmo Fisher-Yates shuffle
     */
    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }
}