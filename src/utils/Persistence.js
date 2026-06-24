import { Layer } from '../network/Layer.js';
import { Neuron } from '../network/Neuron.js';
import { Connection } from '../network/Connection.js';

/**
 * Exporta el estado absoluto de la red, configuraciones y datasets a un archivo JSON.
 */
export function exportModel(network, uiController, version = "1.0.0") {
    const modelState = {
        version: version,
        timestamp: new Date().toISOString(),
        
        config: {
            learningRate: network.learningRate,
            momentum: network.momentum,
            dropoutRate: network.dropoutRate,
            weightDecay: network.weightDecay,
            batchSize: network.batchSize,
            lossTypeId: network.lossTypeId,
            initializationStrategyId: network.initializationStrategyId,
            epoch: network.epoch,
            currentLoss: network.currentLoss,
            currentAccuracy: network.currentAccuracy,
            currentMAE: network.currentMAE,
            datasetDiagnostics: uiController.datasetDiagnostics,
        },

        targetMetadata: network.targetMetadata,

        datasets: {
            trainSet: uiController.trainSet || [],
            testSet: uiController.testSet || []
        },

        topology: network.layers.map((layer, lIdx) => ({
            id: layer.id,
            layerIndex: lIdx,
            type: layer.type,
            activationId: layer.activationId,
            connectionTypeId: layer.connectionTypeId,
            neuronSeed: layer.neuronSeed,
            neurons: layer.neurons.map((neuron, nIdx) => ({
                neuronIndex: nIdx,
                id: neuron.id,
                activationId: neuron.activationId,
                bias: neuron.bias,
                value: neuron.value,
                netInput: neuron.netInput,
                errorSignal: neuron.errorSignal,
                biasVelocity: neuron.biasVelocity,
                isDropped: neuron.isDropped,

                inputs: neuron.inputs.map(conn => ({
                    id: conn.id,
                    weight: conn.weight,
                    velocity: conn.velocity,
                    isDropped: conn.isDropped,
                    fromNeuronId: conn.from.id
                }))
            }))
        }))
    };

    // Usamos un Blob en lugar de un DataURI crudo para soportar datasets grandes de forma robusta
    const blob = new Blob([JSON.stringify(modelState, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", url);
    downloadAnchor.setAttribute("download", `modelo_epoca_${network.epoch}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
    URL.revokeObjectURL(url); // Liberar memoria
}

/**
 * Importa y rehidrata por completo la arquitectura, configuraciones y memoria de la red.
 * @param {Object} jsonData - El objeto parseado del archivo JSON cargado
 * @param {Network} network - Instancia viva de la red de la app
 * @param {Object} uiController - Controlador de la interfaz de usuario
 * @param {Object} renderer - Instancia del CanvasRenderer para redibujar el lienzo
 */
export function importModel(jsonData, network, uiController, renderer) {
    try {
        const cfg = jsonData.config;

        // REHIDRATAR HIPERPARÁMETROS GLOBALES DE LA RED
        network.learningRate = cfg.learningRate;
        network.momentum = cfg.momentum;
        network.dropoutRate = cfg.dropoutRate;
        network.weightDecay = cfg.weightDecay;
        network.batchSize = cfg.batchSize;
        network.lossTypeId = cfg.lossTypeId;
        network.initializationStrategyId = cfg.initializationStrategyId;
        network.epoch = cfg.epoch;
        network.currentLoss = cfg.currentLoss;
        network.currentAccuracy = cfg.currentAccuracy;
        network.currentMAE = cfg.currentMAE || null;
        network.targetMetadata = jsonData.targetMetadata || null;

        // RESTAURAR DATASETS Y METADATA DE DIAGNÓSTICO
        uiController.datasetDiagnostics = cfg.datasetDiagnostics;
        uiController.trainSet = jsonData.datasets.trainSet;
        uiController.testSet = jsonData.datasets.testSet;
        
        // Sincronizar el motor de entrenamiento si está instanciado
        if (uiController.engine) {
            uiController.engine.setDatasets(uiController.trainSet, uiController.testSet);
        }

        // RECONSTRUCCIÓN FÍSICA DEL GRAFO
        network.layers = [];
        const neuronInstancesMap = new Map();

        // Reinstanciar Capas y Neuronas con sus propiedades numéricas históricas
        jsonData.topology.forEach(layerData => {
            // Invocar directamente el constructor puro de tu Layer.js
            const liveLayer = new Layer(layerData.id, layerData.type, layerData.activationId);
            liveLayer.connectionTypeId = layerData.connectionTypeId;
            liveLayer.neuronSeed = layerData.neuronSeed;

            layerData.neurons.forEach(nData => {
                const liveNeuron = new Neuron(nData.id, nData.activationId);
                
                liveNeuron.bias = nData.bias;
                liveNeuron.value = nData.value;
                liveNeuron.netInput = nData.netInput;
                liveNeuron.errorSignal = nData.errorSignal;
                liveNeuron.biasVelocity = nData.biasVelocity;
                liveNeuron.isDropped = nData.isDropped;
                
                neuronInstancesMap.set(nData.id, liveNeuron);

                liveLayer.neurons.push(liveNeuron);
            });

            network.layers.push(liveLayer);
        });

        // Rehidratación simétrica de las Conexiones
        jsonData.topology.forEach(layerData => {
            layerData.neurons.forEach(nData => {
                const targetNeuron = neuronInstancesMap.get(nData.id);

                nData.inputs.forEach(connData => {
                    const sourceNeuron = neuronInstancesMap.get(connData.fromNeuronId);
                    
                    if (!sourceNeuron) {
                        console.warn(`No se encontró la neurona origen ${connData.fromNeuronId} para la conexión ${connData.id}`);
                        return;
                    }

                    const liveConnection = new Connection(connData.id, sourceNeuron, targetNeuron);

                    liveConnection.weight = connData.weight;
                    liveConnection.velocity = connData.velocity;
                    liveConnection.isDropped = connData.isDropped;

                    targetNeuron.inputs.push(liveConnection);
                    sourceNeuron.outputs.push(liveConnection);
                });
            });
        });

        if (typeof uiController.syncMetrics === 'function') uiController.syncMetrics();
        if (uiController && typeof uiController.syncSlidersAndSelects === 'function') {
            uiController.syncSlidersAndSelects();
        }

        if (renderer && typeof renderer.render === 'function') {
            renderer.render(network);
        }

        return true;

    } catch (error) {
        alert("Ocurrió un error al procesar el archivo. Asegúrate de que es un JSON válido de este simulador.");
        return false;
    }
}