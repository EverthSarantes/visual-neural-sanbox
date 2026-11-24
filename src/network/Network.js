import { Layer } from './Layer.js';
import { Neuron } from './Neuron.js';
import { getConnectionMethod } from '../connection_methods/registry.js';
import { getLossMethod } from '../loss_methods/registry.js';
import { Connection } from './Connection.js';
import { getInitializationMethod } from '../initialization_methods/registry.js';

export class Network {
    /**
     * @param {string} lossTypeId - ID del método de pérdida global (ej: 'mse')
     * @param {number} learningRate - Velocidad de ajuste de pesos (ej: 0.03)
     */
    constructor(lossTypeId = 'mse', learningRate = 0.03) {
        this.layers = [];

        this.lossTypeId = getLossMethod(lossTypeId) ? lossTypeId : 'mse';
        this.learningRate = learningRate;

        this.initializationStrategyId = 'random'; // 'random', 'xavier' o 'he'
        this.momentum = 0.9;                     // Factor de inercia (0 a 1)
        this.weightDecay = 0.0001;               // Penalización L2 de regularización
        this.dropoutRate = 0.0;                  // 0.0 significa apagado. Ej: 0.2 es 20% de probabilidad
        this.batchSize = 32;

        this.epoch = 0;
        this.currentLoss = 0;
        this.currentAccuracy = 0;
        this.currentMAE = null;
        this.isTraining = false;

        // Metadatos de las columnas objetivo (para desnormalizar métricas)
        this.targetMetadata = [];
        // Activa la ponderación por clase cuando el dataset está desbalanceado
        this.useClassWeighting = false;
    }

    /**
     * Motor quirúrgico de inicialización. Recorre el grafo una vez conectado y calcula los fan-in / fan-out
     */
    applyWeightInitialization() {
        const strategy = getInitializationMethod(this.initializationStrategyId);

        this.layers.forEach(layer => {
            if (layer.layerType === 'dense' || layer.type === 'input' || layer.type === 'output') {
                layer.neurons.forEach(neuron => {
                    const nIn = neuron.inputs.length > 0 ? neuron.inputs.length : 1;
                    const nOut = neuron.outputs.length > 0 ? neuron.outputs.length : 1;

                    neuron.bias = Math.random() * 0.1 - 0.05;
                    neuron.biasVelocity = 0;
                    neuron.isDropped = false;

                    neuron.outputs.forEach(connection => {
                        connection.weight = strategy.generateWeight(nIn, nOut);
                        connection.velocity = 0;
                        connection.isDropped = false;
                    });
                });
            } else if (layer.layerType === 'dense' && layer.weights === null) {
                // Inicializar pesos para capas dense en modo tensor (sin neuronas/conexiones)
                const outF = layer.neurons.length || 1;
                const inF = 1; // se ajustará en forward; usar placeholder
                // Nota: inF real se conoce en forward; aquí inicializamos con tamaño mínimo
                // Para redes MLP, los pesos viven en conexiones, no aquí.
            } else if (layer.layerType === 'conv2d') {
                const p = layer.params || {};
                const filters = p.filters || 8;
                const inC = p.inChannels || 1;
                const k = p.kernelSize || 3;
                const fanIn = inC * k * k;
                const scale = Math.sqrt(2 / fanIn);
                const wSize = filters * inC * k * k;
                layer.weights = new Float32Array(wSize);
                for (let i = 0; i < wSize; i++) layer.weights[i] = (Math.random() * 2 - 1) * scale;
                layer.bias = new Float32Array(filters);
                layer.gradW = new Float32Array(wSize);
                layer.gradB = new Float32Array(filters);
                layer.cache = null;
            }
            // maxpool2d y flatten no tienen pesos
        });
    }

    /**
     * Reestablece el estado de entrenamiento del modelo sin destruir la arquitectura
     */
    resetTrainingState() {
        this.epoch = 0;
        this.currentLoss = 0;
        this.currentAccuracy = 0;
        this.isTraining = false;
        
        // Reiniciar aleatoriamente todos los pesos y sesgos
        this.applyWeightInitialization();
    }

    buildArchitecture(inputCount, hiddenTopology, outputCount) {
        this.layers = [];

        // Capa de Entrada
        const inputLayer = new Layer('input', 'input', 'relu');
        inputLayer.addNeurons(inputCount);
        this.layers.push(inputLayer);

        // Capas Ocultas
        hiddenTopology.forEach((neuronCount, index) => {
            const hiddenLayer = new Layer(`hidden_${index}`, 'hidden', 'relu');
            hiddenLayer.addNeurons(neuronCount);
            this.layers.push(hiddenLayer);
        });

        // Capa de Salida
        const outputLayer = new Layer('output', 'output', 'sigmoid');
        outputLayer.addNeurons(outputCount);
        this.layers.push(outputLayer);

        this.connectLayers();
        this.applyWeightInitialization();
    }

    /**
     * Construye una red a partir de una lista de especificaciones de capa.
     * specs: [{ layerType, params, activationId, type, neuronCount? }]
     */
    buildFromSpecs(specs) {
        this.layers = [];
        specs.forEach((spec, idx) => {
            const layerType = spec.layerType || 'dense';
            const layerId = spec.id || `${layerType}_${idx}`;
            const layerTypeStr = idx === 0 ? 'input' : (idx === specs.length - 1 ? 'output' : 'hidden');
            const layer = new Layer(layerId, layerTypeStr, spec.activationId || 'relu', layerType, spec.params || {});
            if (layerType === 'dense' && spec.neuronCount) {
                layer.addNeurons(spec.neuronCount);
            }
            this.layers.push(layer);
        });
        this.connectLayers();
        this.applyWeightInitialization();
    }

    connectLayers() {
        for (let i = 0; i < this.layers.length - 1; i++) {
            const currentLayer = this.layers[i];
            const nextLayer = this.layers[i + 1];
            const method = getConnectionMethod(currentLayer.connectionTypeId);
            if (method) method.execute(currentLayer, nextLayer);
        }
    }

    /**
     * Limpia los arrays de conexiones en todas las neuronas antes de re-conectar
     */
    clearAllConnections() {
        this.layers.forEach(layer => {
            layer.neurons.forEach(neuron => {
                neuron.inputs = [];
                neuron.outputs = [];
            });
        });
    }

    /**
     * Reconstruye los enlaces de la red adaptándose a la nueva estructura
     */
    rebuildConnections() {
        this.clearAllConnections();
        this.connectLayers();
        this.applyWeightInitialization();
    }

    /**
     * Añade una nueva capa oculta en una posición específica (izquierda o derecha)
     */
    insertHiddenLayerAt(index) {
        const uniqueId = `hidden_${Date.now()}`;
        const newLayer = new Layer(uniqueId, 'hidden', 'relu');
        newLayer.addNeurons(2);
        
        this.layers.splice(index, 0, newLayer);
        this.rebuildConnections();
        return newLayer;
    }

    /**
     * Elimina una capa oculta asegurando no tocar entrada ni salida
     */
    removeLayer(layerId) {
        const index = this.layers.findIndex(l => l.id === layerId);
        if (index === -1) return false;
        
        const layer = this.layers[index];
        if (layer.type === 'input' || layer.type === 'output') return false;

        this.layers.splice(index, 1);
        this.rebuildConnections();
        return true;
    }

    /**
     * Incrementa la cantidad de neuronas en una capa
     */
    addNeuronToLayer(layerId) {
        const layer = this.layers.find(l => l.id === layerId);
        if (!layer) return;

        // Generar neurona con ID único dentro de la capa
        layer.addNeurons(1);
        this.rebuildConnections();
    }

    /**
     * Elimina una neurona específica de una capa manteniendo al menos una conectada
     */
    removeNeuronFromLayer(layerId, neuronId) {
        const layer = this.layers.find(l => l.id === layerId);
        if (!layer) return false;

        if (layer.neurons.length <= 1) return false;

        layer.neurons = layer.neurons.filter(n => n.id !== neuronId);
        this.rebuildConnections();
        return true;
    }

    /**
     * Busca y elimina una neurona específica de la red por su ID.
     * Respeta la regla de mantener al menos 1 neurona por capa.
     * @param {string} neuronId - El ID de la neurona a eliminar
     * @returns {boolean} true si se eliminó, false si viola la regla de mínimos
     */
    removeNeuron(neuronId) {
        // Encontrar qué capa contiene a la neurona inspeccionada
        const layer = this.layers.find(l => l.neurons.some(n => n.id === neuronId));
        if (!layer) return false;

        // REGLA: No se puede dejar una capa completamente vacía (mínimo 1 neurona)
        if (layer.neurons.length <= 1) return false;

        // Filtrar la neurona para sacarla del array
        layer.neurons = layer.neurons.filter(n => n.id !== neuronId);
        
        // Re-calcular todas las conexiones densas de la red
        this.rebuildConnections();
        return true;
    }

    /**
     * Elimina una conexión específica de la red usando su ID
     * @param {string} connectionId 
     */
    removeConnection(connectionId) {
        let found = false;

        this.layers.forEach(layer => {
            layer.neurons.forEach(neuron => {
                const originalLength = neuron.outputs.length;
                neuron.outputs = neuron.outputs.filter(c => c.id !== connectionId);
                if (neuron.outputs.length < originalLength) found = true;

                neuron.inputs = neuron.inputs.filter(c => c.id !== connectionId);
            });
        });

        return found;
    }

    /**
     * Añade una conexión manual entre dos objetos neurona directamente
     * @param {Neuron} fromNeuron 
     * @param {Neuron} toNeuron 
     */
    addManualConnection(fromNeuron, toNeuron) {
        const connectionId = `c_${fromNeuron.id}_to_${toNeuron.id}`;
        if (fromNeuron.outputs.some(c => c.to.id === toNeuron.id)) return;

        const connection = new Connection(connectionId, fromNeuron, toNeuron);
        fromNeuron.outputs.push(connection);
        toNeuron.inputs.push(connection);
    }

    /**
     * Aplica un snapshot (plain object) recibido desde el worker,
     * reconstruyendo la arquitectura y los pesos en esta instancia.
     */
    applySnapshot(snapshot) {
        if (!snapshot) return;
        this.lossTypeId = snapshot.lossTypeId || this.lossTypeId;
        this.learningRate = snapshot.learningRate ?? this.learningRate;
        this.momentum = snapshot.momentum ?? this.momentum;
        this.weightDecay = snapshot.weightDecay ?? this.weightDecay;
        this.dropoutRate = snapshot.dropoutRate ?? this.dropoutRate;
        this.batchSize = snapshot.batchSize ?? this.batchSize;
        this.initializationStrategyId = snapshot.initializationStrategyId || this.initializationStrategyId;
        this.epoch = snapshot.epoch || 0;
        this.currentLoss = snapshot.currentLoss || 0;
        this.currentAccuracy = snapshot.currentAccuracy || 0;
        this.currentMAE = snapshot.currentMAE ?? null;
        this.targetMetadata = snapshot.targetMetadata || [];
        this.classWeights = snapshot.classWeights || [];
        this.useClassWeighting = snapshot.useClassWeighting || false;

        this.layers = [];
        const neuronMap = new Map();

        (snapshot.layers || []).forEach(layerData => {
            const layer = new Layer(layerData.id, layerData.type, layerData.activationId);
            layer.connectionTypeId = layerData.connectionTypeId;
            layer.neuronSeed = layerData.neuronSeed;

            (layerData.neurons || []).forEach(nData => {
                const neuron = new Neuron(nData.id, nData.activationId);
                neuron.bias = nData.bias;
                neuron.value = nData.value;
                neuron.netInput = nData.netInput;
                neuron.errorSignal = nData.errorSignal;
                neuron.biasVelocity = nData.biasVelocity;
                neuron.isDropped = nData.isDropped;
                neuronMap.set(nData.id, neuron);
                layer.neurons.push(neuron);
            });

            this.layers.push(layer);
        });

        (snapshot.layers || []).forEach(layerData => {
            (layerData.neurons || []).forEach(nData => {
                const targetNeuron = neuronMap.get(nData.id);
                (nData.inputs || []).forEach(connData => {
                    const sourceNeuron = neuronMap.get(connData.fromId);
                    if (!sourceNeuron || !targetNeuron) return;
                    let conn = sourceNeuron.outputs.find(c => c.to === targetNeuron);
                    if (!conn) {
                        conn = new Connection(connData.id, sourceNeuron, targetNeuron);
                        sourceNeuron.outputs.push(conn);
                        targetNeuron.inputs.push(conn);
                    }
                    conn.weight = connData.weight;
                    conn.velocity = connData.velocity;
                    conn.isDropped = connData.isDropped;
                });
            });
        });
    }
}