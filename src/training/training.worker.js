import { Network } from '../network/Network.js';
import { TrainingEngine } from './Engine.js';
import { forwardPass } from '../propagation/forward.js';

let network = null;
let engine = null;
let trainSet = [];
let testSet = [];

function serializeNetwork(net) {
    return {
        lossTypeId: net.lossTypeId,
        learningRate: net.learningRate,
        momentum: net.momentum,
        weightDecay: net.weightDecay,
        dropoutRate: net.dropoutRate,
        batchSize: net.batchSize,
        initializationStrategyId: net.initializationStrategyId,
        epoch: net.epoch,
        currentLoss: net.currentLoss,
        currentAccuracy: net.currentAccuracy,
        currentMAE: net.currentMAE,
        targetMetadata: net.targetMetadata,
        classWeights: net.classWeights,
        useClassWeighting: net.useClassWeighting,
        layers: net.layers.map(layer => ({
            id: layer.id,
            type: layer.type,
            activationId: layer.activationId,
            connectionTypeId: layer.connectionTypeId,
            neuronSeed: layer.neuronSeed,
            neurons: layer.neurons.map(neuron => ({
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
}

function rebuildNetworkFromSnapshot(snapshot) {
    // Reconstruye conexiones a partir del snapshot serializado
    const net = new Network(snapshot.lossTypeId || 'mse', snapshot.learningRate || 0.03);
    net.momentum = snapshot.momentum ?? 0.9;
    net.weightDecay = snapshot.weightDecay ?? 0.0001;
    net.dropoutRate = snapshot.dropoutRate ?? 0.0;
    net.batchSize = snapshot.batchSize ?? 32;
    net.initializationStrategyId = snapshot.initializationStrategyId || 'random';
    net.epoch = snapshot.epoch || 0;
    net.currentLoss = snapshot.currentLoss || 0;
    net.currentAccuracy = snapshot.currentAccuracy || 0;
    net.currentMAE = snapshot.currentMAE ?? null;
    net.targetMetadata = snapshot.targetMetadata || [];
    net.classWeights = snapshot.classWeights || [];
    net.useClassWeighting = snapshot.useClassWeighting || false;

    const neuronMap = new Map();
    net.layers = [];
    (snapshot.layers || []).forEach(layerData => {
        const layer = new (require('../network/Layer.js').Layer)(layerData.id, layerData.type, layerData.activationId);
        // Evitar require en ESM: import dinámico no disponible sync; usar global import arriba
        layer.connectionTypeId = layerData.connectionTypeId;
        layer.neuronSeed = layerData.neuronSeed;
        layerData.neurons.forEach(nData => {
            const neuron = new (require('../network/Neuron.js').Neuron)(nData.id, nData.activationId);
            neuron.bias = nData.bias;
            neuron.value = nData.value;
            neuron.netInput = nData.netInput;
            neuron.errorSignal = nData.errorSignal;
            neuron.biasVelocity = nData.biasVelocity;
            neuron.isDropped = nData.isDropped;
            neuronMap.set(nData.id, neuron);
            layer.neurons.push(neuron);
        });
        net.layers.push(layer);
    });

    // Conexiones
    (snapshot.layers || []).forEach(layerData => {
        layerData.neurons.forEach(nData => {
            const targetNeuron = neuronMap.get(nData.id);
            (nData.inputs || []).forEach(connData => {
                const sourceNeuron = neuronMap.get(connData.fromId);
                if (!sourceNeuron || !targetNeuron) return;
                // Reusar Connection si existe, sino crear
                let conn = sourceNeuron.outputs.find(c => c.to === targetNeuron);
                if (!conn) {
                    const { Connection } = require('../network/Connection.js');
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

    return net;
}

// Nota: require no está disponible en ESM. Reescribimos sin require usando imports estáticos.

self.onmessage = async (e) => {
    const { type, payload } = e.data;
    try {
        switch (type) {
            case 'init': {
                // Importar clases dinámicamente (ya importadas arriba de forma estática no posible para Network? sí)
                const { Layer } = await import('../network/Layer.js');
                const { Neuron } = await import('../network/Neuron.js');
                const { Connection } = await import('../network/Connection.js');

                // Construir arquitectura
                network = new Network(payload.lossTypeId || 'mse', payload.learningRate || 0.03);
                network.momentum = payload.momentum ?? 0.9;
                network.weightDecay = payload.weightDecay ?? 0.0001;
                network.dropoutRate = payload.dropoutRate ?? 0.0;
                network.batchSize = payload.batchSize ?? 32;
                network.initializationStrategyId = payload.initializationStrategyId || 'random';
                network.epoch = payload.epoch || 0;
                network.currentLoss = payload.currentLoss || 0;
                network.currentAccuracy = payload.currentAccuracy || 0;
                network.currentMAE = payload.currentMAE ?? null;
                network.targetMetadata = payload.targetMetadata || [];
                network.classWeights = payload.classWeights || [];
                network.useClassWeighting = payload.useClassWeighting || false;

                // buildArchitecture manual para controlar activaciones
                const inputLayer = new Layer('input', 'input', 'none');
                inputLayer.addNeurons(payload.inputCount);
                network.layers.push(inputLayer);

                (payload.hiddenTopology || []).forEach((count, idx) => {
                    const hl = new Layer(`hidden_${idx}`, 'hidden', 'relu');
                    hl.addNeurons(count);
                    network.layers.push(hl);
                });

                const outputLayer = new Layer('output', 'output', payload.outputActivation || 'sigmoid');
                outputLayer.addNeurons(payload.outputCount);
                network.layers.push(outputLayer);

                network.connectLayers();
                network.applyWeightInitialization();

                // datasets
                trainSet = payload.trainSet || [];
                testSet = payload.testSet || [];
                engine = new TrainingEngine(network, (epoch, loss, accuracy, mae) => {
                    self.postMessage({ type: 'epoch', payload: { epoch, loss, accuracy, mae, network: serializeNetwork(network) } });
                }, () => {});

                engine.setDatasets(trainSet, testSet);
                self.postMessage({ type: 'ready', payload: { network: serializeNetwork(network) } });
                break;
            }
            case 'start': {
                if (!engine) return;
                engine.start();
                break;
            }
            case 'pause': {
                if (!engine) return;
                engine.pause();
                self.postMessage({ type: 'paused' });
                break;
            }
            case 'step': {
                if (!engine) return;
                engine.step();
                break;
            }
            case 'reset': {
                if (!network || !engine) return;
                network.resetTrainingState();
                engine.setDatasets(trainSet, testSet);
                self.postMessage({ type: 'resetDone', payload: { network: serializeNetwork(network) } });
                break;
            }
            case 'setHyperparams': {
                if (!network) return;
                const p = payload || {};
                if (p.learningRate !== undefined) network.learningRate = p.learningRate;
                if (p.momentum !== undefined) network.momentum = p.momentum;
                if (p.weightDecay !== undefined) network.weightDecay = p.weightDecay;
                if (p.dropoutRate !== undefined) network.dropoutRate = p.dropoutRate;
                if (p.batchSize !== undefined) network.batchSize = p.batchSize;
                if (p.lossTypeId !== undefined) network.lossTypeId = p.lossTypeId;
                if (p.initializationStrategyId !== undefined) {
                    network.initializationStrategyId = p.initializationStrategyId;
                    network.applyWeightInitialization();
                }
                break;
            }
            case 'setDatasets': {
                trainSet = payload.trainSet || [];
                testSet = payload.testSet || [];
                if (engine) engine.setDatasets(trainSet, testSet);
                break;
            }
            case 'reinit': {
                if (!network) return;
                network.initializationStrategyId = payload.strategy || network.initializationStrategyId;
                network.applyWeightInitialization();
                self.postMessage({ type: 'resetDone', payload: { network: serializeNetwork(network) } });
                break;
            }
            case 'setNetwork': {
                if (!network || !payload?.network) return;
                const snap = payload.network;
                const neuronMap = new Map();
                network.layers.forEach(l => l.neurons.forEach(n => neuronMap.set(n.id, n)));
                (snap.layers || []).forEach(l => {
                    (l.neurons || []).forEach(n => {
                        const target = neuronMap.get(n.id);
                        if (!target) return;
                        if (n.activationId !== undefined) target.activationId = n.activationId;
                        if (n.bias !== undefined) target.bias = n.bias;
                        if (n.biasVelocity !== undefined) target.biasVelocity = n.biasVelocity;
                        (n.inputs || []).forEach(ci => {
                            const conn = target.inputs.find(c => c.id === ci.id);
                            if (conn) {
                                if (ci.weight !== undefined) conn.weight = ci.weight;
                                if (ci.velocity !== undefined) conn.velocity = ci.velocity;
                                if (ci.isDropped !== undefined) conn.isDropped = ci.isDropped;
                            }
                        });
                    });
                });
                if (snap.learningRate !== undefined) network.learningRate = snap.learningRate;
                if (snap.momentum !== undefined) network.momentum = snap.momentum;
                if (snap.weightDecay !== undefined) network.weightDecay = snap.weightDecay;
                if (snap.dropoutRate !== undefined) network.dropoutRate = snap.dropoutRate;
                if (snap.batchSize !== undefined) network.batchSize = snap.batchSize;
                if (snap.lossTypeId !== undefined) network.lossTypeId = snap.lossTypeId;
                break;
            }
            case 'terminate': {
                self.close();
                break;
            }
            default:
                break;
        }
    } catch (err) {
        self.postMessage({ type: 'error', payload: { message: err?.message || String(err) } });
    }
};
