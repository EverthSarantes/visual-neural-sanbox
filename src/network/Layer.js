import { Neuron } from './Neuron.js';
import { getActivation } from '../activation_methods/registry.js';
import { getLayerType } from '../layer_types/registry.js';

export class Layer {
    constructor(id, type, activationId = 'relu', layerType = 'linear', params = {}) {
        this.id = id;
        this.type = type; // 'input' | 'hidden' | 'output'

        this.activationId = getActivation(activationId) ? activationId : 'relu';
        this.connectionTypeId = 'dense';
        this.neurons = [];
        this.neuronSeed = 0;

        // Tipo de capa heterogénea
        this.layerType = layerType; // 'dense' | 'conv2d' | 'maxpool2d' | 'flatten'
        this.params = params || {};

        // Estado para capas con pesos (conv2d, dense usa neuronas)
        this.weights = null;
        this.bias = null;
        this.gradW = null;
        this.gradB = null;
        this.cache = null; // para backward
    }

    addNeurons(count) {
        for (let i = 0; i < count; i++) {
            const neuronId = `n_${this.id}_${this.neuronSeed++}`;
            this.neurons.push(new Neuron(neuronId, this.activationId));
        }
    }

    getLayerTypeInfo() {
        return getLayerType(this.layerType);
    }
}