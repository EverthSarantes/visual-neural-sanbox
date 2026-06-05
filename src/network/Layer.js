import { Neuron } from './Neuron.js';
import { getActivation } from '../activation_methods/registry.js';

export class Layer {
    constructor(id, type, activationId = 'relu') {
        this.id = id;
        this.type = type;
        
        this.activationId = getActivation(activationId) ? activationId : 'relu';
        this.connectionTypeId = 'dense'; 
        this.neurons = [];
        this.neuronSeed = 0;
    }

    addNeurons(count) {
        for (let i = 0; i < count; i++) {
            const neuronId = `n_${this.id}_${this.neuronSeed++}`;
            this.neurons.push(new Neuron(neuronId, this.activationId));
        }
    }
}