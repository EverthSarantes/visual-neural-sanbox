import { Connection } from '../network/Connection.js';

/**
 * Enlaza las neuronas de dos capas de forma lineal (neurona i con neurona i)
 */
export function connectOneToOne(fromLayer, toLayer) {
    const minNeurons = Math.min(fromLayer.neurons.length, toLayer.neurons.length);

    for (let i = 0; i < minNeurons; i++) {
        const fromNeuron = fromLayer.neurons[i];
        const toNeuron = toLayer.neurons[i];
        
        const connectionId = `c_${fromNeuron.id}_to_${toNeuron.id}`;
        const connection = new Connection(connectionId, fromNeuron, toNeuron);
        
        fromNeuron.outputs.push(connection);
        toNeuron.inputs.push(connection);
    }
}