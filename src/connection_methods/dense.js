import { Connection } from '../network/Connection.js';

export function connectDense(fromLayer, toLayer) {
    fromLayer.neurons.forEach(fromNeuron => {
        toLayer.neurons.forEach(toNeuron => {
            const connectionId = `c_${fromNeuron.id}_to_${toNeuron.id}`;
            const connection = new Connection(connectionId, fromNeuron, toNeuron);
            
            fromNeuron.outputs.push(connection);
            toNeuron.inputs.push(connection);
        });
    });
}