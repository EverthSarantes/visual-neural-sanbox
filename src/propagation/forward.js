import { getActivation } from '../activation_methods/registry.js';

/**
 * Ejecuta la propagación hacia adelante (Forward Pass) en todo el grafo de la red.
 * @param {Network} network - La instancia maestra de la red.
 * @param {Array<number>} inputVector - Array con los valores de entrada de una fila del CSV.
 * @returns {Array<number>} Los valores de activación finales de la capa de salida.
 */
export function forwardPass(network, inputVector) {
    if (network.layers.length === 0) return [];

    // INYECTAR ENTRADAS
    const inputLayer = network.layers[0];
    inputLayer.neurons.forEach((neuron, idx) => {
        neuron.value = inputVector[idx] || 0;
        neuron.netInput = neuron.value;
        neuron.isDropped = false;
    });

    // APLICAR MÁSCARA DE DROPOUT
    if (network.isTraining && network.dropoutRate > 0) {
        for (let i = 1; i < network.layers.length - 1; i++) {
            const layer = network.layers[i];
            layer.neurons.forEach(neuron => {
                neuron.isDropped = Math.random() < network.dropoutRate;
            });
        }
    } else {
        network.layers.forEach(layer => {
            layer.neurons.forEach(n => n.isDropped = false);
        });
    }

    // PROPAGACIÓN CAPA POR CAPA
    for (let i = 1; i < network.layers.length; i++) {
        const layer = network.layers[i];
        const activationMethod = getActivation(layer.activationId);

        layer.neurons.forEach(neuron => {
            if (neuron.isDropped) {
                neuron.netInput = 0;
                neuron.value = 0;
                return;
            }

            let sum = 0;
            neuron.inputs.forEach(connection => {
                if (!connection.from.isDropped && !connection.isDropped) {
                    sum += connection.from.value * connection.weight;
                }
            });

            neuron.netInput = sum + neuron.bias;
        });

        if (activationMethod.calculateLayer) {
            activationMethod.calculateLayer(layer, network);
        } else {
            layer.neurons.forEach(neuron => {
                if (neuron.isDropped) return;

                let activationValue = activationMethod.calculate(neuron.netInput);

                if (network.isTraining && network.dropoutRate > 0 && layer.type === 'hidden') {
                    activationValue /= (1 - network.dropoutRate);
                }

                neuron.value = activationValue;
            });
        }
    }

    // RECOLECTAR RESULTADOS DE SALIDA
    const outputLayer = network.layers[network.layers.length - 1];
    return outputLayer.neurons.map(n => n.value);
}