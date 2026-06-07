import { getActivation } from '../activation_methods/registry.js';
import { getLossMethod } from '../loss_methods/registry.js';

/**
 * Ejecuta la retropropagación (Backward Pass) para calcular las señales de error (gradientes)
 * de cada neurona en el grafo, moviéndose de derecha a izquierda.
 * @param {Network} network - La instancia maestra de la red.
 * @param {Array<number>} targetVector - Array con los valores reales esperados del CSV (Target).
 */
export function backwardPass(network, targetVector) {
    if (network.layers.length < 2) return;

    const totalLayers = network.layers.length;

    const outputLayer = network.layers[totalLayers - 1];
    const lossMethod = getLossMethod(network.lossTypeId);
    const outputActivationMethod = getActivation(outputLayer.activationId);

    if (outputActivationMethod.derivativeLayer) {
        outputActivationMethod.derivativeLayer(outputLayer, targetVector, lossMethod);
    } else {
        outputLayer.neurons.forEach((neuron, idx) => {
            if (neuron.isDropped) {
                neuron.errorSignal = 0;
                return;
            }
            const lossTerm = lossMethod.errorSignalTerm(neuron.value, targetVector[idx] || 0);
            neuron.errorSignal = lossTerm * outputActivationMethod.derivative(neuron.netInput);
        });
    }

    for (let i = totalLayers - 2; i > 0; i--) {
        const layer = network.layers[i];
        const activationMethod = getActivation(layer.activationId);

        const errorSums = layer.neurons.map(neuron => {
            if (neuron.isDropped) return 0;

            let errorSum = 0;
            neuron.outputs.forEach(connection => {
                if (!connection.isDropped && !connection.to.isDropped) {
                    errorSum += connection.to.errorSignal * connection.weight;
                }
            });
            return errorSum;
        });

        if (activationMethod.derivativeLayerHidden) {
            activationMethod.derivativeLayerHidden(layer, errorSums);
        } else {
            layer.neurons.forEach((neuron, idx) => {
                if (neuron.isDropped) {
                    neuron.errorSignal = 0;
                    return;
                }
                neuron.errorSignal = errorSums[idx] * activationMethod.derivative(neuron.netInput);
            });
        }
    }
}