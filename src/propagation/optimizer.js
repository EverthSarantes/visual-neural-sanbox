import { getLossMethod } from '../loss_methods/registry.js';

/**
 * Captura los gradientes instantáneos de la fila actual del CSV y los suma
 * en variables temporales dentro de las neuronas y conexiones antes de que se borren.
 */
export function accumulateBatchGradients(network) {
    const lossMethod = getLossMethod(network.lossTypeId);
    const multiplier = (lossMethod && lossMethod.multiplier !== undefined) ? lossMethod.multiplier : 1;

    for (let i = 1; i < network.layers.length; i++) {
        const layer = network.layers[i];

        layer.neurons.forEach(neuron => {
            if (neuron.isDropped) return;

            const adjustedError = neuron.errorSignal * multiplier;
            if (neuron.batchBiasGrad === undefined) neuron.batchBiasGrad = 0;
            neuron.batchBiasGrad += adjustedError;

            neuron.inputs.forEach(connection => {
                if (connection.isDropped || connection.from.isDropped) return;

                const weightGradient = adjustedError * connection.from.value;

                if (connection.batchWeightGrad === undefined) connection.batchWeightGrad = 0;
                connection.batchWeightGrad += weightGradient;
            });
        });
    }
}

/**
 * Actualiza los pesos y sesgos utilizando el promedio de los gradientes acumulados.
 * @param {Network} network 
 * @param {number} batchSize - Cantidad de muestras acumuladas en este lote.
 */
export function updateParameters(network, batchSize) {
    const PARAM_LIMIT = 100.0;
    
    const divisor = batchSize || 1; 

    const lossMethod = getLossMethod(network.lossTypeId);
    const multiplier = (lossMethod && lossMethod.multiplier !== undefined) ? lossMethod.multiplier : 1;

    for (let i = 1; i < network.layers.length; i++) {
        const layer = network.layers[i];

        layer.neurons.forEach(neuron => {
            if (neuron.isDropped) return;

            let biasGradient = (neuron.batchBiasGrad || 0) / divisor;

            neuron.biasVelocity = (network.momentum * neuron.biasVelocity) + (network.learningRate * biasGradient);
            neuron.bias += neuron.biasVelocity;
            neuron.bias = Math.max(Math.min(neuron.bias, PARAM_LIMIT), -PARAM_LIMIT);
            neuron.batchBiasGrad = 0;

            neuron.inputs.forEach(connection => {
                if (connection.isDropped || connection.from.isDropped) return;

                let weightGradient = (connection.batchWeightGrad || 0) / divisor;

                if (network.weightDecay > 0) {
                    weightGradient -= network.weightDecay * connection.weight;
                }

                connection.velocity = (network.momentum * connection.velocity) + (network.learningRate * weightGradient);
                connection.weight += connection.velocity;
                connection.weight = Math.max(Math.min(connection.weight, PARAM_LIMIT), -PARAM_LIMIT);
                connection.batchWeightGrad = 0;
            });
        });
    }
}