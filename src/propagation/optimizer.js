/**
 * Captura los gradientes instantáneos de la fila actual del CSV y los suma
 * en variables temporales dentro de las neuronas y conexiones antes de que se borren.
 */
export function accumulateBatchGradients(network) {
    for (let i = 1; i < network.layers.length; i++) {
        const layer = network.layers[i];

        layer.neurons.forEach(neuron => {
            if (neuron.isDropped) return;

            if (neuron.batchBiasGrad === undefined) neuron.batchBiasGrad = 0;
            neuron.batchBiasGrad += neuron.errorSignal;

            neuron.inputs.forEach(connection => {
                if (connection.isDropped || connection.from.isDropped) return;

                const weightGradient = neuron.errorSignal * connection.from.value;

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
    const divisor = batchSize || 1;

    for (let i = 1; i < network.layers.length; i++) {
        const layer = network.layers[i];

        layer.neurons.forEach(neuron => {
            if (neuron.isDropped) return;

            const biasGradient = (neuron.batchBiasGrad || 0) / divisor;
            neuron.updateBias(biasGradient, network);
            neuron.batchBiasGrad = 0;

            neuron.inputs.forEach(connection => {
                if (connection.isDropped || connection.from.isDropped) return;

                const weightGradient = (connection.batchWeightGrad || 0) / divisor;
                connection.updateWeight(weightGradient, network);
                connection.batchWeightGrad = 0;
            });
        });
    }
}