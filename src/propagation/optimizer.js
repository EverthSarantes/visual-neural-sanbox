/**
 * Actualiza los pesos y sesgos de la red utilizando las señales de error calculadas,
 * aplicando Tasa de Aprendizaje (LR), Momento y Regularización L2 (Weight Decay).
 * @param {Network} network - La instancia maestra de la red.
 */
export function updateParameters(network) {
    const PARAM_LIMIT = 100.0;

    for (let i = 1; i < network.layers.length; i++) {
        const layer = network.layers[i];

        layer.neurons.forEach(neuron => {
            if (neuron.isDropped) return;

            // ACTUALIZAR SESGO (BIAS) CON MOMENTO
            const biasGradient = neuron.errorSignal;
            neuron.biasVelocity = (network.momentum * neuron.biasVelocity) + (network.learningRate * biasGradient);
            neuron.bias += neuron.biasVelocity;

            neuron.bias = Math.max(Math.min(neuron.bias, PARAM_LIMIT), -PARAM_LIMIT);

            // ACTUALIZAR PESOS DE LAS CONEXIONES ENTRANTES
            neuron.inputs.forEach(connection => {
                if (connection.isDropped || connection.from.isDropped) return;

                let weightGradient = neuron.errorSignal * connection.from.value;

                if (network.weightDecay > 0) {
                    weightGradient -= network.weightDecay * connection.weight;
                }

                connection.velocity = (network.momentum * connection.velocity) + (network.learningRate * weightGradient);
                connection.weight += connection.velocity;

                connection.weight = Math.max(Math.min(connection.weight, PARAM_LIMIT), -PARAM_LIMIT);
            });
        });
    }
}