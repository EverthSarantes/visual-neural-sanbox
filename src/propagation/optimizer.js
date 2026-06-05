/**
 * Actualiza los pesos y sesgos de la red utilizando las señales de error calculadas,
 * aplicando Tasa de Aprendizaje (LR), Momento y Regularización L2 (Weight Decay).
 * @param {Network} network - La instancia maestra de la red.
 */
export function updateParameters(network) {
    // Recorremos desde la capa 1 en adelante (la capa de entrada 0 no tiene conexiones entrantes ni sesgos)
    for (let i = 1; i < network.layers.length; i++) {
        const layer = network.layers[i];

        layer.neurons.forEach(neuron => {
            //  Si la neurona durmió en esta iteración, sus parámetros quedan congelados
            if (neuron.isDropped) return;

            // ACTUALIZAR SESGO (BIAS) CON MOMENTO
            // El gradiente del sesgo es directamente la señal de error de la neurona
            const biasGradient = neuron.errorSignal;

            // Ecuación de inercia: v_b = (α * v_b) + (η * gradiente)
            neuron.biasVelocity = (network.momentum * neuron.biasVelocity) + (network.learningRate * biasGradient);
            
            // Modificar el sesgo físico
            neuron.bias += neuron.biasVelocity;


            // ACTUALIZAR PESOS DE LAS CONEXIONES ENTRANTES
            neuron.inputs.forEach(connection => {
                // Si la conexión esta apagada o su origen está apagado, nos lo saltamos
                if (connection.isDropped || connection.from.isDropped) return;

                // El gradiente del peso combina el error del destino con la activación del origen
                let weightGradient = neuron.errorSignal * connection.from.value;

                // Aplicar Regularización L2 (Weight Decay)
                if (network.weightDecay > 0) {
                    weightGradient -= network.weightDecay * connection.weight;
                }

                // Ecuación de inercia para el peso: v_w = (α * v_w) + (η * gradiente)
                connection.velocity = (network.momentum * connection.velocity) + (network.learningRate * weightGradient);

                // Modificar el peso físico
                connection.weight += connection.velocity;
            });
        });
    }
}