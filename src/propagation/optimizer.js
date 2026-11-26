export function accumulateBatchGradients(network) {
    // MLP path: accumulate into neuron/connection
    const isConvNet = network.layers.some(l => l.layerType === 'conv2d' || l.layerType === 'maxpool2d' || l.layerType === 'flatten');
    if (!isConvNet) {
        // original

        for (let i = 1; i < network.layers.length; i++) {
            const layer = network.layers[i];
            layer.neurons.forEach(neuron => {
                if (neuron.isDropped) return;
                if (neuron.batchBiasGrad === undefined) neuron.batchBiasGrad = 0;
                neuron.batchBiasGrad += neuron.errorSignal;
                neuron.inputs.forEach(connection => {
                    if (connection.isDropped || connection.from.isDropped) return;
                    if (connection.batchWeightGrad === undefined) connection.batchWeightGrad = 0;
                    connection.batchWeightGrad += neuron.errorSignal * connection.from.value;
                });
            });
        }
        return;
    }
    // Conv path: gradients already accumulated in layer.gradW/gradB during backward.
    // Nothing extra needed; updateParameters will use them.
}

export function updateParameters(network, batchSize) {
    const divisor = batchSize || 1;
    const isConvNet = network.layers.some(l => l.layerType === 'conv2d' || l.layerType === 'maxpool2d' || l.layerType === 'flatten');

    if (!isConvNet) {
        // original MLP path
        const PARAM_LIMIT = 100.0;
        for (let i = 1; i < network.layers.length; i++) {
            const layer = network.layers[i];
            layer.neurons.forEach(neuron => {
                if (neuron.isDropped) return;
                let biasGradient = (neuron.batchBiasGrad || 0) / divisor;
                neuron.biasVelocity = (network.momentum * neuron.biasVelocity) + (network.learningRate * biasGradient);
                neuron.bias -= neuron.biasVelocity;
                neuron.bias = Math.max(Math.min(neuron.bias, PARAM_LIMIT), -PARAM_LIMIT);
                neuron.batchBiasGrad = 0;
                neuron.inputs.forEach(connection => {
                    if (connection.isDropped || connection.from.isDropped) return;
                    let weightGradient = (connection.batchWeightGrad || 0) / divisor;
                    if (network.weightDecay > 0) weightGradient += network.weightDecay * connection.weight;
                    connection.velocity = (network.momentum * connection.velocity) + (network.learningRate * weightGradient);
                    connection.weight -= connection.velocity;
                    connection.weight = Math.max(Math.min(connection.weight, PARAM_LIMIT), -PARAM_LIMIT);
                    connection.batchWeightGrad = 0;
                });
            });
        }
        return;
    }

    // Conv path: update conv weights and dense tensor weights
    const PARAM_LIMIT = 100.0;
    const lr = network.learningRate;
    for (const layer of network.layers) {
        if (layer.layerType === 'conv2d' && layer.gradW) {
            for (let i = 0; i < layer.weights.length; i++) {
                let g = layer.gradW[i] / divisor;
                if (network.weightDecay > 0) g += network.weightDecay * layer.weights[i];
                layer.weights[i] -= lr * g;
                layer.weights[i] = Math.max(Math.min(layer.weights[i], PARAM_LIMIT), -PARAM_LIMIT);
            }
            for (let i = 0; i < layer.bias.length; i++) {
                let g = layer.gradB[i] / divisor;
                layer.bias[i] -= lr * g;
                layer.bias[i] = Math.max(Math.min(layer.bias[i], PARAM_LIMIT), -PARAM_LIMIT);
            }
            layer.gradW.fill(0);
            layer.gradB.fill(0);
        } else if (layer.layerType === 'linear' && layer.weights) {
            // reset dense tensor grads
            if (layer.gradW) layer.gradW.fill(0);
            if (layer.gradB) layer.gradB.fill(0);
            const outF = layer.neurons.length;
            const inF = layer.weights.length / outF;
            for (let i = 0; i < layer.weights.length; i++) {
                let g = layer.gradW[i] / divisor;
                if (network.weightDecay > 0) g += network.weightDecay * layer.weights[i];
                layer.weights[i] -= lr * g;
                layer.weights[i] = Math.max(Math.min(layer.weights[i], PARAM_LIMIT), -PARAM_LIMIT);
            }
            for (let i = 0; i < layer.bias.length; i++) {
                let g = layer.gradB[i] / divisor;
                layer.bias[i] -= lr * g;
                layer.bias[i] = Math.max(Math.min(layer.bias[i], PARAM_LIMIT), -PARAM_LIMIT);
            }
        }
    }
    // Dense layers in conv net use neuron connections (MLP logic) — update via neuron path
    for (let i = 1; i < network.layers.length; i++) {
        const layer = network.layers[i];
        if (layer.layerType === 'linear' && layer.neurons && layer.neurons.length > 0 && !layer.weights) {
            layer.neurons.forEach(neuron => {
                if (neuron.isDropped) return;
                let biasGradient = (neuron.batchBiasGrad || 0) / divisor;
                neuron.biasVelocity = (network.momentum * neuron.biasVelocity) + (network.learningRate * biasGradient);
                neuron.bias -= neuron.biasVelocity;
                neuron.bias = Math.max(Math.min(neuron.bias, PARAM_LIMIT), -PARAM_LIMIT);
                neuron.batchBiasGrad = 0;
                neuron.inputs.forEach(connection => {
                    if (connection.isDropped || connection.from.isDropped) return;
                    let weightGradient = (connection.batchWeightGrad || 0) / divisor;
                    if (network.weightDecay > 0) weightGradient += network.weightDecay * connection.weight;
                    connection.velocity = (network.momentum * connection.velocity) + (network.learningRate * weightGradient);
                    connection.weight -= connection.velocity;
                    connection.weight = Math.max(Math.min(connection.weight, PARAM_LIMIT), -PARAM_LIMIT);
                    connection.batchWeightGrad = 0;
                });
            });
        }
    }
}

