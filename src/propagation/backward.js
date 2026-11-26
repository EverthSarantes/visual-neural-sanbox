import { getActivation } from '../activation_methods/registry.js';
import { getLossMethod } from '../loss_methods/registry.js';
import { conv2dBackward } from '../math/layers.js';

export function backwardPass(network, targetVector) {
    if (network.layers.length < 2) return;

    const isConvNet = network.layers.some(l => l.layerType === 'conv2d' || l.layerType === 'maxpool2d' || l.layerType === 'flatten');

    if (!isConvNet) {
        // ===== MLP PATH (original) =====
        const totalLayers = network.layers.length;
        const outputLayer = network.layers[totalLayers - 1];
        const lossMethod = getLossMethod(network.lossTypeId);
        const outputActivationMethod = getActivation(outputLayer.activationId);
        const GRAD_LIMIT = 10.0;
        let sampleWeight = 1.0;

        if (network.useClassWeighting && network.classWeights && network.classWeights.length > 0) {
            if (outputLayer.neurons.length === 1) {
                sampleWeight = network.classWeights[0] || 1.0;
            } else {
                const trueClassIdx = targetVector.indexOf(1);
                if (trueClassIdx !== -1 && network.classWeights[trueClassIdx] !== undefined) {
                    sampleWeight = network.classWeights[trueClassIdx];
                }
            }
        }

        if (outputActivationMethod.derivativeLayer) {
            outputActivationMethod.derivativeLayer(outputLayer, targetVector, lossMethod);
            outputLayer.neurons.forEach(neuron => {
                neuron.errorSignal = neuron.errorSignal * sampleWeight;
                neuron.errorSignal = Math.max(Math.min(neuron.errorSignal, GRAD_LIMIT), -GRAD_LIMIT);
            });
        } else {
            outputLayer.neurons.forEach((neuron, idx) => {
                if (neuron.isDropped) { neuron.errorSignal = 0; return; }
                const lossTerm = lossMethod.errorSignalTerm(neuron.value, targetVector[idx] || 0);
                let gradient = lossTerm * outputActivationMethod.derivative(neuron.netInput);
                neuron.errorSignal = gradient * sampleWeight;
                neuron.errorSignal = Math.max(Math.min(neuron.errorSignal, GRAD_LIMIT), -GRAD_LIMIT);
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
                layer.neurons.forEach(neuron => {
                    neuron.errorSignal = Math.max(Math.min(neuron.errorSignal, GRAD_LIMIT), -GRAD_LIMIT);
                });
            } else {
                layer.neurons.forEach((neuron, idx) => {
                    if (neuron.isDropped) { neuron.errorSignal = 0; return; }
                    neuron.errorSignal = errorSums[idx] * activationMethod.derivative(neuron.netInput);
                    neuron.errorSignal = Math.max(Math.min(neuron.errorSignal, GRAD_LIMIT), -GRAD_LIMIT);
                });
            }
        }
        return;
    }

    // ===== CONV PATH =====
    const GRAD_LIMIT = 10.0;
    const outputLayer = network.layers[network.layers.length - 1];

    // Output gradient for dense output
    let grad = null;
    if (outputLayer.layerType === 'linear') {
        const outF = outputLayer.neurons.length;
        grad = new Float32Array(outF);
        const lossMethod = getLossMethod(network.lossTypeId);
        const outAct = getActivation(outputLayer.activationId);
        const inF = outputLayer.cache ? outputLayer.cache.inputShape[1] : 1;
        const gradInput = new Float32Array(inF);
        for (let o = 0; o < outF; o++) {
            const neuron = outputLayer.neurons[o];
            const lossTerm = lossMethod.errorSignalTerm(neuron.value, targetVector[o] || 0);
            grad[o] = lossTerm * outAct.derivative(neuron.netInput);
            grad[o] = Math.max(Math.min(grad[o], GRAD_LIMIT), -GRAD_LIMIT);
            neuron.errorSignal = grad[o];
            // accumulate grads
            if (!outputLayer.gradW) outputLayer.gradW = new Float32Array(inF * outF);
            if (!outputLayer.gradB) outputLayer.gradB = new Float32Array(outF);
            outputLayer.gradB[o] += grad[o];
            for (let i = 0; i < inF; i++) {
                outputLayer.gradW[i * outF + o] += outputLayer.cache.input[i] * grad[o];
                gradInput[i] += outputLayer.weights[i * outF + o] * grad[o];
            }
        }
        outputLayer.cache.gradInput = gradInput;
    }

    // Backprop in reverse
    for (let i = network.layers.length - 2; i >= 0; i--) {
        const layer = network.layers[i];
        const nextLayer = network.layers[i + 1];
        const lt = layer.layerType || 'linear';

        if (lt === 'linear' && layer.weights) {
            // Tensor-based dense backward
            const outF = layer.neurons.length;
            const inF = layer.cache ? layer.cache.inputShape[1] : 1;
            const gradInput = new Float32Array(inF);
            if (!layer.gradW) layer.gradW = new Float32Array(inF * outF);
            if (!layer.gradB) layer.gradB = new Float32Array(outF);
            const gradW = layer.gradW;
            const gradB = layer.gradB;
            for (let o = 0; o < outF; o++) {
                const g = grad[o]; // gradient from next layer
                layer.neurons[o].errorSignal = g;
                gradB[o] += g;
                for (let i = 0; i < inF; i++) {
                    gradW[i * outF + o] += layer.cache.input[i] * g;
                    gradInput[i] += layer.weights[i * outF + o] * g;
                }
            }
            layer.cache.gradInput = gradInput;
            // grad for previous layer
            grad = gradInput;
        } else if (lt === 'linear') {
            // Neuron-based dense (MLP) backward
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
            const activationMethod = getActivation(layer.activationId);
            layer.neurons.forEach((neuron, idx) => {
                if (neuron.isDropped) { neuron.errorSignal = 0; return; }
                neuron.errorSignal = errorSums[idx] * activationMethod.derivative(neuron.netInput);
                neuron.errorSignal = Math.max(Math.min(neuron.errorSignal, GRAD_LIMIT), -GRAD_LIMIT);
            });
        } else if (lt === 'flatten') {
            // Flatten: grad passes through; but dense backward already computed grad for flatten
            // The flatten layer's gradOutput is the dense gradInput (stored in nextLayer cache)
            // For simplicity, flatten has no grad to propagate further (identity)
            // grad for conv = nextLayer's gradInput reshaped
            if (nextLayer.layerType === 'linear' && nextLayer.cache && nextLayer.cache.gradInput) {
                grad = nextLayer.cache.gradInput;
            }
            layer.cache.gradOutput = grad ? new Float32Array(grad) : null;
        } else if (lt === 'maxpool2d') {
            if (layer.cache && grad) {
                const gradInput = new Float32Array(layer.cache.inputShape.reduce((a, b) => a * b, 1));
                const indices = layer.cache.indices;
                for (let j = 0; j < grad.length; j++) {
                    const idx = indices[j];
                    if (idx >= 0) gradInput[idx] += grad[j];
                }
                layer.cache.gradOutput = gradInput;
                grad = gradInput;
            }
        } else if (lt === 'conv2d') {
            if (layer.cache) {
                const p = layer.cache.params;
                const input = layer.cache.inputData;
                const out = layer.cache.output;
                // Gate grad by ReLU
                const gatedGrad = new Float32Array(grad.length);
                for (let i = 0; i < grad.length; i++) {
                    gatedGrad[i] = out[i] > 0 ? grad[i] : 0;
                }
                const { gradInput, gradW, gradB } = conv2dBackward(
                    gatedGrad, input, layer.weights, p.inC, p.inH, p.inW, p.outC, p.kH, p.kW, p.stride, p.pad
                );
                if (!layer.gradW) layer.gradW = new Float32Array(gradW.length);
                if (!layer.gradB) layer.gradB = new Float32Array(gradB.length);
                for (let i = 0; i < gradW.length; i++) layer.gradW[i] += gradW[i];
                for (let i = 0; i < gradB.length; i++) layer.gradB[i] += gradB[i];
                layer.cache.gradOutput = gradInput;
                grad = gradInput;
            }
        }
    }
}