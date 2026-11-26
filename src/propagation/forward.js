import { getActivation } from '../activation_methods/registry.js';
import { Neuron } from '../network/Neuron.js';
import { Connection } from '../network/Connection.js';
import { conv2dForward, maxpoolForward } from '../math/layers.js';

export function forwardPass(network, inputVector) {
    if (network.layers.length === 0) return [];

    // Determinar si es red convolucional (tiene capas conv/pool/flatten)
    const isConvNet = network.layers.some(l => l.layerType === 'conv2d' || l.layerType === 'maxpool2d' || l.layerType === 'flatten');

    if (!isConvNet) {
        // ===== MLP PATH (lógica original, sin cambios) =====
        const inputLayer = network.layers[0];
        inputLayer.neurons.forEach((neuron, idx) => {
            neuron.value = inputVector[idx] || 0;
            neuron.netInput = neuron.value;
            neuron.isDropped = false;
        });

        if (network.isTraining && network.dropoutRate > 0) {
            for (let i = 1; i < network.layers.length - 1; i++) {
                const layer = network.layers[i];
                layer.neurons.forEach(neuron => { neuron.isDropped = Math.random() < network.dropoutRate; });
            }
        } else {
            network.layers.forEach(layer => layer.neurons.forEach(n => n.isDropped = false));
        }

        for (let i = 1; i < network.layers.length; i++) {
            const layer = network.layers[i];
            const activationMethod = getActivation(layer.activationId);
            layer.neurons.forEach(neuron => {
                if (neuron.isDropped) { neuron.netInput = 0; neuron.value = 0; return; }
                neuron.computeNetInput();
            });
            if (activationMethod.calculateLayer) {
                activationMethod.calculateLayer(layer, network);
            } else {
                layer.neurons.forEach(neuron => {
                    if (neuron.isDropped) return;
                    let v = activationMethod.calculate(neuron.netInput);
                    if (network.isTraining && network.dropoutRate > 0 && layer.type === 'hidden') v /= (1 - network.dropoutRate);
                    neuron.value = v;
                });
            }
        }

        const outputLayer = network.layers[network.layers.length - 1];
        return outputLayer.neurons.map(n => n.value);
    }

    // ===== CONV PATH =====
    // Input: imagen aplanada. Reconstruir shape.
    const imgShape = network.inputShape || null;
    let total = inputVector.length;
    let shape, current;
    if (imgShape) {
        shape = [1, imgShape[0], imgShape[1], imgShape[2]];
    } else {
        const side = Math.round(Math.sqrt(total / 3));
        shape = [1, 3, side, side];
    }
    current = new Float32Array(inputVector);

    for (let i = 0; i < network.layers.length; i++) {
        const layer = network.layers[i];

        if (layer.layerType === 'conv2d') {
            const p = layer.params || {};
            const [N, C, H, W] = shape;
            const k = p.kernelSize || 3;
            const pad = p.padding !== undefined ? p.padding : 1;
            const stride = p.stride || 1;
            const filters = p.filters || 8;
            if (!layer.weights) {
                const fanIn = C * k * k;
                const scale = Math.sqrt(2 / fanIn);
                layer.weights = new Float32Array(filters * C * k * k);
                for (let j = 0; j < layer.weights.length; j++) layer.weights[j] = (Math.random() * 2 - 1) * scale;
                layer.bias = new Float32Array(filters);
                layer.gradW = new Float32Array(layer.weights.length);
                layer.gradB = new Float32Array(filters);
            }
            const inputCopy = new Float32Array(current);
            const res = conv2dForward(current, layer.weights, layer.bias, C, H, W, filters, k, k, stride, pad);
            current = res.data;
            shape = res.shape;
            layer.cache = { inputData: inputCopy, inputShape: [N, C, H, W], output: current, outputShape: shape, params: { inC: C, inH: H, inW: W, outC: filters, kH: k, kW: k, stride, pad } };
        } else if (layer.layerType === 'maxpool2d') {
            const [N, C, H, W] = shape;
            const p = layer.params || {};
            const pool = p.poolSize || 2;
            const stride = p.stride || pool;
            const Hout = Math.floor((H - pool) / stride) + 1;
            const Wout = Math.floor((W - pool) / stride) + 1;
            const out = new Float32Array(N * C * Hout * Wout);
            const indices = new Int32Array(out.length);
            for (let n = 0; n < N; n++) {
                for (let c = 0; c < C; c++) {
                    for (let oh = 0; oh < Hout; oh++) {
                        for (let ow = 0; ow < Wout; ow++) {
                            let maxVal = -Infinity; let maxIdx = -1;
                            for (let ph = 0; ph < pool; ph++) {
                                for (let pw = 0; pw < pool; pw++) {
                                    const ih = oh * stride + ph;
                                    const iw = ow * stride + pw;
                                    const idx = ((n * C + c) * H + ih) * W + iw;
                                    if (current[idx] > maxVal) { maxVal = current[idx]; maxIdx = idx; }
                                }
                            }
                            const outIdx = ((n * C + c) * Hout + oh) * Wout + ow;
                            out[outIdx] = maxVal;
                            indices[outIdx] = maxIdx;
                        }
                    }
                }
            }
            layer.cache = { inputShape: [N, C, H, W], outputShape: [N, C, Hout, Wout], indices, params: { pool, stride, H, W } };
            current = out;
            shape = [N, C, Hout, Wout];
        } else if (layer.layerType === 'flatten') {
            layer.cache = { inputShape: shape.slice(), outputShape: [shape[0], current.length / shape[0]] };
            // Create virtual neurons for rendering/connections
            const flatSize = current.length / shape[0];
            if (layer.neurons.length !== flatSize) {
                layer.neurons = [];
                layer.neuronSeed = 0;
                for (let i = 0; i < flatSize; i++) {
                    const n = new Neuron(`n_${layer.id}_${i}`, 'relu');
                    n.value = current[i];
                    layer.neurons.push(n);
                }
            } else {
                layer.neurons.forEach((n, i) => { n.value = current[i]; n.netInput = current[i]; });
            }
        } else if (layer.layerType === 'linear' || layer.type === 'linear') {
            // Dense layer: neurons use connections (MLP logic). But in conv net, previous is tensor.
            // Bridge: if current is 4D/2D tensor, set flatten neuron values? 
            // For simplicity, treat current as flat input and compute dense manually.
            const n = shape[0];
            // Initialize weights if needed (tensor-based dense)
            if (!layer.weights) {
                const inF0 = (current.length / n) || 1;
                const outF0 = layer.neurons.length;
                const scale = Math.sqrt(2 / inF0);
                layer.weights = new Float32Array(inF0 * outF0);
                for (let j = 0; j < layer.weights.length; j++) layer.weights[j] = (Math.random() * 2 - 1) * scale;
                layer.bias = new Float32Array(outF0);
                layer.gradW = new Float32Array(inF0 * outF0);
                layer.gradB = new Float32Array(outF0);
            }
            if (layer.bias) layer.neurons.forEach((nn, idx) => { nn.bias = layer.bias[idx]; });

            // Flatten current if needed
            let flatCurrent = current;
            let flatInF = current.length / shape[0];
            if (shape.length > 2) {
                flatInF = shape.reduce((a, b) => a * b, 1) / shape[0];
                flatCurrent = new Float32Array(shape[0] * flatInF);
                flatCurrent.set(current.subarray(0, shape[0] * flatInF));
            }
            const inF = (layer.weights) ? (layer.weights.length / layer.neurons.length) : flatInF;
            const outF = layer.neurons.length;

            // use flatCurrent in computation below
            // (replace current references) -- we'll set current = flatCurrent for this block
            current = flatCurrent;
            shape = [n, flatInF];
            // Initialize weights if needed
            if (!layer.weights) {
                const scale = Math.sqrt(2 / inF);
                layer.weights = new Float32Array(inF * outF);
                for (let j = 0; j < layer.weights.length; j++) layer.weights[j] = (Math.random() * 2 - 1) * scale;
                layer.bias = new Float32Array(outF);
                layer.gradW = new Float32Array(inF * outF);
                layer.gradB = new Float32Array(outF);
            }
            const preAct = new Float32Array(n * outF);
            const out = new Float32Array(n * outF);
            for (let b = 0; b < n; b++) {
                for (let o = 0; o < outF; o++) {
                    let sum = layer.bias[o];
                    for (let i = 0; i < inF; i++) {
                        sum += current[b * inF + i] * layer.weights[i * outF + o];
                    }
                    preAct[b * outF + o] = sum;
                }
            }
            const activationMethod = getActivation(layer.activationId);
            for (let i = 0; i < out.length; i++) {
                out[i] = activationMethod.calculate(preAct[i]);
            }
            // set neuron values for renderer
            layer.neurons.forEach((neuron, idx) => {
                neuron.value = out[idx];
                neuron.netInput = preAct[idx];
            });
            current = out;
            shape = [n, outF];
            layer.cache = { input: new Float32Array(flatCurrent), inputShape: [n, inF], output: current, outputShape: shape };
        }
    }

    const outputLayer = network.layers[network.layers.length - 1];
    if (outputLayer.layerType === 'linear' || outputLayer.type === 'linear') {
        if (outputLayer.neurons && outputLayer.neurons.length > 0) {
            return outputLayer.neurons.map(n => n.value);
        }
    }
    return Array.from(current);
}