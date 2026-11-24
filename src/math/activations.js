/**
 * Módulo matemático puro para funciones de activación.
 * Cada función recibe un escalar (o una capa, para Softmax) y devuelve
 * el valor transformado. No tiene efectos secundarios.
 */

// ---------- Identidad ----------
export function identity(x) {
    return x;
}
export function identityDerivative(_x) {
    return 1;
}

// ---------- ReLU ----------
export function relu(x) {
    return x > 0 ? x : 0;
}
export function reluDerivative(x) {
    return x > 0 ? 1 : 0;
}

// ---------- Sigmoide ----------
export function sigmoid(x) {
    const safeX = Math.max(Math.min(x, 50), -50);
    const s = 1 / (1 + Math.exp(-safeX));
    return Math.max(Math.min(s, 1 - 1e-7), 1e-7);
}
export function sigmoidDerivative(x) {
    const safeX = Math.max(Math.min(x, 50), -50);
    const s = 1 / (1 + Math.exp(-safeX));
    const p = Math.max(Math.min(s, 1 - 1e-7), 1e-7);
    return p * (1 - p);
}

// ---------- Tangente hiperbólica ----------
export function tanh(x) {
    return Math.tanh(x);
}
export function tanhDerivative(x) {
    const t = Math.tanh(x);
    return 1 - (t * t);
}

// ---------- Softmax (operaciones a nivel de capa) ----------
export function softmaxCalculateLayer(layer, network) {
    let maxNet = -Infinity;
    layer.neurons.forEach(n => {
        if (!n.isDropped && n.netInput > maxNet) maxNet = n.netInput;
    });

    let sumExp = 0;
    layer.neurons.forEach(n => {
        if (n.isDropped) return;
        n.tmpExp = Math.exp(n.netInput - maxNet);
        sumExp += n.tmpExp;
    });

    if (sumExp === 0) sumExp = 1e-7;

    layer.neurons.forEach(n => {
        if (n.isDropped) {
            n.value = 0;
            return;
        }

        let activationValue = n.tmpExp / sumExp;
        activationValue = Math.max(Math.min(activationValue, 1 - 1e-7), 1e-7);

        if (network.isTraining && network.dropoutRate > 0 && layer.type === 'hidden') {
            activationValue /= (1 - network.dropoutRate);
        }

        n.value = activationValue;
        delete n.tmpExp;
    });
}

// Derivada de la capa de salida Softmax combinada con la pérdida (produce -dL/dz)
export function softmaxOutputDelta(layer, targetVector, lossErrorSignalTerm) {
    // lossErrorSignalTerm(actual, target) debe devolver el término de la pérdida
    const lossTerms = layer.neurons.map((n, idx) => {
        if (n.isDropped) return 0;
        return lossErrorSignalTerm(n.value, targetVector[idx] || 0);
    });

    let sumLossTimesValue = 0;
    layer.neurons.forEach((n, idx) => {
        if (n.isDropped) return;
        sumLossTimesValue += lossTerms[idx] * n.value;
    });

    layer.neurons.forEach((n, idx) => {
        if (n.isDropped) {
            n.errorSignal = 0;
            return;
        }
        // Verdadero gradiente dL/dz para Softmax + pérdida
        n.errorSignal = n.value * (lossTerms[idx] - sumLossTimesValue);
    });
}

// Derivada para capas ocultas con activación Softmax
export function softmaxHiddenDelta(layer, errorSums) {
    let sumErrorTimesValue = 0;
    layer.neurons.forEach((n, idx) => {
        if (n.isDropped) return;
        sumErrorTimesValue += errorSums[idx] * n.value;
    });

    layer.neurons.forEach((n, idx) => {
        if (n.isDropped) {
            n.errorSignal = 0;
            return;
        }
        n.errorSignal = n.value * (errorSums[idx] - sumErrorTimesValue);
    });
}

// Mapa de activaciones escalares (para uso en registros)
export const scalarActivations = {
    none: { feed: identity, derivative: identityDerivative },
    relu: { feed: relu, derivative: reluDerivative },
    sigmoid: { feed: sigmoid, derivative: sigmoidDerivative },
    tanh: { feed: tanh, derivative: tanhDerivative },
};