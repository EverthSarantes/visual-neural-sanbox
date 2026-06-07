export const softmaxLogic = {
    feed: (x) => 1,
    derivative: (x) => 1,

    calculateLayer: (layer, network) => {
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
    },

    derivativeLayer: (layer, targetVector, lossMethod) => {
        const lossTerms = layer.neurons.map((n, idx) => {
            if (n.isDropped) return 0;
            return lossMethod.errorSignalTerm(n.value, targetVector[idx] || 0);
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

            n.errorSignal = n.value * (lossTerms[idx] - sumLossTimesValue);
        });
    },

    derivativeLayerHidden: (layer, errorSums) => {
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
};