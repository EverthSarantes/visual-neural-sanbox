export class TrainingClient {
    constructor() {
        this.worker = new Worker(new URL('./training.worker.js', import.meta.url), { type: 'module' });
        this.listeners = {
            ready: [],
            epoch: [],
            paused: [],
            resetDone: [],
            error: [],
        };
        this.worker.onmessage = (e) => {
            const { type, payload } = e.data || {};
            const cbs = this.listeners[type];
            if (cbs) cbs.forEach(cb => cb(payload));
        };
        this.worker.onerror = (err) => {
            this.listeners.error.forEach(cb => cb({ message: err?.message || String(err) }));
        };
    }

    on(type, cb) {
        if (!this.listeners[type]) this.listeners[type] = [];
        this.listeners[type].push(cb);
        return () => {
            this.listeners[type] = this.listeners[type].filter(x => x !== cb);
        };
    }

    init(payload) {
        this.worker.postMessage({ type: 'init', payload });
    }

    start() { this.worker.postMessage({ type: 'start' }); }
    pause() { this.worker.postMessage({ type: 'pause' }); }
    step() { this.worker.postMessage({ type: 'step' }); }
    reset() { this.worker.postMessage({ type: 'reset' }); }
    setHyperparams(payload) { this.worker.postMessage({ type: 'setHyperparams', payload }); }
    setDatasets(trainSet, testSet) { this.worker.postMessage({ type: 'setDatasets', payload: { trainSet, testSet } }); }
    reinit(strategy) { this.worker.postMessage({ type: 'reinit', payload: { strategy } }); }
    setNetwork(network) {
        // Serializa la red actual y la envía al worker para sincronizar pesos editados
        const snapshot = {
            layers: network.layers.map(layer => ({
                id: layer.id,
                neurons: layer.neurons.map(neuron => ({
                    id: neuron.id,
                    activationId: neuron.activationId,
                    bias: neuron.bias,
                    biasVelocity: neuron.biasVelocity,
                    inputs: neuron.inputs.map(conn => ({
                        id: conn.id,
                        weight: conn.weight,
                        velocity: conn.velocity,
                        isDropped: conn.isDropped,
                    }))
                }))
            })),
            learningRate: network.learningRate,
            momentum: network.momentum,
            weightDecay: network.weightDecay,
            dropoutRate: network.dropoutRate,
            batchSize: network.batchSize,
            lossTypeId: network.lossTypeId,
        };
        this.worker.postMessage({ type: 'setNetwork', payload: { network: snapshot } });
    }
    terminate() { this.worker.postMessage({ type: 'terminate' }); this.worker.terminate(); }
}
