export class Connection {
    /**
     * @param {string} id - Identificador único (ej: "c_l0_n0_to_l1_n2")
     * @param {Neuron} fromNeuron - Neurona de origen (capa anterior)
     * @param {Neuron} toNeuron - Neurona de destino (capa siguiente)
     */
    constructor(id, fromNeuron, toNeuron) {
        this.id = id;
        this.from = fromNeuron;
        this.to = toNeuron;

        this.weight = 0;
        this.velocity = 0;
        this.isDropped = false;
    }

    /**
     * Actualiza el peso usando un gradiente ya promediado por lote.
     * Encapsula la regla de actualización (momentum + L2 + clipping).
     */
    updateWeight(gradient, network) {
        const PARAM_LIMIT = 100.0;
        // gradient es la verdadera derivada parcial dL/dw
        let g = gradient;
        if (network.weightDecay > 0) {
            g += network.weightDecay * this.weight; // L2: gradiente += decay * peso
        }
        this.velocity = (network.momentum * this.velocity) + (network.learningRate * g);
        this.weight -= this.velocity; // descenso por gradiente
        this.weight = Math.max(Math.min(this.weight, PARAM_LIMIT), -PARAM_LIMIT);
    }
}