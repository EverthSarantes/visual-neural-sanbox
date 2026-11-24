import { getActivation } from '../activation_methods/registry.js';

export class Neuron {
    /**
     * @param {string} id
     * @param {string} activationId - ID de la activación por defecto (ej: 'relu')
     */
    constructor(id, activationId = 'relu') {
        this.id = id;
        this.activationId = activationId;

        this.bias = 0;
        this.value = 0;                    // Inicializado en 0 antes del forward pass
        this.netInput = 0;                 // Almacena la suma neta
        this.errorSignal = 0;              // Almacena el gradiente local durante el backprop

        this.inputs = [];
        this.outputs = [];

        this.biasVelocity = 0;
        this.isDropped = false;

        // Coordenadas para el CanvasRenderer
        this.x = 0;
        this.y = 0;
    }

    /**
     * Calcula la suma neta (bias + Σ peso·valor) y la almacena en netInput.
     * @returns {number}
     */
    computeNetInput() {
        let sum = this.bias;
        for (const connection of this.inputs) {
            if (!connection.from.isDropped && !connection.isDropped) {
                sum += connection.from.value * connection.weight;
            }
        }
        this.netInput = sum;
        return sum;
    }

    /**
     * Aplica la función de activación configurada sobre un valor escalar.
     * @param {number} x
     * @returns {number}
     */
    applyActivation(x) {
        const activation = getActivation(this.activationId);
        if (activation && typeof activation.calculate === 'function') {
            return activation.calculate(x);
        }
        return x;
    }

    /**
     * Calcula la suma neta y ejecuta la activación matemática real.
     * Representa físicamente el comportamiento de una neurona.
     */
    activate() {
        const sum = this.computeNetInput();
        const activation = getActivation(this.activationId);
        if (activation && typeof activation.calculate === 'function') {
            this.value = activation.calculate(sum);
        } else {
            this.value = sum;
        }
        return this.value;
    }

    /**
     * Actualiza el sesgo usando un gradiente ya promediado por lote.
     */
    updateBias(gradient, network) {
        const PARAM_LIMIT = 100.0;
        // gradient es la verdadera derivada parcial dL/dbias
        this.biasVelocity = (network.momentum * this.biasVelocity) + (network.learningRate * gradient);
        this.bias -= this.biasVelocity;
        this.bias = Math.max(Math.min(this.bias, PARAM_LIMIT), -PARAM_LIMIT);
    }
}