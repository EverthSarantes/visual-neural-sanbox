import { getActivation } from '../activation_methods/registry.js';

export class Neuron {
    /**
     * @param {string} id 
     * @param {string} activationId - ID de la activación por defecto (ej: 'relu')
     */
    constructor(id, activationId = 'relu') {
        this.id = id;
        this.activationId = activationId; //
        
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
     * Calcula la suma neta y ejecuta la activación matemática real.
     */
    activate() {
        // Sumatoria literal: sesgo + (cada peso por su valor de entrada)
        let sum = this.bias;
        for (let connection of this.inputs) {
            sum += connection.from.value * connection.weight;
        }
        
        this.netInput = sum;

        // 🧮 Consumo del registrador matemático centralizado
        const activation = getActivation(this.activationId);
        
        if (activation) {
            this.value = activation.feed(sum);
        } else {
            this.value = sum; // Fallback lineal por seguridad si no se encuentra
        }

        return this.value;
    }
}