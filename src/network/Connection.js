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
        
        // El peso inicial aleatorio servirá para que el Canvas pinte líneas de grosores variados desde el inicio
        this.weight = Math.random() * 2 - 1; // Entre -1 y 1
    }
}