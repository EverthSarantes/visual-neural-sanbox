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
}