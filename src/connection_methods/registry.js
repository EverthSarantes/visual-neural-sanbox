import { connectDense } from './dense.js';
import { connectOneToOne } from './oneToOne.js';

export const connectionRegistry = {
    dense: {
        id: 'dense',
        name: 'Totalmente Conectada (Densa)',
        description: 'Cada neurona se conecta con todas las neuronas de la capa siguiente.',
        execute: connectDense
    },
    oneToOne: {
        id: 'oneToOne',
        name: 'Uno a Uno (Directa)',
        description: 'Conecta la neurona en la posición i de esta capa estrictamente con la neurona en la misma posición i de la capa siguiente.',
        execute: connectOneToOne
    }
};

/**
 * Busca y retorna el objeto de conexión completo por su ID.
 * @param {string} id 
 * @returns {Object|null}
 */
export function getConnectionMethod(id) {
    return connectionRegistry[id] || null;
}