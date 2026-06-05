import { reluLogic } from './relu.js';
import { sigmoidLogic } from './sigmoid.js';
import { tanhLogic } from './tanh.js';

export const activationRegistry = {
    relu: {
        id: 'relu',
        name: 'ReLU',
        description: 'Modela umbrales biológicos eliminando valores negativos. Devuelve 0 si la entrada es negativa, o el mismo valor si es positiva.',
        feed: reluLogic.feed,
        derivative: reluLogic.derivative
    },
    sigmoid: {
        id: 'sigmoid',
        name: 'Sigmoide',
        description: 'Curva suave en forma de "S" que transforma valores en un rango de 0 a 1. Ideal para decisiones probabilísticas finales.',
        feed: sigmoidLogic.feed,
        derivative: sigmoidLogic.derivative
    },
    tanh: {
        id: 'tanh',
        name: 'Tangente Hiperbólica (Tanh)',
        description: 'Similar a la sigmoide pero con un rango de -1 a 1. Mantiene las señales balanceadas y con media cercana a cero.',
        feed: tanhLogic.feed,
        derivative: tanhLogic.derivative
    }
};

/**
 * Busca y retorna el objeto de activación completo por su ID.
 * @param {string} id 
 * @returns {Object|null}
 */
export function getActivation(id) {
    return activationRegistry[id] || null;
}