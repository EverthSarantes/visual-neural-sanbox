import { noneLogic } from './none.js';
import { reluLogic } from './relu.js';
import { sigmoidLogic } from './sigmoid.js';
import { tanhLogic } from './tanh.js';
import { softmaxLogic } from './softmax.js';

export const activationRegistry = {
    none: {
        id: 'none',
        name: 'Ninguna (Identidad)',
        description: 'No aplica ninguna transformación. La salida es igual a la entrada, lo que puede ser útil para capas ocultas o como opción de activación neutral.',
        calculate: noneLogic.feed,
        derivative: noneLogic.derivative
    },
    relu: {
        id: 'relu',
        name: 'ReLU',
        description: 'Modela umbrales biológicos eliminando valores negativos. Devuelve 0 si la entrada es negativa, o el mismo valor si es positiva.',
        calculate: reluLogic.feed,
        derivative: reluLogic.derivative
    },
    sigmoid: {
        id: 'sigmoid',
        name: 'Sigmoide',
        description: 'Curva suave en forma de "S" que transforma valores en un rango de 0 a 1. Ideal para decisiones probabilísticas finales.',
        calculate: sigmoidLogic.feed,
        derivative: sigmoidLogic.derivative
    },
    tanh: {
        id: 'tanh',
        name: 'Tangente Hiperbólica (Tanh)',
        description: 'Similar a la sigmoide pero con un rango de -1 a 1. Mantiene las señales balanceadas y con media cercana a cero.',
        calculate: tanhLogic.feed,
        derivative: tanhLogic.derivative
    },
    softmax: {
        id: 'softmax',
        name: 'Softmax (Capa Completa)',
        description: 'Función vectorial que hace competir a las neuronas de salida convirtiendo sus valores en una distribución de probabilidad que suma 100%.',
        calculate: softmaxLogic.feed,
        derivative: softmaxLogic.derivative,
        calculateLayer: softmaxLogic.calculateLayer,
        derivativeLayer: softmaxLogic.derivativeLayer,
        derivativeLayerHidden: softmaxLogic.derivativeLayerHidden
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