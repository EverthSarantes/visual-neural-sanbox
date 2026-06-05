import { random } from './random.js';
import { xavier } from './xavier.js';
import { he } from './he.js';

export const initializationRegistry = {
    random: {
        id: 'random',
        name: 'Aleatoria Uniforme',
        description: 'Inicializa valores puramente al azar entre -1 y 1. Tradicional pero propensa a saturar capas grandes.',
        generateWeight: random.generate
    },
    xavier: {
        id: 'xavier',
        name: 'Xavier / Glorot (Uniforme)',
        description: 'Optimiza el rango basándose en la suma de entradas y salidas de la capa. Es la reina absoluta para funciones de activación Sigmoide y Tanh.',
        generateWeight: xavier.generate
    },
    he: {
        id: 'he',
        name: 'He Initialization (Uniforme)',
        description: 'Escala los pesos considerando únicamente las neuronas de entrada. Diseñada específicamente para evitar que la función ReLU muera por falta de señal.',
        generateWeight: he.generate
    }
};

export function getInitializationMethod(id) {
    return initializationRegistry[id] || initializationRegistry.random;
}