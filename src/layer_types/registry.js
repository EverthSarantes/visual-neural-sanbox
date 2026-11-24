/**
 * Registro de tipos de capa, análogo a activation/loss registries.
 * Cada tipo define metadatos y funciones matemáticas (forward/backward/update).
 * Para añadir un nuevo tipo de capa, crear un módulo en src/layer_types/ y registrarlo aquí.
 */
import * as L from '../math/layers.js';

export const layerTypeRegistry = {
    dense: {
        id: 'dense',
        name: 'Densa (Fully Connected)',
        description: 'Capa tradicional: cada neurona se conecta con todas las de la capa anterior.',
        hasWeights: true,
        forward: L.denseForward,
        backward: L.denseBackward,
    },
    conv2d: {
        id: 'conv2d',
        name: 'Convolución 2D',
        description: 'Aplica filtros deslizantes sobre la entrada. Ideal para imágenes.',
        hasWeights: true,
        forward: L.conv2dForward,
        backward: L.conv2dBackward,
    },
    maxpool2d: {
        id: 'maxpool2d',
        name: 'Max Pooling 2D',
        description: 'Reduce la resolución tomando el máximo de cada bloque.',
        hasWeights: false,
        forward: L.maxpoolForward,
        backward: L.maxpoolBackward,
    },
    flatten: {
        id: 'flatten',
        name: 'Flatten',
        description: 'Aplana el tensor multidimensional a un vector.',
        hasWeights: false,
        forward: L.flattenForward,
        backward: L.flattenBackward,
    },
};

export function getLayerType(id) {
    return layerTypeRegistry[id] || layerTypeRegistry.dense;
}