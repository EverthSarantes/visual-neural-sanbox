import {
    softmaxCalculateLayer,
    softmaxOutputDelta,
    softmaxHiddenDelta,
} from '../math/activations.js';

// Adaptador para mantener la interfaz { feed, derivative, calculateLayer, ... }
// que esperan el registry y la propagación.
export const softmaxLogic = {
    feed: (_x) => 1,
    derivative: (_x) => 1,

    calculateLayer: (layer, network) => softmaxCalculateLayer(layer, network),

    // derivativeLayer recibe (layer, targetVector, lossMethod). Usamos el término
    // de la pérdida almacenado en lossMethod para mantener el contrato.
    derivativeLayer: (layer, targetVector, lossMethod) =>
        softmaxOutputDelta(layer, targetVector, lossMethod.errorSignalTerm),

    derivativeLayerHidden: (layer, errorSums) => softmaxHiddenDelta(layer, errorSums),
};