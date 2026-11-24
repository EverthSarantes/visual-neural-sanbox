import { tanh, tanhDerivative } from '../math/activations.js';

export const tanhLogic = {
    feed: tanh,
    derivative: tanhDerivative,
};