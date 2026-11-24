import { relu, reluDerivative } from '../math/activations.js';

export const reluLogic = {
    feed: relu,
    derivative: reluDerivative,
};