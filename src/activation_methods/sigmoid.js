import { sigmoid, sigmoidDerivative } from '../math/activations.js';

export const sigmoidLogic = {
    feed: sigmoid,
    derivative: sigmoidDerivative,
};