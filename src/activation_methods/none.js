import { identity, identityDerivative } from '../math/activations.js';

export const noneLogic = {
    feed: identity,
    derivative: identityDerivative,
};