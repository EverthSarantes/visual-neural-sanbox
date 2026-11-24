/**
 * Módulo matemático puro para inicialización de pesos.
 */

export function randomWeight(_nIn, _nOut) {
    return Math.random() * 2 - 1;
}

export function xavierWeight(nIn, nOut) {
    const fanIn = nIn > 0 ? nIn : 1;
    const fanOut = nOut > 0 ? nOut : 1;
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    return Math.random() * (limit * 2) - limit;
}

export function heWeight(nIn, _nOut) {
    const fanIn = nIn > 0 ? nIn : 1;
    const limit = Math.sqrt(6 / fanIn);
    return Math.random() * (limit * 2) - limit;
}