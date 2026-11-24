/**
 * Módulo matemático puro para funciones de pérdida.
 * Se asume la convención de descenso: el "término de error" devuelto es la
 * contribución -dL/dz (dirección de descenso) cuando se combina con la
 * derivada de la activación, o el término que usa la capa Softmax.
 */

// ---------- MSE ----------
export function mseCalculate(actual, target) {
    return 0.5 * Math.pow(actual - target, 2);
}
export function mseErrorSignalTerm(actual, target) {
    return actual - target;
}

// ---------- BCE ----------
export function bceCalculate(actual, target) {
    const p = Math.max(Math.min(actual, 1 - 1e-7), 1e-7);
    return -(target * Math.log(p) + (1 - target) * Math.log(1 - p));
}
export function bceErrorSignalTerm(actual, target) {
    const p = Math.max(Math.min(actual, 1 - 1e-7), 1e-7);
    // Verdadero gradiente dL/dp = (p - target) / (p * (1 - p))
    return (p - target) / (p * (1 - p));
}

// ---------- CCE (término por neurona; se suma sobre la capa) ----------
export function cceCalculate(actual, target) {
    const val = Math.max(actual, 1e-7);
    return -(target || 0) * Math.log(val);
}
export function cceErrorSignalTerm(actual, target) {
    const denominator = Math.max(actual, 1e-7);
    return -target / denominator;
}