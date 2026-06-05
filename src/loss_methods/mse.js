export const mseLogic = {
    // Calcula el error para una sola neurona de salida: 0.5 * (prediccion - real)^2
    calculate: (actual, target) => {
        return 0.5 * Math.pow(actual - target, 2);
    },
    // Devuelve la dirección del cambio (Target - Actual) necesaria para ajustar los pesos hacia atrás
    errorSignalTerm: (actual, target) => {
        return target - actual;
    }
};