export const cceLogic = {
    /**
     * Fórmula: L = - Suma( target * log(actual) )
     */
    calculate: (actualVector, targetVector) => {
        let sum = 0;
        for (let i = 0; i < actualVector.length; i++) {
            const val = Math.max(actualVector[i], 1e-7);
            sum += (targetVector[i] || 0) * Math.log(val);
        }
        return -sum;
    },

    /**
     * La derivada de la pérdida CCE respecto a la activación es: -target / actual
     */
    errorSignalTerm: (actual, target) => {
        const denominator = Math.max(actual, 1e-7);
        return -target / denominator;
    }
};