export const bceLogic = {
    multiplier: -1,
    /**
     * Calcula la pérdida de entropía cruzada binaria
     */
    calculate: (actual, target) => {
        const p = Math.max(Math.min(actual, 1 - 1e-7), 1e-7);
        return -(target * Math.log(p) + (1 - target) * Math.log(1 - p));
    },

    /**
     * Retorna el término de la señal de error corregido para el gradiente (Target - Actual)
     * Derivada de BCE respecto a 'actual', invertida para mantener la convención del optimizador
     */
    errorSignalTerm: (actual, target) => {
        const p = Math.max(Math.min(actual, 1 - 1e-7), 1e-7);
        return (p - target) / (p * (1 - p));
    }
};