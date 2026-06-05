export const bceLogic = {
    /**
     * Calcula la pérdida de entropía cruzada binaria
     */
    calculate: (actual, target) => {
        const epsilon = 1e-15;
        // Comprimir el valor actual entre [epsilon] y [1 - epsilon] para proteger los logaritmos
        const a = Math.max(epsilon, Math.min(1 - epsilon, actual));
        
        return -(target * Math.log(a) + (1 - target) * Math.log(1 - a));
    },

    /**
     * Retorna el término de la señal de error corregido para el gradiente (Target - Actual)
     * Derivada de BCE respecto a 'actual', invertida para mantener la convención del optimizador
     */
    errorSignalTerm: (actual, target) => {
        const epsilon = 1e-15;
        const a = Math.max(epsilon, Math.min(1 - epsilon, actual));
        
        // Fórmula derivada: (y - a) / (a * (1 - a))
        return (target - a) / (a * (1 - a));
    }
};