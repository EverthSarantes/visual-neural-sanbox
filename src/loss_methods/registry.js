import { mseLogic } from './mse.js';
import { bceLogic } from './bce.js';

export const lossRegistry = {
    mse: {
        id: 'mse',
        name: 'Error Cuadrático Medio (MSE)',
        description: 'Penaliza con mayor fuerza los errores grandes al elevar las diferencias al cuadrado. Es la métrica estándar para problemas de regresión.',
        calculate: mseLogic.calculate,
        errorSignalTerm: mseLogic.errorSignalTerm
    },
    bce: {
        id: 'bce',
        name: 'Entropía Cruzada Binaria (BCE)',
        description: 'Mide la divergencia entre dos distribuciones de probabilidad. Es el estándar de oro para problemas de clasificación binaria.',
        calculate: bceLogic.calculate,
        errorSignalTerm: bceLogic.errorSignalTerm
    }
};

/**
 * Busca y retorna el objeto de pérdida completo por su ID.
 * @param {string} id 
 * @returns {Object|null}
 */
export function getLossMethod(id) {
    return lossRegistry[id] || null;
}