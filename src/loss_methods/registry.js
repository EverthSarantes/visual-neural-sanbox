import { mseLogic } from './mse.js';
import { bceLogic } from './bce.js';
import { cceLogic } from './cce.js';

export const lossRegistry = {
    mse: {
        id: 'mse',
        name: 'Error Cuadrático Medio (MSE)',
        description: 'Penaliza con mayor fuerza los errores grandes al elevar las diferencias al cuadrado. Es la métrica estándar para problemas de regresión.',
        multiplier: mseLogic.multiplier,
        calculate: mseLogic.calculate,
        errorSignalTerm: mseLogic.errorSignalTerm
    },
    bce: {
        id: 'bce',
        name: 'Entropía Cruzada Binaria (BCE)',
        description: 'Mide la divergencia entre dos distribuciones de probabilidad. Es el estándar de oro para problemas de clasificación binaria.',
        multiplier: bceLogic.multiplier,
        calculate: bceLogic.calculate,
        errorSignalTerm: bceLogic.errorSignalTerm
    },
    cce: {
        id: 'cce',
        name: 'Entropía Cruzada Categórica (CCE)',
        description: 'La pareja perfecta de Softmax. Mide la discrepancia entre dos distribuciones de probabilidad excluyentes, penalizando drásticamente los errores confiables.',
        multiplier: cceLogic.multiplier,
        calculate: cceLogic.calculate,
        errorSignalTerm: cceLogic.errorSignalTerm
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