export const reluLogic = {
    // Si x es mayor a 0 devuelve x, de lo contrario devuelve 0
    feed: (x) => {
        return x > 0 ? x : 0;
    },
    // La derivada de ReLU es 1 si x > 0, de lo contrario es 0
    derivative: (x) => {
        return x > 0 ? 1 : 0;
    }
};