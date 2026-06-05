export const tanhLogic = {
    // Utiliza la función nativa de JavaScript para la tangente hiperbólica
    feed: (x) => {
        return Math.tanh(x);
    },
    // La derivada de tanh(x) es: 1 - tanh^2(x)
    derivative: (x) => {
        const t = Math.tanh(x);
        return 1 - (t * t);
    }
};