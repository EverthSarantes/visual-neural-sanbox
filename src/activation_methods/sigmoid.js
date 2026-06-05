export const sigmoidLogic = {
    // Aplica la fórmula: 1 / (1 + e^-x)
    feed: (x) => {
        return 1 / (1 + Math.exp(-x));
    },
    // La derivada es: sigmoid(x) * (1 - sigmoid(x))
    derivative: (x) => {
        const s = 1 / (1 + Math.exp(-x));
        return s * (1 - s);
    }
};