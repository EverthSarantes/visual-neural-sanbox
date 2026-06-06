export const sigmoidLogic = {
    // Aplica la fórmula: 1 / (1 + e^-x)
    feed: (x) => {
        const s = 1 / (1 + Math.exp(-x));
        return Math.max(Math.min(s, 1 - 1e-7), 1e-7);
    },
    // La derivada es: sigmoid(x) * (1 - sigmoid(x))
    derivative: (x) => {
        const s = 1 / (1 + Math.exp(-x));
        const p = Math.max(Math.min(s, 1 - 1e-7), 1e-7);
        return p * (1 - p);
    }
};