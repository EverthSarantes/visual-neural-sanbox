export const sigmoidLogic = {
    // Aplica la fórmula: 1 / (1 + e^-x)
    feed: (x) => {
        const safeX = Math.max(Math.min(x, 50), -50);
        const s = 1 / (1 + Math.exp(-safeX));
        return Math.max(Math.min(s, 1 - 1e-7), 1e-7);
    },
    // La derivada es: sigmoid(x) * (1 - sigmoid(x))
    derivative: (x) => {
        const safeX = Math.max(Math.min(x, 50), -50);
        const s = 1 / (1 + Math.exp(-safeX));
        const p = Math.max(Math.min(s, 1 - 1e-7), 1e-7);
        return p * (1 - p);
    }
};