export const xavier = {
    generate: (nIn, nOut) => {
        const limit = Math.sqrt(6 / (nIn + nOut));
        return Math.random() * (limit * 2) - limit;
    }
} 