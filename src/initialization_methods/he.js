export const he = {
    generate: (nIn, nOut) => {
        const limit = Math.sqrt(6 / nIn);
        return Math.random() * (limit * 2) - limit;
    }
}