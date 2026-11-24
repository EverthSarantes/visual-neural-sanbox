/**
 * Funciones matemáticas puras para capas de red neuronal.
 * Cada función recibe tensores/arrays y parámetros, devuelve resultados.
 * No tiene efectos secundarios.
 */

// ---------- Dense ----------
export function denseForward(input, weights, bias, inFeatures, outFeatures) {
    const n = input.length / inFeatures;
    const out = new Float32Array(n * outFeatures);
    for (let b = 0; b < n; b++) {
        for (let o = 0; o < outFeatures; o++) {
            let sum = bias[o];
            for (let i = 0; i < inFeatures; i++) {
                sum += input[b * inFeatures + i] * weights[i * outFeatures + o];
            }
            out[b * outFeatures + o] = sum;
        }
    }
    return out;
}

export function denseBackward(gradOutput, input, weights, inFeatures, outFeatures) {
    const n = gradOutput.length / outFeatures;
    const gradInput = new Float32Array(n * inFeatures);
    const gradW = new Float32Array(inFeatures * outFeatures);
    const gradB = new Float32Array(outFeatures);
    for (let b = 0; b < n; b++) {
        for (let o = 0; o < outFeatures; o++) {
            const g = gradOutput[b * outFeatures + o];
            gradB[o] += g;
            for (let i = 0; i < inFeatures; i++) {
                gradW[i * outFeatures + o] += input[b * inFeatures + i] * g;
                gradInput[b * inFeatures + i] += weights[i * outFeatures + o] * g;
            }
        }
    }
    return { gradInput, gradW, gradB };
}

// ---------- Conv2D ----------
export function conv2dForward(input, weights, bias, inC, inH, inW, outC, kH, kW, stride, padding) {
    const Hout = Math.floor((inH + 2 * padding - kH) / stride) + 1;
    const Wout = Math.floor((inW + 2 * padding - kW) / stride) + 1;
    const N = input.length / (inC * inH * inW);
    const out = new Float32Array(N * outC * Hout * Wout);
    for (let n = 0; n < N; n++) {
        for (let co = 0; co < outC; co++) {
            for (let oh = 0; oh < Hout; oh++) {
                for (let ow = 0; ow < Wout; ow++) {
                    let sum = bias[co];
                    for (let ci = 0; ci < inC; ci++) {
                        for (let kh = 0; kh < kH; kh++) {
                            for (let kw = 0; kw < kW; kw++) {
                                const ih = oh * stride - padding + kh;
                                const iw = ow * stride - padding + kw;
                                if (ih < 0 || ih >= inH || iw < 0 || iw >= inW) continue;
                                const inIdx = ((n * inC + ci) * inH + ih) * inW + iw;
                                const wIdx = ((co * inC + ci) * kH + kh) * kW + kw;
                                sum += input[inIdx] * weights[wIdx];
                            }
                        }
                    }
                    const outIdx = ((n * outC + co) * Hout + oh) * Wout + ow;
                    out[outIdx] = sum > 0 ? sum : 0; // ReLU
                }
            }
        }
    }
    return { data: out, shape: [N, outC, Hout, Wout] };
}

export function conv2dBackward(gradOutput, input, weights, inC, inH, inW, outC, kH, kW, stride, padding) {
    const Hout = Math.floor((inH + 2 * padding - kH) / stride) + 1;
    const Wout = Math.floor((inW + 2 * padding - kW) / stride) + 1;
    const N = input.length / (inC * inH * inW);
    const gradInput = new Float32Array(N * inC * inH * inW);
    const gradW = new Float32Array(outC * inC * kH * kW);
    const gradB = new Float32Array(outC);

    for (let n = 0; n < N; n++) {
        for (let co = 0; co < outC; co++) {
            for (let oh = 0; oh < Hout; oh++) {
                for (let ow = 0; ow < Wout; ow++) {
                    const outIdx = ((n * outC + co) * Hout + oh) * Wout + ow;
                    // ReLU deriv
                    const preAct = gradOutput[outIdx]; // gradOutput already gated? We gate here by checking input? Use stored pre-act not available; gate by gradOutput sign not correct. 
                    // We'll rely on caller to gate gradOutput. Here just propagate.
                    const g = gradOutput[outIdx];
                    gradB[co] += g;
                    for (let ci = 0; ci < inC; ci++) {
                        for (let kh = 0; kh < kH; kh++) {
                            for (let kw = 0; kw < kW; kw++) {
                                const ih = oh * stride - padding + kh;
                                const iw = ow * stride - padding + kw;
                                if (ih < 0 || ih >= inH || iw < 0 || iw >= inW) continue;
                                const inIdx = ((n * inC + ci) * inH + ih) * inW + iw;
                                const wIdx = ((co * inC + ci) * kH + kh) * kW + kw;
                                gradW[wIdx] += input[inIdx] * g;
                                gradInput[inIdx] += weights[wIdx] * g;
                            }
                        }
                    }
                }
            }
        }
    }
    return { gradInput, gradW, gradB };
}

// ---------- MaxPool2D ----------
export function maxpoolForward(input, inC, inH, inW, poolSize, stride) {
    const Hout = Math.floor((inH - poolSize) / stride) + 1;
    const Wout = Math.floor((inW - poolSize) / stride) + 1;
    const N = input.length / (inC * inH * inW);
    const out = new Float32Array(N * inC * Hout * Wout);
    const indices = new Int32Array(out.length);
    for (let n = 0; n < N; n++) {
        for (let c = 0; c < inC; c++) {
            for (let oh = 0; oh < Hout; oh++) {
                for (let ow = 0; ow < Wout; ow++) {
                    let maxVal = -Infinity;
                    let maxIdx = -1;
                    for (let ph = 0; ph < poolSize; ph++) {
                        for (let pw = 0; pw < poolSize; pw++) {
                            const ih = oh * stride + ph;
                            const iw = ow * stride + pw;
                            const idx = ((n * inC + c) * inH + ih) * inW + iw;
                            if (input[idx] > maxVal) { maxVal = input[idx]; maxIdx = idx; }
                        }
                    }
                    const outIdx = ((n * inC + c) * Hout + oh) * Wout + ow;
                    out[outIdx] = maxVal;
                    indices[outIdx] = maxIdx;
                }
            }
        }
    }
    return { data: out, shape: [N, inC, Hout, Wout], indices };
}

export function maxpoolBackward(gradOutput, indices, totalSize) {
    const gradInput = new Float32Array(totalSize);
    for (let i = 0; i < gradOutput.length; i++) {
        const idx = indices[i];
        if (idx >= 0) gradInput[idx] += gradOutput[i];
    }
    return gradInput;
}

// ---------- Flatten ----------
export function flattenForward(input) {
    return new Float32Array(input); // identity, just a view copy
}

export function flattenBackward(gradOutput, originalShape) {
    return new Float32Array(gradOutput); // identity
}