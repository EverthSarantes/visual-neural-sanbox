export class ImageLoader {
    constructor() {
        this.classes = [];
        this.images = [];
        this.imageSize = 32;
        this.channels = 3; // 1 (grayscale) o 3 (RGB)
    }

    async loadFromFiles(fileList) {
        this.classes = [];
        this.images = [];
        const classMap = new Map();
        const files = Array.from(fileList).filter(f => /\.(png|jpg|jpeg|gif|bmp|webp)$/i.test(f.name));

        for (const file of files) {
            const rel = file.webkitRelativePath || file.name;
            const parts = rel.split('/');
            let className = 'default';
            if (parts.length > 1) className = parts[parts.length - 2];
            if (!classMap.has(className)) classMap.set(className, classMap.size);
        }
        this.classes = Array.from(classMap.keys());

        for (const file of files) {
            const rel = file.webkitRelativePath || file.name;
            const parts = rel.split('/');
            let className = 'default';
            if (parts.length > 1) className = parts[parts.length - 2];
            const classIndex = classMap.get(className);
            const imgData = await this._readImage(file, this.imageSize);
            this.images.push({ className, classIndex, data: imgData.data, width: this.imageSize, height: this.imageSize, channels: this.channels });
        }
        return this._summary();
    }

    async loadFromZip(zipFile) {
        this.classes = [];
        this.images = [];
        const classMap = new Map();
        const zip = await JSZip.loadAsync(zipFile);
        const entries = Object.keys(zip.files).filter(p => !zip.files[p].dir && /\.(png|jpg|jpeg|gif|bmp|webp)$/i.test(p));
        for (const path of entries) {
            const parts = path.split('/');
            let className = 'default';
            if (parts.length > 1) className = parts[parts.length - 2];
            if (!classMap.has(className)) classMap.set(className, classMap.size);
        }
        this.classes = Array.from(classMap.keys());
        for (const path of entries) {
            const parts = path.split('/');
            let className = 'default';
            if (parts.length > 1) className = parts[parts.length - 2];
            const classIndex = classMap.get(className);
            const blob = await zip.files[path].async('blob');
            const file = new File([blob], path.split('/').pop(), { type: blob.type });
            const imgData = await this._readImage(file, this.imageSize);
            this.images.push({ className, classIndex, data: imgData.data, width: this.imageSize, height: this.imageSize, channels: this.channels });
        }
        return this._summary();
    }

    async _readImage(file, size) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            const url = URL.createObjectURL(file);
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = size; canvas.height = size;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, size, size);
                const imageData = ctx.getImageData(0, 0, size, size);
                const pixels = imageData.data;
                const ch = this.channels;
                const out = new Float32Array(size * size * ch);
                for (let i = 0; i < size * size; i++) {
                    const r = pixels[i * 4] / 255;
                    const g = pixels[i * 4 + 1] / 255;
                    const b = pixels[i * 4 + 2] / 255;
                    if (ch === 1) {
                        // Grayscale: luminancia estándar
                        out[i] = 0.299 * r + 0.587 * g + 0.114 * b;
                    } else {
                        out[i * 3] = r;
                        out[i * 3 + 1] = g;
                        out[i * 3 + 2] = b;
                    }
                }
                URL.revokeObjectURL(url);
                resolve({ data: out });
            };
            img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('No se pudo leer la imagen: ' + file.name)); };
            img.src = url;
        });
    }

    _summary() {
        const counts = {};
        this.classes.forEach(c => counts[c] = 0);
        this.images.forEach(img => counts[img.className]++);
        return { classes: this.classes, totalImages: this.images.length, counts, imageSize: this.imageSize };
    }

    compileDataset(trainPercent = 0.8, augmentFactor = 1) {
        const shuffled = [...this.images];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        const cutoff = Math.floor(shuffled.length * trainPercent);
        const trainImages = shuffled.slice(0, cutoff);
        const testImages = shuffled.slice(cutoff);

        const toSample = (img, augment) => {
            let data = img.data;
            if (augment) data = ImageLoader.augment(data, img.width, img.height, img.channels);
            const target = new Array(this.classes.length).fill(0);
            target[img.classIndex] = 1;
            return { inputs: Array.from(data), targets: target };
        };

        const trainSet = [];
        for (const img of trainImages) {
            trainSet.push(toSample(img, false));
            for (let a = 0; a < augmentFactor; a++) trainSet.push(toSample(img, true));
        }
        const testSet = testImages.map(img => toSample(img, false));
        return { trainSet, testSet, classes: this.classes };
    }

    static augment(data, width, height, channels) {
        const out = new Float32Array(data.length);
        const flip = Math.random() < 0.5;
        const brightness = (Math.random() - 0.5) * 0.2;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const srcX = flip ? (width - 1 - x) : x;
                for (let c = 0; c < channels; c++) {
                    const idx = (y * width + x) * channels + c;
                    const srcIdx = (y * width + srcX) * channels + c;
                    let val = data[srcIdx] + brightness;
                    val = Math.max(0, Math.min(1, val));
                    out[idx] = val;
                }
            }
        }
        return out;
    }

    getThumbnails(maxPerClass = 4) {
        const result = {};
        this.classes.forEach(c => result[c] = []);
        for (const img of this.images) {
            if (result[img.className].length < maxPerClass) {
                result[img.className].push(this._toDataURL(img.data, img.width, img.height));
            }
        }
        return result;
    }

    _toDataURL(data, width, height) {
        const canvas = document.createElement('canvas');
        canvas.width = width; canvas.height = height;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(width, height);
        const ch = this.channels;
        for (let i = 0; i < width * height; i++) {
            if (ch === 1) {
                const v = Math.round(data[i] * 255);
                imageData.data[i * 4] = v;
                imageData.data[i * 4 + 1] = v;
                imageData.data[i * 4 + 2] = v;
            } else {
                imageData.data[i * 4] = Math.round(data[i * 3] * 255);
                imageData.data[i * 4 + 1] = Math.round(data[i * 3 + 1] * 255);
                imageData.data[i * 4 + 2] = Math.round(data[i * 3 + 2] * 255);
            }
            imageData.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(imageData, 0, 0);
        return canvas.toDataURL();
    }
}