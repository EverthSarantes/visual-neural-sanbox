import { ImageLoader } from '../utils/ImageLoader.js';

export class ImageController {
    constructor(uiController) {
        this.ui = uiController;
        this.loader = new ImageLoader();
    }

    get dom() { return this.ui.dom; }
    get network() { return this.ui.network; }

    async handleImageUpload(fileListOrFile) {
        try {
            let summary;
            if (fileListOrFile instanceof File && /\.zip$/i.test(fileListOrFile.name)) {
                summary = await this.loader.loadFromZip(fileListOrFile);
            } else {
                summary = await this.loader.loadFromFiles(fileListOrFile);
            }
            this._renderSummary(summary);
            this._renderThumbnails();
        } catch (error) {
            alert(error.message);
        }
    }

    _renderSummary(summary) {
        const container = document.getElementById('image-summary');
        if (!container) return;
        container.innerHTML = `
            <div class="text-xs text-slate-300 space-y-1">
                <div>Clases: <span class="text-indigo-400 font-bold">${summary.classes.join(', ')}</span></div>
                <div>Total imágenes: <span class="text-indigo-400 font-bold">${summary.totalImages}</span></div>
                ${summary.classes.map(c => `<div class="text-slate-400">${c}: ${summary.counts[c]}</div>`).join('')}
            </div>
        `;
    }

    _renderThumbnails() {
        const container = document.getElementById('image-thumbnails');
        if (!container) return;
        container.innerHTML = '';
        const thumbs = this.loader.getThumbnails(4);
        for (const cls of this.loader.classes) {
            const group = document.createElement('div');
            group.className = 'space-y-1';
            const label = document.createElement('div');
            label.className = 'text-[10px] text-slate-400 font-medium';
            label.innerText = cls;
            group.appendChild(label);
            const row = document.createElement('div');
            row.className = 'flex gap-1 flex-wrap';
            for (const url of thumbs[cls] || []) {
                const img = document.createElement('img');
                img.src = url;
                img.className = 'w-10 h-10 rounded border border-slate-700 object-cover';
                row.appendChild(img);
            }
            group.appendChild(row);
            container.appendChild(group);
        }
    }

    compileImageArchitecture() {
        if (this.loader.images.length === 0) {
            alert('Primero carga imágenes.');
            return;
        }
        const imageSize = this.loader.imageSize;
        const channels = this.loader.channels;
        const inputCount = imageSize * imageSize * channels;
        const outputCount = this.loader.classes.length;
        const hiddenTopology = [16, 8];

        this.network.targetMetadata = [];
        this.network.classWeights = new Array(outputCount).fill(1.0);
        this.network.useClassWeighting = false;
        this.network.inputShape = [channels, imageSize, imageSize];

        this.network.buildArchitecture(inputCount, hiddenTopology, outputCount);
        this.ui.renderer.render(this.network);
        this.ui.closeInspector();

        const { trainSet, testSet } = this.loader.compileDataset(0.8, 2);
        this.ui.trainSet = trainSet;
        this.ui.testSet = testSet;

        this.ui.trainingClient.init({
            inputCount,
            hiddenTopology,
            outputCount,
            outputActivation: 'softmax',
            lossTypeId: 'cce',
            learningRate: 0.01,
            momentum: 0.9,
            weightDecay: 0.0001,
            dropoutRate: 0.0,
            batchSize: 16,
            initializationStrategyId: 'he',
            epoch: 0,
            currentLoss: 0,
            currentAccuracy: 0,
            currentMAE: null,
            targetMetadata: [],
            classWeights: this.network.classWeights,
            useClassWeighting: false,
            trainSet,
            testSet,
        });

        this.ui.dom.selectLossType.value = 'cce';
        this.ui.dom.selectLossType.dispatchEvent(new Event('change', { bubbles: true }));

        this.ui.syncMetrics();
        this.ui.updatePlaybackButtonsUI();
    }
}