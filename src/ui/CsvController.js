/**
 * Controlador de carga y mapeo de archivos CSV.
 * Encapsula la lógica de diagnóstico, renderizado de selectores y compilación
 * de la arquitectura a partir del CSV.
 */
export class CsvController {
    constructor(uiController) {
        this.ui = uiController;
    }

    get dom() { return this.ui.dom; }
    get network() { return this.ui.network; }

    async handleCsvUpload(file) {
        try {
            this.dom.fileNameLabel.innerText = file.name;
            this.ui.datasetDiagnostics = await this.ui.csvParser.scanFile(file);

            this.dom.dropzone.classList.add('hidden');
            this.dom.mappingArea.classList.remove('hidden');

            this.renderColumnMappers();
        } catch (error) {
            alert(error.message);
        }
    }

    renderColumnMappers() {
        this.dom.columnsListContainer.innerHTML = '';

        Object.values(this.ui.datasetDiagnostics).forEach(col => {
            const row = document.createElement('div');
            row.className = 'bg-slate-900/40 border border-slate-800/80 p-2.5 rounded-lg text-xs space-y-2';

            const headerInfo = document.createElement('div');
            headerInfo.className = 'flex justify-between items-center text-[11px] font-medium text-slate-300';

            let metaLabel = '';
            if (col.isFullyNumeric) {
                metaLabel = `[${col.min} a ${col.max}]`;
            } else if (col.uniqueValues.length === 2) {
                metaLabel = '[Binario]';
            } else {
                metaLabel = `[${col.uniqueValues.length} cat]`;
            }

            headerInfo.innerHTML = `<span class="font-semibold truncate max-w-[140px]" title="${col.name}">${col.name}</span> <span class="font-mono text-slate-500">${metaLabel}</span>`;
            row.appendChild(headerInfo);

            const controlsGrid = document.createElement('div');
            controlsGrid.className = 'grid grid-cols-2 gap-2';

            const selectRole = document.createElement('select');
            selectRole.className = 'w-full bg-slate-950 border border-slate-800 rounded p-1 text-[11px] text-slate-300 focus:outline-none cursor-pointer';
            selectRole.innerHTML = `
                <option value="input">Entrada</option>
                <option value="output">Salida</option>
                <option value="ignore">Ignorar</option>
            `;

            col.role = col.role || 'input';
            selectRole.value = col.role;
            controlsGrid.appendChild(selectRole);

            const contextualContainer = document.createElement('div');

            if (col.isFullyNumeric) {
                const selectNorm = document.createElement('select');
                selectNorm.className = 'w-full bg-slate-950 border border-slate-800 rounded p-1 text-[11px] text-slate-300 focus:outline-none';
                selectNorm.innerHTML = `
                    <option value="minmax_0_1">Norm [0, 1]</option>
                    <option value="minmax_1_1">Norm [-1, 1]</option>
                    <option value="none">Crudo (None)</option>
                `;

                col.normalization = col.normalization || 'minmax_0_1';
                selectNorm.value = col.normalization;

                selectNorm.addEventListener('change', (e) => {
                    col.normalization = e.target.value;
                });

                contextualContainer.appendChild(selectNorm);
            } else if (col.uniqueValues.length === 2) {
                const binaryGrid = document.createElement('div');
                binaryGrid.className = 'grid grid-cols-2 gap-1 text-[10px] text-slate-400 items-center';
                binaryGrid.innerHTML = `
                    <div class="flex items-center space-x-0.5"><span>0:</span><span class="text-indigo-400 font-bold truncate max-w-[40px]" title="${col.uniqueValues[0]}">${col.uniqueValues[0]}</span></div>
                    <div class="flex items-center space-x-0.5"><span>1:</span><span class="text-emerald-400 font-bold truncate max-w-[40px]" title="${col.uniqueValues[1]}">${col.uniqueValues[1]}</span></div>
                `;
                contextualContainer.appendChild(binaryGrid);
            } else {
                const badge = document.createElement('div');
                badge.className = 'text-[10px] text-center bg-indigo-950/40 border border-indigo-900/60 text-indigo-300 rounded py-0.5 font-medium';
                badge.innerText = 'One-Hot Encod';
                contextualContainer.appendChild(badge);
            }

            controlsGrid.appendChild(contextualContainer);
            row.appendChild(controlsGrid);

            const weightingContainer = document.createElement('div');
            weightingContainer.className = 'mt-1 pt-1.5 border-t border-slate-800/40 flex items-center hidden';

            const checkWeight = document.createElement('input');
            checkWeight.type = 'checkbox';
            checkWeight.className = 'rounded bg-slate-950 border-slate-800 text-indigo-500 focus:ring-0 w-3 h-3 cursor-pointer mr-1.5';
            col.useClassWeighting = col.useClassWeighting || false;
            checkWeight.checked = col.useClassWeighting;

            checkWeight.addEventListener('change', (e) => {
                col.useClassWeighting = e.target.checked;
            });

            let distributionStats = '';
            if (!col.isFullyNumeric && col.frequencies) {
                distributionStats = Object.entries(col.frequencies)
                    .map(([val, percent]) => `${val}:${(percent * 100).toFixed(0)}%`)
                    .join(', ');
            }

            const labelWeight = document.createElement('label');
            labelWeight.className = 'flex items-center text-[10px] text-slate-400 cursor-pointer select-none w-full';
            labelWeight.appendChild(checkWeight);

            const textSpan = document.createElement('span');
            textSpan.innerHTML = `Ponderar desbalance <span class="text-slate-500 font-mono text-[9px] ml-0.5">(${distributionStats})</span>`;
            labelWeight.appendChild(textSpan);

            weightingContainer.appendChild(labelWeight);
            row.appendChild(weightingContainer);

            const syncWeightingVisibility = (currentRole) => {
                if (currentRole === 'output' && !col.isFullyNumeric) {
                    weightingContainer.classList.remove('hidden');
                } else {
                    weightingContainer.classList.add('hidden');
                    col.useClassWeighting = false;
                    checkWeight.checked = false;
                }
            };

            syncWeightingVisibility(col.role);

            selectRole.addEventListener('change', (e) => {
                col.role = e.target.value;
                syncWeightingVisibility(col.role);
            });

            this.dom.columnsListContainer.appendChild(row);
        });
    }

    compileSelectedArchitecture() {
        let inputCount = 0;
        let outputCount = 0;
        const classWeights = [];
        this.network.targetMetadata = [];

        Object.values(this.ui.datasetDiagnostics).forEach(col => {
            const role = col.role;
            if (role === 'ignore') return;

            let weightIncrement = 1;
            if (!col.isFullyNumeric && col.uniqueValues.length > 2) {
                weightIncrement = col.uniqueValues.length;
            }

            if (role === 'input') inputCount += weightIncrement;

            if (role === 'output') {
                if (col.isFullyNumeric) {
                    this.network.targetMetadata.push({
                        name: col.name,
                        min: col.min,
                        max: col.max,
                        normalization: col.normalization || 'none'
                    });
                }

                if (col.useClassWeighting && !col.isFullyNumeric) {
                    const numClasses = col.uniqueValues.length;

                    if (numClasses === 2) {
                        const freq0 = col.frequencies[col.uniqueValues[0]] || 0.5;
                        const freq1 = col.frequencies[col.uniqueValues[1]] || 0.5;
                        const w0 = 1 / (2 * freq0);
                        const w1 = 1 / (2 * freq1);
                        classWeights.push((w0 + w1) / 2);
                    } else {
                        col.uniqueValues.forEach(val => {
                            const freq = col.frequencies[val] || 1e-7;
                            const inverseWeight = 1 / (numClasses * freq);
                            classWeights.push(inverseWeight);
                        });
                    }
                } else {
                    for (let k = 0; k < weightIncrement; k++) {
                        classWeights.push(1.0);
                    }
                }
                outputCount += weightIncrement;
            }
        });

        if (inputCount === 0 || outputCount === 0) {
            alert('Configuración inválida: Debes asignar al menos una columna de Entrada y una de Salida.');
            return;
        }
        this.network.classWeights = classWeights;
        this.network.useClassWeighting = Object.values(this.ui.datasetDiagnostics).some(
            col => col.role === 'output' && !col.isFullyNumeric && col.useClassWeighting === true
        );

        this.network.buildArchitecture(inputCount, [4], outputCount);
        this.ui.renderer.render(this.network);
        this.ui.closeInspector();

        const fullDataset = this.ui.csvParser.compileDataset(this.ui.datasetDiagnostics);

        for (let i = fullDataset.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [fullDataset[i], fullDataset[j]] = [fullDataset[j], fullDataset[i]];
        }

        const trainPercent = parseInt(this.dom.sliderSplit.value, 10) / 100;
        const cutoff = Math.floor(fullDataset.length * trainPercent);

        this.ui.trainSet = fullDataset.slice(0, cutoff);
        this.ui.testSet = fullDataset.slice(cutoff);

        this.ui.engine.setDatasets(this.ui.trainSet, this.ui.testSet);

        this.ui.syncMetrics();
        this.ui.updatePlaybackButtonsUI();
    }
}