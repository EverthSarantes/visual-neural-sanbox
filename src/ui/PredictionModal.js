import { forwardPass } from '../propagation/forward.js';

export class PredictionModal {
    constructor(network, uiController) {
        this.network = network;
        this.ui = uiController;

        this.dom = {
            modal: document.getElementById('modal-prediction'),
            btnClose: document.getElementById('btn-close-modal'),
            btnPredict: document.getElementById('btn-modal-predict'),
            toggleNormalize: document.getElementById('modal-toggle-normalize'),
            inputsContainer: document.getElementById('modal-inputs-container'),
            outputsContainer: document.getElementById('modal-outputs-container')
        };

        this.initListeners();
    }

    initListeners() {
        if (!this.dom.modal) return;
        this.dom.btnClose.addEventListener('click', () => this.close());
        this.dom.btnPredict.addEventListener('click', () => this.executeInference());
    }

    open() {
        if (!this.ui.datasetDiagnostics) {
            alert('No hay configuraciones de datos activas. Carga y mapea un CSV primero.');
            return;
        }
        this.dom.modal.classList.remove('hidden');
        this.buildDynamicForm();
        this.clearResults();
    }

    close() {
        this.dom.modal.classList.add('hidden');
    }

    clearResults() {
        this.dom.outputsContainer.innerHTML = '<p class="text-xs text-slate-500 italic text-center py-2">Ingresa datos y presiona calcular...</p>';
    }

    /**
     * Construye el formulario leyendo directamente los elementos DOM selectores de tu CsvParser
     */
    buildDynamicForm() {
        this.dom.inputsContainer.innerHTML = '';
        
        const allColumns = Object.values(this.ui.datasetDiagnostics);
        
        const inputColumns = allColumns.filter(col => col.domRoleSelect && col.domRoleSelect.value === 'input');

        inputColumns.forEach(col => {
            const group = document.createElement('div');
            group.className = 'bg-slate-950/30 border border-slate-800/40 p-2.5 rounded-lg space-y-1';
            
            if (col.isFullyNumeric) {
                group.innerHTML = `
                    <label class="block text-[11px] text-slate-400 font-medium truncate" title="${col.name}">
                        ${col.name}
                    </label>
                    <input type="number" 
                           data-col-name="${col.name}" 
                           data-type="numeric"
                           step="any" 
                           value="${((col.min + col.max) / 2).toFixed(2)}" 
                           class="modal-input-field w-full bg-slate-900 border border-slate-700/60 rounded p-1 text-xs text-white font-mono focus:outline-none focus:border-indigo-500"
                    />
                    <span class="block text-[9px] text-slate-500 font-mono">Rango: [${col.min}, ${col.max}]</span>
                `;
            } 
            else {
                const optionsHTML = col.uniqueValues.map(val => `<option value="${val}">${val}</option>`).join('');
                group.innerHTML = `
                    <label class="block text-[11px] text-slate-400 font-medium truncate" title="${col.name}">
                        ${col.name}
                    </label>
                    <select data-col-name="${col.name}" data-type="categorical" class="modal-input-field w-full bg-slate-900 border border-slate-700/60 rounded p-1 text-xs text-white focus:outline-none focus:border-indigo-500">
                        ${optionsHTML}
                    </select>
                    <span class="block text-[9px] text-slate-500 font-mono">Categorías: ${col.uniqueValues.length}</span>
                `;
            }

            this.dom.inputsContainer.appendChild(group);
        });
    }

    /**
     * Recopila, transforma y empaqueta el vector emulando al 100% las condiciones de compileDataset
     */
    executeInference() {
        const inputFields = document.querySelectorAll('.modal-input-field');
        const finalInputVector = [];
        const shouldNormalize = this.dom.toggleNormalize.checked;

        for (let field of inputFields) {
            const colName = field.getAttribute('data-col-name');
            const type = field.getAttribute('data-type');
            const col = this.ui.datasetDiagnostics[colName];

            if (type === 'numeric') {
                const num = parseFloat(field.value);
                if (isNaN(num)) {
                    alert('Por favor, ingresa valores numéricos válidos en todos los campos.');
                    return;
                }

                const range = col.max - col.min;
                const normType = col.domContextualElement.value;

                if (range === 0) {
                    finalInputVector.push(0);
                } else if (shouldNormalize && normType === 'minmax_0_1') {
                    finalInputVector.push((num - col.min) / range);
                } else if (shouldNormalize && normType === 'minmax_1_1') {
                    finalInputVector.push(2 * ((num - col.min) / range) - 1);
                } else {
                    finalInputVector.push(num);
                }
            }
            else if (col.uniqueValues.length === 2) {
                const cellValue = field.value;
                const numericRepresentation = (cellValue === col.uniqueValues[0]) ? 0 : 1;
                finalInputVector.push(numericRepresentation);
            } 
            else {
                const cellValue = field.value;
                const oneHotVector = new Array(col.uniqueValues.length).fill(0);
                const categoryIdx = col.uniqueValues.indexOf(cellValue);
                if (categoryIdx !== -1) {
                    oneHotVector[categoryIdx] = 1;
                }
                finalInputVector.push(...oneHotVector);
            }
        }

        const originalTrainingFlag = this.network.isTraining;
        this.network.isTraining = false;
        
        const outputActivations = forwardPass(this.network, finalInputVector);
        
        this.network.isTraining = originalTrainingFlag;

        this.renderResults(outputActivations);
    }

    /**
     * Muestra los resultados interpretando los targets del dataset de forma dinámica
     */
    renderResults(outputs) {
        this.dom.outputsContainer.innerHTML = '';
        
        const allColumns = Object.values(this.ui.datasetDiagnostics);
        const targetCol = allColumns.find(col => col.domRoleSelect && col.domRoleSelect.value === 'output');

        let classNames = [];
        if (targetCol) {
            classNames = targetCol.uniqueValues;
        } else {
            classNames = outputs.map((_, i) => `Salida Neurona ${i}`);
        }

        const maxIndex = outputs.indexOf(Math.max(...outputs));

        outputs.forEach((value, idx) => {
            const isWinner = (idx === maxIndex);
            const percentage = Math.max(0, Math.min(100, value * 100));
            const name = classNames[idx] || `Clase ${idx}`;

            const row = document.createElement('div');
            row.className = `p-2.5 rounded-lg border transition-all ${
                isWinner 
                    ? 'bg-emerald-950/20 border-emerald-500/40 shadow-lg shadow-emerald-500/5' 
                    : 'bg-slate-950/20 border-slate-800/60'
            }`;

            row.innerHTML = `
                <div class="flex justify-between text-xs mb-1">
                    <span class="${isWinner ? 'text-emerald-400 font-bold' : 'text-slate-300'} flex items-center gap-1">
                        ${isWinner ? '🏆' : '▪️'} ${name}
                    </span>
                    <span class="font-mono ${isWinner ? 'text-emerald-400 font-bold' : 'text-slate-400'}">
                        ${value.toFixed(4)} (${percentage.toFixed(1)}%)
                    </span>
                </div>
                <div class="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
                    <div class="h-full transition-all duration-300 ${isWinner ? 'bg-emerald-500' : 'bg-slate-700'}" style="width: ${percentage}%"></div>
                </div>
            `;
            this.dom.outputsContainer.appendChild(row);
        });
    }
}