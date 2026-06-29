import { CollisionDetector } from './CollisionDetector.js';
import { activationRegistry } from '../activation_methods/registry.js';
import { connectionRegistry } from '../connection_methods/registry.js';
import { lossRegistry } from '../loss_methods/registry.js';
import { initializationRegistry } from '../initialization_methods/registry.js';
import { CsvParser } from '../utils/CsvParser.js';
import { TrainingEngine } from '../training/Engine.js';
import { MetricsChart } from './MetricsChart.js';
import { exportModel, importModel } from '../utils/Persistence.js';
import { PredictionModal } from './PredictionModal.js';

export class UIController {
    /**
     * @param {Network} network - Instancia de la red neuronal
     * @param {CanvasRenderer} renderer - Instancia del renderizador
     */
    constructor(network, networkRenderer) {
        this.network = network;
        this.renderer = networkRenderer;
        this.csvParser = new CsvParser();
        this.activeInspectedComponent = null;
        this.datasetDiagnostics = null;

        // Cachear elementos críticos del DOM
        this.dom = {
            btnSave: document.getElementById('btn-save'),
            btnLoad: document.getElementById('btn-load'),
            btnOpenPredictModal: document.getElementById('btn-open-predict-modal'),

            canvas: document.getElementById('neural-canvas'),
            panelLeft: document.getElementById('panel-left'),
            panelRight: document.getElementById('panel-right'),
            btnToggleLeft: document.getElementById('btn-toggle-left'),
            btnToggleRight: document.getElementById('btn-toggle-right'),

            btnPlay: document.getElementById('btn-play'),
            btnPause: document.getElementById('btn-pause'),
            btnStep: document.getElementById('btn-step'),
            btnReset: document.getElementById('btn-reset'),

            dropzone: document.getElementById('csv-dropzone'),
            fileInput: document.getElementById('csv-file-input'),
            mappingArea: document.getElementById('csv-mapping-area'),
            fileNameLabel: document.getElementById('csv-file-name'),
            btnResetCsv: document.getElementById('btn-reset-csv'),
            columnsListContainer: document.getElementById('csv-columns-list'),
            btnConfirmMapping: document.getElementById('btn-confirm-mapping'),
            sliderSplit: document.getElementById('param-split'),
            valSplit: document.getElementById('val-split'),

            sliderLr: document.getElementById('param-lr'),
            valLr: document.getElementById('val-lr'),
            sliderMomentum: document.getElementById('param-momentum'),
            valMomentum: document.getElementById('val-momentum'),
            sliderDropout: document.getElementById('param-dropout'),
            valDropout: document.getElementById('val-dropout'),
            sliderDecay: document.getElementById('param-decay'),
            valDecay: document.getElementById('val-decay'),

            selectBatchSize: document.getElementById('param-batch-size'),
            selectLossType: document.getElementById('param-loss-type'),
            selectInitType: document.getElementById('param-init-type'),
            
            inspectorEmpty: document.getElementById('inspector-empty-state'),
            inspectorContent: document.getElementById('inspector-content'),
            inspectType: document.getElementById('inspect-type'),
            inspectId: document.getElementById('inspect-id'),
            inspectValue: document.getElementById('inspect-value'),
            inspectNet: document.getElementById('inspect-net'),
            dynamicActions: document.getElementById('inspector-dynamic-actions'),
            lblLoss: document.getElementById('lbl-current-loss'),
            lblAccuracy: document.getElementById('lbl-current-accuracy'),
            lblEpoch: document.getElementById('lbl-epoch-counter')
        };

        this.engine = new TrainingEngine(
            this.network,
            (epoch, loss, accuracy, mae) => this.handleEpochComplete(epoch, loss, accuracy, mae), // Al acabar época
            () => this.renderer.render(this.network) // Al avanzar la animación visual
        );

        this.metricsChart = new MetricsChart();
        this.predictionModal = new PredictionModal(this.network, this);

        this.populateLossSelect();
        this.populateInitializationSelect();
        this.initEventListeners();
        this.initCsvListeners();
        this.initPlaybackListeners();
        this.syncMetrics();
    }

    /**
     * Vincula todos los listeners de eventos nativos del navegador
     */
    initEventListeners() {
        this.dom.btnSave.addEventListener('click', () => {
            exportModel(this.network, this);
        });

        this.dom.btnLoad.addEventListener('click', () => {
            this.initModel();
        });

        this.dom.btnOpenPredictModal.addEventListener('click', () => {
            this.predictionModal.open();
        });

        this.dom.canvas.addEventListener('click', (e) => {
            const rect = this.dom.canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;

            const hit = CollisionDetector.findComponentAt(clickX, clickY, this.network, this.renderer.neuronRadius);

            if (hit) {
                this.openInspector(hit.type, hit.object);
            } else {
                this.closeInspector();
            }
        });

        this.dom.selectLossType.addEventListener('change', (e) => {
            this.network.lossTypeId = e.target.value;
        });

        this.dom.sliderLr.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.dom.valLr.innerText = val.toFixed(3);
            this.network.learningRate = val;
        });

        this.dom.sliderMomentum.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.dom.valMomentum.innerText = val.toFixed(2);
            this.network.momentum = val;
        });

        this.dom.sliderDropout.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.dom.valDropout.innerText = val.toFixed(2);
            this.network.dropoutRate = val;
            
            this.renderer.render(this.network); 
        });

        this.dom.sliderDecay.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.dom.valDecay.innerText = val.toFixed(4);
            this.network.weightDecay = val;
        });

        this.dom.selectBatchSize.addEventListener('change', (e) => {
            const val = parseInt(e.target.value, 10);
            this.network.batchSize = val;
        });

        this.dom.selectInitType.addEventListener('change', (e) => {
            this.network.initializationStrategyId = e.target.value;

            if(this.isSyncing) return;

            this.network.applyWeightInitialization();
            this.renderer.render(this.network);
            this.updateInspectorValues();
        });

        this.dom.btnToggleLeft.addEventListener('click', () => this.togglePanel('left'));
        this.dom.btnToggleRight.addEventListener('click', () => this.togglePanel('right'));

        window.addEventListener('resize', () => {
            this.renderer.resize();
            this.renderer.render(this.network);
        });
    }

    /**
     * Sincroniza todos los controles deslizantes (sliders),
     * selectores desplegables (selects) y etiquetas numéricas de la UI con el estado real de la red.
     */
    syncSlidersAndSelects() {
        if (!this.network) return;

        this.isSyncing = true;

        // MAPEO DE SLIDERS Y SUS ETIQUETAS DE TEXTO ASOCIADAS
        const sliderMap = {
            'slider-learning-rate': [this.network.learningRate, this.dom.sliderLr, this.dom.valLr],
            'slider-momentum': [this.network.momentum, this.dom.sliderMomentum, this.dom.valMomentum],
            'slider-dropout': [this.network.dropoutRate, this.dom.sliderDropout, this.dom.valDropout],
            'slider-weight-decay': [this.network.weightDecay, this.dom.sliderDecay, this.dom.valDecay]
        };

        Object.entries(sliderMap).forEach(([controlId, [value, element, elementVal]]) => {
            const slider = element;
            if (slider) {
                slider.value = value;
                slider.dispatchEvent(new Event('input', { bubbles: true }));
                
                const label = elementVal;
                if (label) {
                    label.innerText = value < 0.001 ? value.toFixed(5) : value;
                }
            }
        });

        // MAPEO DE SELECTORES DESPLEGABLES (SELECT)
        const selectMap = {
            'select-batch-size': [this.network.batchSize, this.dom.selectBatchSize],
            'select-loss-type': [this.network.lossTypeId, this.dom.selectLossType],
            'select-initialization': [this.network.initializationStrategyId, this.dom.selectInitType]
        };

        Object.entries(selectMap).forEach(([controlId, [value, element]]) => {
            const selectElement = element;
            if (selectElement) {
                selectElement.value = value;
                
                selectElement.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });

        this.isSyncing = false;
    }

    /**
     * Inicializa los eventos de arrastre y selección del archivo CSV
     */
    initCsvListeners() {
        this.dom.dropzone.addEventListener('click', () => this.dom.fileInput.click());

        this.dom.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) this.handleCsvUpload(e.target.files[0]);
        });

        this.dom.dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dom.dropzone.classList.add('border-indigo-500', 'bg-indigo-950/10');
        });

        this.dom.dropzone.addEventListener('dragleave', () => {
            this.dom.dropzone.classList.remove('border-indigo-500', 'bg-indigo-950/10');
        });

        this.dom.dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dom.dropzone.classList.remove('border-indigo-500', 'bg-indigo-950/10');
            if (e.dataTransfer.files.length > 0) {
                this.dom.fileInput.files = e.dataTransfer.files;
                this.handleCsvUpload(e.dataTransfer.files[0]);
            }
        });

        this.dom.btnResetCsv.addEventListener('click', () => {
            this.dom.fileInput.value = '';
            this.dom.mappingArea.classList.add('hidden');
            this.dom.dropzone.classList.remove('hidden');
            this.datasetDiagnostics = null;
        });

        this.dom.sliderSplit.addEventListener('input', (e) => {
            const trainPercent = parseInt(e.target.value, 10);
            const testPercent = 100 - trainPercent;
            this.dom.valSplit.innerText = `${trainPercent}% / ${testPercent}%`;
        });

        this.dom.btnConfirmMapping.addEventListener('click', () => {
            this.compileSelectedArchitecture();
        });
    }

    /**
     * Conecta los botones físicos de reproducción con los métodos del motor asíncrono
     */
    initPlaybackListeners() {
        if (!this.dom.btnPlay) return;

        // Botón  PLAY
        this.dom.btnPlay.addEventListener('click', () => {
            if (!this.trainSet || this.trainSet.length === 0) {
                alert('Primero debes cargar un archivo CSV y confirmar el mapeo estructural.');
                return;
            }
            this.engine.start();
            this.updatePlaybackButtonsUI();
        });

        // Botón  PAUSE
        this.dom.btnPause.addEventListener('click', () => {
            this.engine.pause();
            this.updatePlaybackButtonsUI();
        });

        // Botón STEP (Paso a paso, época por época)
        this.dom.btnStep.addEventListener('click', () => {
            if (!this.trainSet || this.trainSet.length === 0) return;
            this.engine.step();
        });

        // Botón RESET (Reiniciar pesos y épocas)
        this.dom.btnReset.addEventListener('click', () => {
            this.engine.pause();
            this.network.resetTrainingState();
            this.metricsChart.clear();
            // Refrescar el lienzo físico y los textos de analíticas
            this.renderer.render(this.network);
            this.syncMetrics();
            this.updatePlaybackButtonsUI();
            this.updateInspectorValues();
        });
    }

    /**
     * Captura las analíticas de fin de época emitidas por el motor y refresca los textos de la UI
     */
    handleEpochComplete(epoch, loss, accuracy, mae) {
        if (this.dom.lblEpoch) this.dom.lblEpoch.innerText = epoch;
        if (this.dom.lblLoss) this.dom.lblLoss.innerText = loss.toFixed(5);
        
        const isRegression = mae !== null && mae !== undefined;

        const accuracyHeader = document.getElementById('chart-accuracy')?.closest('.flex-col')?.querySelector('h2');
        
        if (accuracyHeader) {
            accuracyHeader.innerText = isRegression ? "Error Promedio (MAE)" : "Precisión (Accuracy)";
        }

        if (this.dom.lblAccuracy) {
            if (isRegression) {
                const unidad = this.network.targetMetadata?.[0]?.name || "pts";
                this.dom.lblAccuracy.innerText = `±${mae.toFixed(2)} ${unidad}`;
            } else {
                this.dom.lblAccuracy.innerText = `${accuracy.toFixed(2)}%`;
            }
        }
        
        this.updateInspectorValues();

        this.metricsChart.pushMetrics(epoch, loss, isRegression ? mae : accuracy, isRegression);
    }

    /**
     * Altera visualmente los estados de los botones (estilo activo/inactivo) usando Tailwind
     */
    updatePlaybackButtonsUI() {
        if (this.network.isTraining) {
            this.dom.btnPlay.classList.add('bg-indigo-950', 'text-indigo-500', 'border-indigo-800/40');
            this.dom.btnPlay.classList.remove('bg-slate-900', 'text-slate-400');
            
            this.dom.btnPause.classList.remove('bg-indigo-950', 'text-indigo-500', 'border-indigo-800/40');
            this.dom.btnPause.classList.add('bg-slate-900', 'text-slate-400');
        } else {
            this.dom.btnPlay.classList.remove('bg-indigo-950', 'text-indigo-500', 'border-indigo-800/40');
            this.dom.btnPlay.classList.add('bg-slate-900', 'text-slate-400');
            
            this.dom.btnPause.classList.add('bg-indigo-950', 'text-indigo-500', 'border-indigo-800/40');
            this.dom.btnPause.classList.remove('bg-slate-900', 'text-slate-400');
        }
    }

    /**
     * Llena de forma dinámica las opciones del selector de pérdidas leyendo el registro central
     */
    populateLossSelect() {
        if (!this.dom.selectLossType) return;
        
        // Limpiamos cualquier opción hardcodeada previa
        this.dom.selectLossType.innerHTML = '';

        Object.values(lossRegistry).forEach(loss => {
            const option = document.createElement('option');
            option.value = loss.id;
            option.innerText = loss.name;
            option.title = loss.description; // Hint sutil al pasar el mouse
            
            // Si la red tiene esta pérdida activa por defecto, la dejamos seleccionada
            if (this.network.lossTypeId === loss.id) {
                option.selected = true;
            }
            this.dom.selectLossType.appendChild(option);
        });
    }

    /**
     * Llena dinámicamente las opciones del selector de inicialización leyendo su registro central
     */
    populateInitializationSelect() {
        if (!this.dom.selectInitType) return;
        this.dom.selectInitType.innerHTML = '';

        Object.values(initializationRegistry).forEach(strategy => {
            const option = document.createElement('option');
            option.value = strategy.id;
            option.innerText = strategy.name;
            option.title = strategy.description; // Muestra la descripción al pasar el cursor
            
            if (this.network.initializationStrategyId === strategy.id) {
                option.selected = true;
            }
            this.dom.selectInitType.appendChild(option);
        });
    }

    /**
     * Abre y rellena el panel lateral de inspección dinámicamente
     */
    openInspector(type, object) {
        this.activeInspectedComponent = { type, object };
        this.renderer.selectedComponentId = object.id;
        this.renderer.render(this.network);

        this.dom.inspectType.innerText = type.toUpperCase();
        this.dom.inspectId.innerText = object.id;
        
        this.dom.dynamicActions.innerHTML = ''; 

        if (type === 'layer') {
            this.renderLayerInspector(object, this.dom.dynamicActions);
        } else {
            this.renderStandardInspector(type, object, this.dom.dynamicActions);
        }

        this.dom.inspectorEmpty.classList.add('hidden');
        
        this.dom.inspectorContent.classList.remove('hidden'); 
        this.dom.inspectorContent.classList.add('flex');
    }

    /**
     * Actualiza solo los valores numéricos variables dentro del inspector abierto
     */
    updateInspectorValues() {
        if (!this.activeInspectedComponent) return;
        const obj = this.activeInspectedComponent.object;

        if (this.activeInspectedComponent.type === 'neuron') {
            this.dom.inspectValue.innerText = obj.value.toFixed(4);
            this.dom.inspectNet.innerText = obj.netInput.toFixed(4);
        } else if (this.activeInspectedComponent.type === 'connection') {
            this.dom.inspectValue.innerText = obj.weight.toFixed(4);
        }
    }

    /**
     * Cierra el panel de inspección y limpia las selecciones visuales
     */
    closeInspector() {
        this.activeInspectedComponent = null;
        this.renderer.selectedComponentId = null;
        this.renderer.render(this.network);

        this.dom.inspectorContent.classList.remove('flex');
        this.dom.inspectorContent.classList.add('hidden');
        this.dom.inspectorEmpty.classList.remove('hidden');
    }

    /**
     * Gestiona el colapso fluido de los paneles y redibuja el canvas en su nuevo tamaño expandido
     */
    togglePanel(side) {
        const panel = side === 'left' ? this.dom.panelLeft : this.dom.panelRight;
        const btn = side === 'left' ? this.dom.btnToggleLeft : this.dom.btnToggleRight;

        if (panel.classList.contains('w-80')) {
            panel.classList.remove('w-80', 'p-4', 'border');
            panel.classList.add('w-0', 'p-0', 'border-0', 'overflow-hidden');
            btn.innerText = side === 'left' ? '▶' : '◀';
        } else {
            panel.classList.remove('w-0', 'p-0', 'border-0', 'overflow-hidden');
            panel.classList.add('w-80', 'p-4', 'border');
            btn.innerText = side === 'left' ? '◀' : '▶';
        }

        setTimeout(() => {
            this.renderer.resize();
            this.renderer.render(this.network);
        }, 310);
    }

    /**
     * Sincroniza los textos de métricas globales leyendo directamente desde el objeto Network
     */
    syncMetrics() {
        this.dom.lblLoss.innerText = this.network.currentLoss.toFixed(4);
        this.dom.lblAccuracy.innerText = `${(this.network.currentAccuracy * 100).toFixed(2)}%`;
        this.dom.lblEpoch.innerText = String(this.network.epoch).padStart(4, '0');
        this.updateInspectorValues();
    }

    /**
     * Renderiza las opciones dinámicas de manipulación de la capa en el Inspector
     */
    renderLayerInspector(layer, container) {
        document.getElementById('row-inspect-value').style.display = 'none';
        document.getElementById('row-inspect-net').style.display = 'none';

        // Selector de Función de Activación de la Capa
        const labelAct = document.createElement('label');
        labelAct.className = 'block text-slate-400 mb-1 text-xs';
        labelAct.innerText = 'Función de Activación:';
        
        const selectAct = document.createElement('select');
        selectAct.className = 'w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-white mb-3 focus:outline-none focus:border-indigo-500';
        
        Object.values(activationRegistry).forEach(act => {
            const opt = document.createElement('option');
            opt.value = act.id;
            opt.innerText = act.name;
            if (layer.activationId === act.id) opt.selected = true;
            selectAct.appendChild(opt);
        });

        selectAct.addEventListener('change', (e) => {
            layer.activationId = e.target.value;
            layer.neurons.forEach(n => n.activationId = e.target.value);
        });
        container.appendChild(labelAct);
        container.appendChild(selectAct);

        if (layer.type !== 'output') {
            const labelConn = document.createElement('label');
            labelConn.className = 'block text-slate-400 mb-1 text-xs';
            labelConn.innerText = 'Método de Conexión Saliente:';
            
            const selectConn = document.createElement('select');
            selectConn.className = 'w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-white mb-3 focus:outline-none focus:border-indigo-500';
            
            Object.values(connectionRegistry).forEach(conn => {
                const opt = document.createElement('option');
                opt.value = conn.id;
                opt.innerText = conn.name;
                if (layer.connectionTypeId === conn.id) opt.selected = true;
                selectConn.appendChild(opt);
            });

            selectConn.addEventListener('change', (e) => {
                layer.connectionTypeId = e.target.value;
                this.network.rebuildConnections();
                this.renderer.render(this.network);
            });
            container.appendChild(labelConn);
            container.appendChild(selectConn);
        }

        // Información de Neuronas
        const labelNeurons = document.createElement('p');
        labelNeurons.className = 'text-slate-400 mb-1 text-xs';
        labelNeurons.innerText = `Neuronas Actuales: ${layer.neurons.length}`;
        container.appendChild(labelNeurons);

        if (layer.type === 'hidden') {
            const btnAddN = document.createElement('button');
            btnAddN.className = 'w-full bg-slate-700 hover:bg-slate-600 text-white p-2 rounded-lg text-xs font-semibold transition-colors mb-4';
            btnAddN.innerText = '➕ Añadir Neurona a la Capa';
            btnAddN.onclick = () => {
                this.network.addNeuronToLayer(layer.id);
                this.openInspector('layer', layer);
            };
            container.appendChild(btnAddN);
        } else {
            const notice = document.createElement('div');
            notice.className = 'text-[11px] bg-indigo-950/40 text-indigo-400 border border-indigo-900/50 p-2.5 rounded-lg leading-relaxed mb-4';
            notice.innerHTML = layer.type === 'input'
                ? `ℹ️ El tamaño de la capa de <b>Entrada</b> depende de las columnas de entrada de datos.`
                : `ℹ️ El tamaño de la capa de <b>Salida</b> depende de las columnas de salida de datos.`;
            container.appendChild(notice);
        }

        const idx = this.network.layers.findIndex(l => l.id === layer.id);

        // Insertar Capa a la Izquierda (Disponible en Ocultas y Salida)
        if (layer.type === 'hidden' || layer.type === 'output') {
            const btnLeft = document.createElement('button');
            btnLeft.className = 'w-full bg-indigo-600 hover:bg-indigo-500 text-white p-2 rounded-lg text-xs mb-2 font-medium transition-colors';
            btnLeft.innerText = '⬅️ Insertar Capa a la Izquierda';
            btnLeft.onclick = () => {
                this.network.insertHiddenLayerAt(idx);
                this.closeInspector();
            };
            container.appendChild(btnLeft);
        }

        // Insertar Capa a la Derecha (Disponible en Entrada y Ocultas)
        if (layer.type === 'input' || layer.type === 'hidden') {
            const btnRight = document.createElement('button');
            btnRight.className = 'w-full bg-indigo-600 hover:bg-indigo-500 text-white p-2 rounded-lg text-xs mb-4 font-medium transition-colors';
            btnRight.innerText = '➡️ Insertar Capa a la Derecha';
            btnRight.onclick = () => {
                this.network.insertHiddenLayerAt(idx + 1);
                this.closeInspector();
            };
            container.appendChild(btnRight);
        }

        // Eliminar Capa
        if (layer.type === 'hidden') {
            const btnDeleteLayer = document.createElement('button');
            btnDeleteLayer.className = 'w-full bg-red-600 hover:bg-red-500 text-white p-2 rounded-lg text-xs font-bold transition-colors';
            btnDeleteLayer.innerText = '🗑️ Eliminar Capa Completa';
            btnDeleteLayer.onclick = () => {
                this.network.removeLayer(layer.id);
                this.closeInspector();
            };
            container.appendChild(btnDeleteLayer);
        }
    }

    /**
     * Re-construye la vista estándar del inspector para Neuronas y Conexiones
     */
    renderStandardInspector(type, object, container) {
        document.getElementById('row-inspect-value').style.display = 'flex';
        
        const label = document.createElement('label');
        label.className = 'block text-slate-400 mb-1 text-xs';
        label.innerText = type === 'neuron' ? 'Sesgo (Bias):' : 'Peso (Weight):';

        const wrapper = document.createElement('div');
        wrapper.className = 'flex space-x-2 mb-4';

        const input = document.createElement('input'); 
        input.type = 'number';
        input.step = '0.01';
        input.className = 'flex-1 bg-slate-900 border border-slate-700 rounded-lg p-2 font-mono text-xs text-white focus:outline-none focus:border-indigo-500';
        input.value = type === 'neuron' ? object.bias.toFixed(4) : object.weight.toFixed(4);

        const btn = document.createElement('button');
        btn.className = 'px-3 bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded-lg text-xs transition-colors';
        btn.innerText = 'Aplicar';
        btn.onclick = () => {
            const val = parseFloat(input.value);
            if (isNaN(val)) return;
            if (type === 'neuron') {
                object.bias = val;
                object.activate();
            } else {
                object.weight = val;
            }
            this.renderer.render(this.network);
            this.updateInspectorValues();
        };

        wrapper.appendChild(input);
        wrapper.appendChild(btn);
        container.appendChild(label);
        container.appendChild(wrapper);

        const currentLayerIdx = this.network.layers.findIndex(l => l.neurons.some(n => n.id === object.id));

        if (type === 'neuron') {
            document.getElementById('row-inspect-net').style.display = 'flex';
            this.dom.inspectValue.innerText = object.value.toFixed(4);
            this.dom.inspectNet.innerText = object.netInput.toFixed(4);

            const labelAct = document.createElement('label');
            labelAct.className = 'block text-slate-400 mb-1 text-xs pt-1';
            labelAct.innerText = 'Función de Activación:';
            
            const selectAct = document.createElement('select');
            selectAct.className = 'w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-white mb-4 focus:outline-none focus:border-indigo-500';
            
            Object.values(activationRegistry).forEach(act => {
                const opt = document.createElement('option');
                opt.value = act.id;
                opt.innerText = act.name;
                if (object.activationId === act.id) opt.selected = true;
                selectAct.appendChild(opt);
            });

            selectAct.addEventListener('change', (e) => {
                object.activationId = e.target.value;
                object.activate();
                this.renderer.render(this.network);
                this.updateInspectorValues();
            });
            container.appendChild(labelAct);
            container.appendChild(selectAct);

            if (currentLayerIdx > 0) {
                const prevLayer = this.network.layers[currentLayerIdx - 1];
                
                const unconnectedPrevNeurons = prevLayer.neurons.filter(prevNeuron => 
                    !prevNeuron.outputs.some(conn => conn.to.id === object.id)
                );

                if (unconnectedPrevNeurons.length > 0) {
                    const divider = document.createElement('hr');
                    divider.className = 'border-slate-800 my-3';
                    container.appendChild(divider);

                    const labelLeft = document.createElement('label');
                    labelLeft.className = 'block text-slate-400 mb-1 text-xs';
                    labelLeft.innerText = '🔌 Traer Enlace desde la Izquierda:';

                    const selectLeft = document.createElement('select');
                    selectLeft.className = 'w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-white mb-2 focus:outline-none focus:border-indigo-500';
                    
                    unconnectedPrevNeurons.forEach(n => {
                        const opt = document.createElement('option');
                        opt.value = n.id;
                        opt.innerText = `Nodo origen (${n.id.split('_').pop()})`;
                        selectLeft.appendChild(opt);
                    });

                    const btnAddLeft = document.createElement('button');
                    btnAddLeft.className = 'w-full bg-indigo-950 hover:bg-indigo-900 text-indigo-300 border border-indigo-800/50 p-2 rounded-lg text-xs font-semibold transition-colors mb-2';
                    btnAddLeft.innerText = '⚡ Conectar desde la Izquierda';
                    btnAddLeft.onclick = () => {
                        const sourceNeuron = prevLayer.neurons.find(n => n.id === selectLeft.value);
                        if (sourceNeuron) {
                            this.network.addManualConnection(sourceNeuron, object);
                            this.openInspector('neuron', object);
                        }
                    };

                    container.appendChild(labelLeft);
                    container.appendChild(selectLeft);
                    container.appendChild(btnAddLeft);
                }
            }


            if (currentLayerIdx !== -1 && currentLayerIdx < this.network.layers.length - 1) {
                const nextLayer = this.network.layers[currentLayerIdx + 1];
                
                const unconnectedNextNeurons = nextLayer.neurons.filter(nextNeuron => 
                    !object.outputs.some(conn => conn.to.id === nextNeuron.id)
                );

                if (unconnectedNextNeurons.length > 0) {
                    const divider = document.createElement('hr');
                    divider.className = 'border-slate-800 my-3';
                    container.appendChild(divider);

                    const labelRight = document.createElement('label');
                    labelRight.className = 'block text-slate-400 mb-1 text-xs';
                    labelRight.innerText = '🔌 Enviar Enlace hacia la Derecha:';

                    const selectRight = document.createElement('select');
                    selectRight.className = 'w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-white mb-2 focus:outline-none focus:border-indigo-500';
                    
                    unconnectedNextNeurons.forEach(n => {
                        const opt = document.createElement('option');
                        opt.value = n.id;
                        opt.innerText = `Nodo destino (${n.id.split('_').pop()})`;
                        selectRight.appendChild(opt);
                    });

                    const btnAddRight = document.createElement('button');
                    btnAddRight.className = 'w-full bg-indigo-950 hover:bg-indigo-900 text-indigo-300 border border-indigo-800/50 p-2 rounded-lg text-xs font-semibold transition-colors mb-4';
                    btnAddRight.innerText = '⚡ Conectar hacia la Derecha';
                    btnAddRight.onclick = () => {
                        const targetNeuron = nextLayer.neurons.find(n => n.id === selectRight.value);
                        if (targetNeuron) {
                            this.network.addManualConnection(object, targetNeuron);
                            this.openInspector('neuron', object);
                        }
                    };

                    container.appendChild(labelRight);
                    container.appendChild(selectRight);
                    container.appendChild(btnAddRight);
                }
            }

            const dividerFinal = document.createElement('hr');
            dividerFinal.className = 'border-slate-800 my-3';
            container.appendChild(dividerFinal);

            const layerAsociada = this.network.layers[currentLayerIdx];
            const btnDeleteN = document.createElement('button');
            
            if (layerAsociada && layerAsociada.neurons.length <= 1) {
                btnDeleteN.disabled = true;
                btnDeleteN.className = 'w-full bg-slate-800 text-slate-500 p-2 rounded-lg text-xs font-bold cursor-not-allowed mt-2 border border-slate-700/50';
                btnDeleteN.innerText = '🔒 Mínimo 1 Neurona Requerida';
            } else {
                btnDeleteN.className = 'w-full bg-rose-950 hover:bg-rose-900 text-rose-300 border border-rose-800/60 p-2 rounded-lg text-xs font-bold transition-colors mt-2';
                btnDeleteN.innerText = '🗑️ Eliminar esta Neurona';
                btnDeleteN.onclick = () => {
                    const exito = this.network.removeNeuron(object.id);
                    if (exito) this.closeInspector();
                };
            }
            container.appendChild(btnDeleteN);
            
        } else {
            document.getElementById('row-inspect-net').style.display = 'none';
            this.dom.inspectValue.innerText = object.weight.toFixed(4);

            const btnDeleteConn = document.createElement('button');
            btnDeleteConn.className = 'w-full bg-rose-950 hover:bg-rose-900 text-rose-300 border border-rose-800/60 p-2 rounded-lg text-xs font-bold transition-colors mt-4';
            btnDeleteConn.innerText = '🗑️ Romper esta Conexión';
            btnDeleteConn.onclick = () => {
                const exito = this.network.removeConnection(object.id);
                if (exito) this.closeInspector();
            };
            container.appendChild(btnDeleteConn);
        }
    }

    /**
     * Procesa el archivo y dispara el renderizado de los selectores alineados
     */
    async handleCsvUpload(file) {
        try {
            this.dom.fileNameLabel.innerText = file.name;
            this.datasetDiagnostics = await this.csvParser.scanFile(file);

            this.dom.dropzone.classList.add('hidden');
            this.dom.mappingArea.classList.remove('hidden');

            this.renderColumnMappers();
        } catch (error) {
            alert(error.message);
        }
    }

    /**
     * Pinta cada columna de forma ultra-compacta, adaptando las opciones a su naturaleza matemática
     */
    renderColumnMappers() {
        this.dom.columnsListContainer.innerHTML = '';

        Object.values(this.datasetDiagnostics).forEach(col => {
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

    /**
     * Recolecta el mapeo del usuario, cuenta las entradas/salidas finales y reconfigura la red en caliente
     */
    compileSelectedArchitecture() {
        let inputCount = 0;
        let outputCount = 0;
        const classWeights = [];
        this.network.targetMetadata = [];

        Object.values(this.datasetDiagnostics).forEach(col => {
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

        this.network.buildArchitecture(inputCount, [4], outputCount);
        this.renderer.render(this.network);
        this.closeInspector();

        const fullDataset = this.csvParser.compileDataset(this.datasetDiagnostics);

        for (let i = fullDataset.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [fullDataset[i], fullDataset[j]] = [fullDataset[j], fullDataset[i]];
        }

        const trainPercent = parseInt(this.dom.sliderSplit.value, 10) / 100;
        const cutoff = Math.floor(fullDataset.length * trainPercent);

        this.trainSet = fullDataset.slice(0, cutoff);
        this.testSet = fullDataset.slice(cutoff);

        this.engine.setDatasets(this.trainSet, this.testSet);

        this.syncMetrics();
        this.updatePlaybackButtonsUI();
    }

    initModel() {
        if(!confirm('Esto eliminará la configuración actual. ¿Deseas continuar?')) return;

        const fileInput = document.createElement('input');
        fileInput.id = 'hidden-file-input';
        fileInput.type = 'file';
        fileInput.accept = '.json';
        fileInput.className = 'hidden';
        fileInput.click();

        fileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            
            reader.onload = (e) => {
                try {
                    const jsonData = JSON.parse(e.target.result);
                    
                    const success = importModel(jsonData, this.network, this, this.renderer);
                    
                } catch (err) {
                    alert("El archivo seleccionado no contiene un formato JSON válido.");
                }
                finally {
                    fileInput.remove();
                }
            };

            reader.onerror = () => {
                alert("Error físico de lectura al intentar procesar el archivo.");
                fileInput.remove();
            };

            reader.readAsText(file);
            fileInput.remove();
        });
    }
}