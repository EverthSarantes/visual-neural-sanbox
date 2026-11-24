import { CollisionDetector } from './CollisionDetector.js';
import { activationRegistry } from '../activation_methods/registry.js';
import { connectionRegistry } from '../connection_methods/registry.js';
import { lossRegistry } from '../loss_methods/registry.js';
import { initializationRegistry } from '../initialization_methods/registry.js';
import { CsvParser } from '../utils/CsvParser.js';
import { TrainingClient } from '../training/TrainingClient.js';
import { MetricsChart } from './MetricsChart.js';
import { exportModel, importModel } from '../utils/Persistence.js';
import { PredictionModal } from './PredictionModal.js';
import { CsvController } from './CsvController.js';

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

        this.trainingClient = new TrainingClient();

        // Cuando el worker MLP termina una época, actualizamos la red local y la UI
        this.trainingClient.on('ready', (payload) => {
            if (payload?.network) this.network.applySnapshot(payload.network);
            this.renderer.render(this.network);
            this.syncMetrics();
        });

        this.trainingClient.on('epoch', (payload) => {
            if (payload?.network) this.network.applySnapshot(payload.network);
            this.handleEpochComplete(payload.epoch, payload.loss, payload.accuracy, payload.mae);
            this.renderer.render(this.network);
        });

        this.trainingClient.on('resetDone', (payload) => {
            if (payload?.network) this.network.applySnapshot(payload.network);
            this.metricsChart.clear();
            this.renderer.render(this.network);
            this.syncMetrics();
            this.updateInspectorValues();
            this.updatePlaybackButtonsUI();
        });

        this.trainingClient.on('paused', () => {
            this.updatePlaybackButtonsUI();
        });

        this.trainingClient.on('error', (payload) => {
            console.error('Worker error:', payload?.message);
            alert('Error en el worker de entrenamiento: ' + (payload?.message || 'desconocido'));
        });



        this.metricsChart = new MetricsChart();
        this.predictionModal = new PredictionModal(this.network, this);
        this.csvController = new CsvController(this);

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
            this.trainingClient.setHyperparams({ lossTypeId: e.target.value });
        });

        this.dom.sliderLr.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.dom.valLr.innerText = val.toFixed(3);
            this.network.learningRate = val;
            this.trainingClient.setHyperparams({ learningRate: val });
        });

        this.dom.sliderMomentum.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.dom.valMomentum.innerText = val.toFixed(2);
            this.network.momentum = val;
            this.trainingClient.setHyperparams({ momentum: val });
        });

        this.dom.sliderDropout.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.dom.valDropout.innerText = val.toFixed(2);
            this.network.dropoutRate = val;
            this.trainingClient.setHyperparams({ dropoutRate: val });

            this.renderer.render(this.network); 
        });

        this.dom.sliderDecay.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.dom.valDecay.innerText = val.toFixed(4);
            this.network.weightDecay = val;
            this.trainingClient.setHyperparams({ weightDecay: val });
        });

        this.dom.selectBatchSize.addEventListener('change', (e) => {
            const val = parseInt(e.target.value, 10);
            this.network.batchSize = val;
            this.trainingClient.setHyperparams({ batchSize: val });
        });

        this.dom.selectInitType.addEventListener('change', (e) => {
            this.network.initializationStrategyId = e.target.value;

            if (this.isSyncing) return;

            this.network.applyWeightInitialization();
            this.renderer.render(this.network);
            this.updateInspectorValues();
            this.trainingClient.reinit(e.target.value);
        });

        this.dom.btnToggleLeft.addEventListener('click', () => this.togglePanel('left'));
        this.dom.btnToggleRight.addEventListener('click', () => this.togglePanel('right'));

        window.addEventListener('resize', () => {
            // Normalizar estado de paneles al cambiar de breakpoint
            if (window.innerWidth >= 1024) {
                this.dom.panelLeft.classList.remove('-translate-x-full', 'translate-x-full');
                this.dom.panelRight.classList.remove('-translate-x-full', 'translate-x-full');
            }
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
            if (e.target.files.length > 0) this.csvController.handleCsvUpload(e.target.files[0]);
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
                this.csvController.handleCsvUpload(e.dataTransfer.files[0]);
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
            this.csvController.compileSelectedArchitecture();
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
            this.network.isTraining = true;
            this.trainingClient.start();
            this.updatePlaybackButtonsUI();
        });

        // Botón  PAUSE
        this.dom.btnPause.addEventListener('click', () => {
            this.network.isTraining = false;
            this.trainingClient.pause();
            this.updatePlaybackButtonsUI();
        });

        // Botón STEP (Paso a paso, época por época)
        this.dom.btnStep.addEventListener('click', () => {
            if (!this.trainSet || this.trainSet.length === 0) return;
            this.trainingClient.step();
        });

        // Botón RESET (Reiniciar pesos y épocas)
        this.dom.btnReset.addEventListener('click', () => {
            this.network.isTraining = false;
            this.trainingClient.pause();
            this.trainingClient.reset();
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
        const isMobile = window.innerWidth < 1024;

        if (isMobile) {
            const hiddenClass = side === 'left' ? '-translate-x-full' : 'translate-x-full';
            const isOpen = !panel.classList.contains(hiddenClass);
            if (isOpen) {
                panel.classList.add(hiddenClass);
                btn.innerText = side === 'left' ? '▶' : '◀';
            } else {
                // Cerrar el otro panel si está abierto
                const otherSide = side === 'left' ? 'right' : 'left';
                const otherPanel = otherSide === 'left' ? this.dom.panelLeft : this.dom.panelRight;
                const otherHidden = otherSide === 'left' ? '-translate-x-full' : 'translate-x-full';
                otherPanel.classList.add(otherHidden);
                panel.classList.remove(hiddenClass);
                btn.innerText = side === 'left' ? '◀' : '▶';
            }
        } else {
            if (panel.classList.contains('w-0')) {
                panel.classList.remove('w-0', 'p-0', 'border-0', 'overflow-hidden');
                panel.classList.add('w-80', 'p-4', 'border');
                btn.innerText = side === 'left' ? '◀' : '▶';
            } else {
                panel.classList.remove('w-72', 'sm:w-80', 'w-80', 'p-4', 'border');
                panel.classList.add('w-0', 'p-0', 'border-0', 'overflow-hidden');
                btn.innerText = side === 'left' ? '▶' : '◀';
            }
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
        this.dom.lblLoss.innerText = (this.network.currentLoss || 0).toFixed(4);

        const isRegression = this.network.currentMAE !== null && this.network.currentMAE !== undefined;
        if (isRegression) {
            const unidad = this.network.targetMetadata?.[0]?.name || 'pts';
            this.dom.lblAccuracy.innerText = `±${(this.network.currentMAE || 0).toFixed(2)} ${unidad}`;
        } else {
            this.dom.lblAccuracy.innerText = `${(this.network.currentAccuracy || 0).toFixed(2)}%`;
        }

        this.dom.lblEpoch.innerText = String(this.network.epoch || 0).padStart(4, '0');
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
            if (this.trainSet && this.trainSet.length > 0) {
                this.trainingClient.setNetwork(this.network);
            }
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
                if (this.trainSet && this.trainSet.length > 0) {
                    this.trainingClient.init({
                        inputCount: this.network.layers[0].neurons.length,
                        hiddenTopology: this.network.layers.slice(1, -1).map(l => l.neurons.length),
                        outputCount: this.network.layers[this.network.layers.length - 1].neurons.length,
                        outputActivation: this.network.layers[this.network.layers.length - 1].activationId,
                        lossTypeId: this.network.lossTypeId,
                        learningRate: this.network.learningRate,
                        momentum: this.network.momentum,
                        weightDecay: this.network.weightDecay,
                        dropoutRate: this.network.dropoutRate,
                        batchSize: this.network.batchSize,
                        initializationStrategyId: this.network.initializationStrategyId,
                        epoch: this.network.epoch,
                        currentLoss: this.network.currentLoss,
                        currentAccuracy: this.network.currentAccuracy,
                        currentMAE: this.network.currentMAE,
                        targetMetadata: this.network.targetMetadata,
                        classWeights: this.network.classWeights,
                        useClassWeighting: this.network.useClassWeighting,
                        trainSet: this.trainSet,
                        testSet: this.testSet,
                    });
                }
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
                if (this.trainSet && this.trainSet.length > 0) {
                    this.trainingClient.init({
                        inputCount: this.network.layers[0].neurons.length,
                        hiddenTopology: this.network.layers.slice(1, -1).map(l => l.neurons.length),
                        outputCount: this.network.layers[this.network.layers.length - 1].neurons.length,
                        outputActivation: this.network.layers[this.network.layers.length - 1].activationId,
                        lossTypeId: this.network.lossTypeId,
                        learningRate: this.network.learningRate,
                        momentum: this.network.momentum,
                        weightDecay: this.network.weightDecay,
                        dropoutRate: this.network.dropoutRate,
                        batchSize: this.network.batchSize,
                        initializationStrategyId: this.network.initializationStrategyId,
                        epoch: this.network.epoch,
                        currentLoss: this.network.currentLoss,
                        currentAccuracy: this.network.currentAccuracy,
                        currentMAE: this.network.currentMAE,
                        targetMetadata: this.network.targetMetadata,
                        classWeights: this.network.classWeights,
                        useClassWeighting: this.network.useClassWeighting,
                        trainSet: this.trainSet,
                        testSet: this.testSet,
                    });
                }
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
                if (this.trainSet && this.trainSet.length > 0) {
                    this.trainingClient.init({
                        inputCount: this.network.layers[0].neurons.length,
                        hiddenTopology: this.network.layers.slice(1, -1).map(l => l.neurons.length),
                        outputCount: this.network.layers[this.network.layers.length - 1].neurons.length,
                        outputActivation: this.network.layers[this.network.layers.length - 1].activationId,
                        lossTypeId: this.network.lossTypeId,
                        learningRate: this.network.learningRate,
                        momentum: this.network.momentum,
                        weightDecay: this.network.weightDecay,
                        dropoutRate: this.network.dropoutRate,
                        batchSize: this.network.batchSize,
                        initializationStrategyId: this.network.initializationStrategyId,
                        epoch: this.network.epoch,
                        currentLoss: this.network.currentLoss,
                        currentAccuracy: this.network.currentAccuracy,
                        currentMAE: this.network.currentMAE,
                        targetMetadata: this.network.targetMetadata,
                        classWeights: this.network.classWeights,
                        useClassWeighting: this.network.useClassWeighting,
                        trainSet: this.trainSet,
                        testSet: this.testSet,
                    });
                }
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
                if (this.trainSet && this.trainSet.length > 0) {
                    this.trainingClient.init({
                        inputCount: this.network.layers[0].neurons.length,
                        hiddenTopology: this.network.layers.slice(1, -1).map(l => l.neurons.length),
                        outputCount: this.network.layers[this.network.layers.length - 1].neurons.length,
                        outputActivation: this.network.layers[this.network.layers.length - 1].activationId,
                        lossTypeId: this.network.lossTypeId,
                        learningRate: this.network.learningRate,
                        momentum: this.network.momentum,
                        weightDecay: this.network.weightDecay,
                        dropoutRate: this.network.dropoutRate,
                        batchSize: this.network.batchSize,
                        initializationStrategyId: this.network.initializationStrategyId,
                        epoch: this.network.epoch,
                        currentLoss: this.network.currentLoss,
                        currentAccuracy: this.network.currentAccuracy,
                        currentMAE: this.network.currentMAE,
                        targetMetadata: this.network.targetMetadata,
                        classWeights: this.network.classWeights,
                        useClassWeighting: this.network.useClassWeighting,
                        trainSet: this.trainSet,
                        testSet: this.testSet,
                    });
                }
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
                if (this.trainSet && this.trainSet.length > 0) {
                    this.trainingClient.init({
                        inputCount: this.network.layers[0].neurons.length,
                        hiddenTopology: this.network.layers.slice(1, -1).map(l => l.neurons.length),
                        outputCount: this.network.layers[this.network.layers.length - 1].neurons.length,
                        outputActivation: this.network.layers[this.network.layers.length - 1].activationId,
                        lossTypeId: this.network.lossTypeId,
                        learningRate: this.network.learningRate,
                        momentum: this.network.momentum,
                        weightDecay: this.network.weightDecay,
                        dropoutRate: this.network.dropoutRate,
                        batchSize: this.network.batchSize,
                        initializationStrategyId: this.network.initializationStrategyId,
                        epoch: this.network.epoch,
                        currentLoss: this.network.currentLoss,
                        currentAccuracy: this.network.currentAccuracy,
                        currentMAE: this.network.currentMAE,
                        targetMetadata: this.network.targetMetadata,
                        classWeights: this.network.classWeights,
                        useClassWeighting: this.network.useClassWeighting,
                        trainSet: this.trainSet,
                        testSet: this.testSet,
                    });
                }
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

            if (this.trainSet && this.trainSet.length > 0) {
                this.trainingClient.setNetwork(this.network);
            }
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
                if (this.trainSet && this.trainSet.length > 0) {
                    this.trainingClient.setNetwork(this.network);
                }
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
                            if (this.trainSet && this.trainSet.length > 0) {
                                this.trainingClient.setNetwork(this.network);
                            }
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
                            if (this.trainSet && this.trainSet.length > 0) {
                                this.trainingClient.setNetwork(this.network);
                            }
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
                    if (exito) {
                        this.closeInspector();
                        if (this.trainSet && this.trainSet.length > 0) {
                            this.trainingClient.init({
                                inputCount: this.network.layers[0].neurons.length,
                                hiddenTopology: this.network.layers.slice(1, -1).map(l => l.neurons.length),
                                outputCount: this.network.layers[this.network.layers.length - 1].neurons.length,
                                outputActivation: this.network.layers[this.network.layers.length - 1].activationId,
                                lossTypeId: this.network.lossTypeId,
                                learningRate: this.network.learningRate,
                                momentum: this.network.momentum,
                                weightDecay: this.network.weightDecay,
                                dropoutRate: this.network.dropoutRate,
                                batchSize: this.network.batchSize,
                                initializationStrategyId: this.network.initializationStrategyId,
                                epoch: this.network.epoch,
                                currentLoss: this.network.currentLoss,
                                currentAccuracy: this.network.currentAccuracy,
                                currentMAE: this.network.currentMAE,
                                targetMetadata: this.network.targetMetadata,
                                classWeights: this.network.classWeights,
                                useClassWeighting: this.network.useClassWeighting,
                                trainSet: this.trainSet,
                                testSet: this.testSet,
                            });
                        }
                    }
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
                if (exito) {
                    this.closeInspector();
                    if (this.trainSet && this.trainSet.length > 0) {
                        this.trainingClient.setNetwork(this.network);
                    }
                }
            };
            container.appendChild(btnDeleteConn);
        }
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

                    if (success) {
                        // Sincronizar el worker con el modelo importado
                        this.trainingClient.init({
                            inputCount: this.network.layers[0].neurons.length,
                            hiddenTopology: this.network.layers.slice(1, -1).map(l => l.neurons.length),
                            outputCount: this.network.layers[this.network.layers.length - 1].neurons.length,
                            outputActivation: this.network.layers[this.network.layers.length - 1].activationId,
                            lossTypeId: this.network.lossTypeId,
                            learningRate: this.network.learningRate,
                            momentum: this.network.momentum,
                            weightDecay: this.network.weightDecay,
                            dropoutRate: this.network.dropoutRate,
                            batchSize: this.network.batchSize,
                            initializationStrategyId: this.network.initializationStrategyId,
                            epoch: this.network.epoch,
                            currentLoss: this.network.currentLoss,
                            currentAccuracy: this.network.currentAccuracy,
                            currentMAE: this.network.currentMAE,
                            targetMetadata: this.network.targetMetadata,
                            classWeights: this.network.classWeights,
                            useClassWeighting: this.network.useClassWeighting,
                            trainSet: this.trainSet || [],
                            testSet: this.testSet || [],
                        });
                    }
                    
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
