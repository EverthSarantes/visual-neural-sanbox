import { CollisionDetector } from './CollisionDetector.js';
import { activationRegistry } from '../activation_methods/registry.js';
import { connectionRegistry } from '../connection_methods/registry.js';
import { lossRegistry } from '../loss_methods/registry.js';
import { initializationRegistry } from '../initialization_methods/registry.js';

export class UIController {
    /**
     * @param {Network} network - Instancia de la red neuronal
     * @param {CanvasRenderer} renderer - Instancia del renderizador
     */
    constructor(network, networkRenderer) {
        this.network = network;
        this.renderer = networkRenderer;
        
        // Almacenar el componente que se está inspeccionando actualmente
        this.activeInspectedComponent = null;

        // Cachear elementos críticos del DOM
        this.dom = {
            canvas: document.getElementById('neural-canvas'),
            panelLeft: document.getElementById('panel-left'),
            panelRight: document.getElementById('panel-right'),
            btnToggleLeft: document.getElementById('btn-toggle-left'),
            btnToggleRight: document.getElementById('btn-toggle-right'),
            sliderLr: document.getElementById('param-lr'),
            valLr: document.getElementById('val-lr'),
            selectLossType: document.getElementById('param-loss-type'),
            selectInitType: document.getElementById('param-init-type'),
            inspectorEmpty: document.getElementById('inspector-empty-state'),
            inspectorContent: document.getElementById('inspector-content'),
            inspectType: document.getElementById('inspect-type'),
            inspectId: document.getElementById('inspect-id'),
            inspectValue: document.getElementById('inspect-value'),
            inspectNet: document.getElementById('inspect-net'),
            inspectEditableLabel: document.getElementById('inspect-editable-label'),
            inspectEditableValue: document.getElementById('inspect-editable-value'),
            btnApplyInspect: document.getElementById('btn-apply-inspect'),
            dynamicActions: document.getElementById('inspector-dynamic-actions'),
            lblLoss: document.getElementById('lbl-current-loss'),
            lblAccuracy: document.getElementById('lbl-current-accuracy'),
            lblEpoch: document.getElementById('lbl-epoch-counter')
        };

        this.populateLossSelect();
        this.populateInitializationSelect();
        this.initEventListeners();
        this.syncMetrics();
    }

    /**
     * Vincula todos los listeners de eventos nativos del navegador
     */
    initEventListeners() {
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

        this.dom.selectInitType.addEventListener('change', (e) => {
            this.network.initializationStrategyId = e.target.value;
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

        // Información de Neuronas y Botón de Añadir
        const labelNeurons = document.createElement('p');
        labelNeurons.className = 'text-slate-400 mb-1 text-xs';
        labelNeurons.innerText = `Neuronas Actuales: ${layer.neurons.length}`;
        container.appendChild(labelNeurons);

        const btnAddN = document.createElement('button');
        btnAddN.className = 'w-full bg-slate-700 hover:bg-slate-600 text-white p-2 rounded-lg text-xs font-semibold transition-colors mb-4';
        btnAddN.innerText = '➕ Añadir Neurona a la Capa';
        btnAddN.onclick = () => {
            this.network.addNeuronToLayer(layer.id);
            this.openInspector('layer', layer);
        };
        container.appendChild(btnAddN);

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
}