export class CanvasRenderer {
    /**
     * @param {HTMLCanvasElement} canvasElement - El elemento canvas del DOM
     */
    constructor(canvasElement) {
        this.canvas = canvasElement;
        this.ctx = this.canvas.getContext('2d');
        
        // Configuración visual por defecto
        this.neuronRadius = 24;
        this.paddingX = 100; // Margen izquierdo y derecho interno
        this.paddingY = 65;  // Margen superior e inferior interno
        
        // Elemento seleccionado actualmente para resaltarlo visualmente
        this.selectedComponentId = null;

        // Ajustar el tamaño del canvas al iniciar
        this.resize();
    }

    /**
     * Ajusta el tamaño de dibujo del canvas al tamaño real que ocupa en la pantalla
     */
    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    /**
     * Limpia el lienzo por completo
     */
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    /**
     * Distribuye las capas en columnas y las neuronas en filas
     * @param {Network} network 
     */
    computePositions(network) {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const totalLayers = network.layers.length;

        if (totalLayers === 0) return;

        const availableWidth = width - (this.paddingX * 2);
        const layerSpacing = totalLayers > 1 ? availableWidth / (totalLayers - 1) : availableWidth;

        network.layers.forEach((layer, i) => {
            const layerX = this.paddingX + (i * layerSpacing);
            
            layer.headerBox = {
                x: layerX - 45,
                y: 15,
                w: 90,
                h: 24
            };

            const totalNeurons = layer.neurons.length;
            const availableHeight = height - (this.paddingY * 2);
            const neuronSpacing = totalNeurons > 1 ? availableHeight / (totalNeurons - 1) : availableHeight;

            const showAsBlock = (layer.layerType && layer.layerType !== 'linear') || totalNeurons > 10 || totalNeurons === 0;
            layer.showAsBlock = showAsBlock;
            if (showAsBlock) {
                // Capa como bloque: posicionar neuronas en el centro del bloque
                // para que las conexiones se dibujen desde el bloque
                layer.blockX = layerX;
                layer.blockY = height / 2 + 10;
                layer.neurons.forEach(neuron => {
                    neuron.x = layerX;
                    neuron.y = height / 2 + 10;
                });
            } else {
                layer.neurons.forEach((neuron, j) => {
                    neuron.x = layerX;
                    if (totalNeurons === 1) {
                        neuron.y = height / 2 + 10;
                    } else {
                        neuron.y = this.paddingY + (j * neuronSpacing) + 10;
                    }
                });
            }
        });
    }

    /**
     * Dibuja los botones interactivos de las capas en la parte superior del lienzo
     */
    drawLayerHeaders(network) {
        network.layers.forEach(layer => {
            const box = layer.headerBox;

            // Caja contenedora
            this.ctx.beginPath();
            this.ctx.roundRect(box.x, box.y, box.w, box.h, 6);
            
            if (this.selectedComponentId === layer.id) {
                this.ctx.fillStyle = 'rgba(99, 102, 241, 0.3)';
                this.ctx.strokeStyle = '#f59e0b';
                this.ctx.lineWidth = 2;
            } else {
                this.ctx.fillStyle = '#1e293b';
                this.ctx.strokeStyle = '#334155';
                this.ctx.lineWidth = 1;
            }
            this.ctx.fill();
            this.ctx.stroke();

            // Texto descriptivo
            this.ctx.fillStyle = '#cbd5e1';
            this.ctx.font = '10px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            
            let label = layer.type === 'input' ? 'Entrada' : layer.type === 'output' ? 'Salida' : 'Oculta';
            this.ctx.fillText(label, box.x + box.w / 2, box.y + box.h / 2);
        });
    }

    /**
     * RENDERIZAR CONEXIONES
     */
    drawConnections(network) {
        // Dibujar conexiones entre capas con neuronas (dense path)
        network.layers.forEach(layer => {
            if (layer.showAsBlock) return; // bloques se manejan abajo
            layer.neurons.forEach(neuron => {
                neuron.outputs.forEach(connection => {
                    const fromN = connection.from;
                    const toN = connection.to;

                    this.ctx.beginPath();
                    this.ctx.moveTo(fromN.x, fromN.y);
                    this.ctx.lineTo(toN.x, toN.y);

                    const maxThickness = 5;
                    this.ctx.lineWidth = Math.min(Math.abs(connection.weight) * 3.0, maxThickness) + 1.5;

                    if (connection.weight >= 0) {
                        this.ctx.strokeStyle = `rgba(99, 102, 241, ${Math.min(Math.abs(connection.weight) + 0.5, 1)})`;
                    } else {
                        this.ctx.strokeStyle = `rgba(239, 68, 68, ${Math.min(Math.abs(connection.weight) + 0.5, 1)})`;
                    }

                    if (this.selectedComponentId === connection.id) {
                        this.ctx.lineWidth += 3;
                        this.ctx.strokeStyle = '#f59e0b';
                    }

                    if (connection.isDropped) {
                        this.ctx.strokeStyle = 'rgba(51, 65, 85, 0.15)';
                        this.ctx.setLineDash([4, 4]);
                    } else {
                        this.ctx.setLineDash([]);
                    }

                    this.ctx.stroke();
                });
                this.ctx.setLineDash([]);
            });
        });

        // Dibujar conexiones entre bloques (capas sin neuronas) o bloque↔neurona
        for (let i = 0; i < network.layers.length - 1; i++) {
            const fromLayer = network.layers[i];
            const toLayer = network.layers[i + 1];
            const fromIsBlock = fromLayer.showAsBlock || (fromLayer.neurons.length === 0);
            const toIsBlock = toLayer.showAsBlock || (toLayer.neurons.length === 0);

            // Si fromLayer tiene neuronas y NO es bloque, las conexiones ya se dibujaron arriba
            if (fromLayer.neurons.length > 0 && !fromLayer.showAsBlock) continue;

            const fromX = fromIsBlock ? (fromLayer.blockX || fromLayer.neurons[0]?.x || 0) : null;
            const fromY = fromIsBlock ? (fromLayer.blockY || fromLayer.neurons[0]?.y || this.canvas.height/2) : null;
            const toX = toIsBlock ? (toLayer.blockX || toLayer.neurons[0]?.x || 0) : null;
            const toY = toIsBlock ? (toLayer.blockY || toLayer.neurons[0]?.y || this.canvas.height/2) : null;

            this.ctx.beginPath();
            this.ctx.setLineDash([]);
            this.ctx.strokeStyle = 'rgba(148, 163, 184, 0.7)';
            this.ctx.lineWidth = 2;

            if (fromIsBlock && toIsBlock) {
                this.ctx.moveTo(fromX + 20, fromY);
                this.ctx.lineTo(toX - 20, toY);
                this.ctx.stroke();
            } else if (fromIsBlock) {
                // bloque → cada neurona destino (una sola línea por neurona destino)
                toLayer.neurons.forEach(n => {
                    this.ctx.beginPath();
                    this.ctx.moveTo(fromX + 20, fromY);
                    this.ctx.lineTo(n.x - 8, n.y);
                    this.ctx.stroke();
                });
            } else {
                // cada neurona origen → bloque (una sola línea por neurona origen)
                fromLayer.neurons.forEach(n => {
                    this.ctx.beginPath();
                    this.ctx.moveTo(n.x + 8, n.y);
                    this.ctx.lineTo(toX - 20, toY);
                    this.ctx.stroke();
                });
            }
            this.ctx.setLineDash([]);
        }
    }

    /**
     * RENDERIZAR NEURONAS
     */
    drawNeurons(network) {
        network.layers.forEach(layer => {
            if (layer.showAsBlock) return;
            layer.neurons.forEach(neuron => {
                
                // --- Dibujo del fondo de la neurona ---
                this.ctx.beginPath();
                this.ctx.arc(neuron.x, neuron.y, this.neuronRadius, 0, 2 * Math.PI);
                
                if (neuron.isDropped) {
                    this.ctx.fillStyle = 'rgba(30, 41, 59, 0.2)';
                } else {
                    this.ctx.fillStyle = `rgba(16, 185, 129, ${Math.min(Math.max(neuron.value, 0), 1)})`;
                }
                this.ctx.fill();

                this.ctx.lineWidth = 2;

                if (this.selectedComponentId === neuron.id) {
                    this.ctx.strokeStyle = '#f59e0b'; 
                    this.ctx.lineWidth = 4;
                } else {
                    this.ctx.strokeStyle = '#475569';
                }
                this.ctx.stroke();

                this.ctx.fillStyle = '#ffffff';
                this.ctx.font = 'bold 10px monospace';
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                
                const valorTexto = neuron.value.toFixed(2);
                this.ctx.fillText(valorTexto, neuron.x, neuron.y);

                this.ctx.fillStyle = '#94a3b8';
                this.ctx.font = '9px sans-serif';

                if (layer.type === 'input') {
                    this.ctx.fillText('Entrada', neuron.x, neuron.y - (this.neuronRadius + 8));
                } else if (layer.type === 'output') {
                    this.ctx.fillText('Salida', neuron.x, neuron.y - (this.neuronRadius + 8));
                }
            });
        });
    }

    /**
     * Dibuja bloques para capas sin neuronas (conv/pool/flatten)
     */
    drawLayerBlocks(network) {
        network.layers.forEach(layer => {
            if (!layer.showAsBlock) return;
            const x = layer.blockX || 0;
            const y = layer.blockY || this.canvas.height / 2;
            const w = 40;
            const h = 60;

            let color = '#6366f1';
            if (layer.layerType === 'conv2d') color = '#f59e0b';
            else if (layer.layerType === 'maxpool2d') color = '#38bdf8';
            else if (layer.layerType === 'flatten') color = '#a78bfa';

            this.ctx.fillStyle = color + '33';
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;
            this.ctx.fillRect(x - w/2, y - h/2, w, h);
            this.ctx.strokeRect(x - w/2, y - h/2, w, h);

            this.ctx.fillStyle = '#e2e8f0';
            this.ctx.font = 'bold 9px sans-serif';
            this.ctx.textAlign = 'center';
            let label = (layer.layerType || 'linear').toUpperCase();
            if (layer.layerType === 'linear' && layer.neurons && layer.neurons.length > 10) {
                label += ` (${layer.neurons.length})`;
            }
            this.ctx.fillText(label, x, y - h/2 - 6);

            this.ctx.fillStyle = '#94a3b8';
            this.ctx.font = '8px monospace';
            const shape = layer.cache && layer.cache.outputShape ? layer.cache.outputShape.join('×') : '';
            if (shape) this.ctx.fillText(shape, x, y + h/2 + 12);
        });
    }

    /**
     * METODO MAESTRO: Coordina todo el ciclo de dibujo
     * @param {Network} network 
     */
    render(network) {
        this.clear();
        this.ctx.setLineDash([]);
        this.computePositions(network);
        this.drawConnections(network);
        this.ctx.setLineDash([]);
        this.drawLayerHeaders(network);
        this.drawLayerBlocks(network);
        this.drawNeurons(network);
    }
}