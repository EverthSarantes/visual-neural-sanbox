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

            layer.neurons.forEach((neuron, j) => {
                neuron.x = layerX;
                if (totalNeurons === 1) {
                    neuron.y = height / 2 + 10;
                } else {
                    neuron.y = this.paddingY + (j * neuronSpacing) + 10;
                }
            });
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
        network.layers.forEach(layer => {
            layer.neurons.forEach(neuron => {
                neuron.outputs.forEach(connection => {
                    const fromN = connection.from;
                    const toN = connection.to;

                    this.ctx.beginPath();
                    this.ctx.moveTo(fromN.x, fromN.y);
                    this.ctx.lineTo(toN.x, toN.y);

                    const maxThickness = 5;
                    this.ctx.lineWidth = Math.min(Math.abs(connection.weight) * 2.5, maxThickness) + 0.5;

                    if (connection.weight >= 0) {
                        this.ctx.strokeStyle = `rgba(99, 102, 241, ${Math.min(Math.abs(connection.weight) + 0.2, 1)})`;
                    } else {
                        this.ctx.strokeStyle = `rgba(239, 68, 68, ${Math.min(Math.abs(connection.weight) + 0.2, 1)})`;
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
    }

    /**
     * RENDERIZAR NEURONAS
     */
    drawNeurons(network) {
        network.layers.forEach(layer => {
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
     * METODO MAESTRO: Coordina todo el ciclo de dibujo
     * @param {Network} network 
     */
    render(network) {
        this.clear();
        this.computePositions(network); // Sincroniza geometría por si cambió el tamaño de pantalla
        this.drawConnections(network);  // Capa inferior
        this.drawLayerHeaders(network); // Capa intermedia (headers)
        this.drawNeurons(network);      // Capa superior
    }
}