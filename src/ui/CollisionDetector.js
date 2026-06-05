export class CollisionDetector {
    /**
     * Busca si el clic del usuario impactó una neurona (prioridad) o una conexión.
     * @param {number} clickX - Coordenada X del clic relativo al canvas
     * @param {number} clickY - Coordenada Y del clic relativo al canvas
     * @param {Network} network - El objeto de la red con posiciones calculadas
     * @param {number} neuronRadius - El radio de las neuronas usado en el renderizador
     * @returns {Object|null} Devuelve { type: 'neuron'|'connection', object: Componente } o null
     */
    static findComponentAt(clickX, clickY, network, neuronRadius) {
        // Verificar colisión con los Botones de Cabecera de Capa
        for (const layer of network.layers) {
            if (layer.headerBox) {
                const box = layer.headerBox;
                if (clickX >= box.x && clickX <= box.x + box.w &&
                    clickY >= box.y && clickY <= box.y + box.h) {
                    return { type: 'layer', object: layer };
                }
            }
        }

        // Verificar colisión con Neuronas
        for (const layer of network.layers) {
            for (const neuron of layer.neurons) {
                const dx = clickX - neuron.x;
                const dy = clickY - neuron.y;
                if (Math.sqrt(dx * dx + dy * dy) <= neuronRadius) {
                    return { type: 'neuron', object: neuron };
                }
            }
        }

        // Verificar colisión con Conexiones
        const lineThreshold = 6;
        for (const layer of network.layers) {
            for (const neuron of layer.neurons) {
                for (const connection of neuron.outputs) {
                    const x1 = connection.from.x;
                    const y1 = connection.from.y;
                    const x2 = connection.to.x;
                    const y2 = connection.to.y;

                    if (this.isPointNearSegment(clickX, clickY, x1, y1, x2, y2, lineThreshold)) {
                        return { type: 'connection', object: connection };
                    }
                }
            }
        }

        return null;
    }

    /**
     * Calcula si un punto está cerca de un segmento de línea limitado
     */
    static isPointNearSegment(px, py, x1, y1, x2, y2, threshold) {
        const A = px - x1;
        const B = py - y1;
        const C = x2 - x1;
        const D = y2 - y1;

        const dot = A * C + B * D;
        const lenSq = C * C + D * D;
        
        let param = -1;
        if (lenSq !== 0) param = dot / lenSq;

        let closestX, closestY;

        if (param < 0) {
            closestX = x1;
            closestY = y1;
        } else if (param > 1) {
            closestX = x2;
            closestY = y2;
        } else {
            closestX = x1 + param * C;
            closestY = y1 + param * D;
        }

        const dx = px - closestX;
        const dy = py - closestY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        return distance <= threshold;
    }
}