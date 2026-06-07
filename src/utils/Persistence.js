/**
 * Exporta el estado absoluto de la red, configuraciones y datasets a un archivo JSON.
 */
export function exportModel(network, uiController, version = "1.0.0") {
    const cleanDiagnostics = {};
    if (uiController.datasetDiagnostics) {
        for (const [key, col] of Object.entries(uiController.datasetDiagnostics)) {
            cleanDiagnostics[key] = {
                name: col.name,
                totalRows: col.totalRows,
                isFullyNumeric: col.isFullyNumeric,
                min: col.min,
                max: col.max,
                uniqueValues: col.uniqueValues,
                sampleRows: col.sampleRows,
                role: col.domRoleSelect ? col.domRoleSelect.value : col.role,
                normalization: col.domContextualElement ? col.domContextualElement.value : col.normalization
            };
        }
    }

    const modelState = {
        version: version,
        timestamp: new Date().toISOString(),
        
        config: {
            learningRate: network.learningRate,
            momentum: network.momentum,
            dropoutRate: network.dropoutRate,
            weightDecay: network.weightDecay,
            batchSize: network.batchSize,
            lossTypeId: network.lossTypeId,
            initializationStrategyId: network.initializationStrategyId,
            epoch: network.epoch,
            currentLoss: network.currentLoss,
            currentAccuracy: network.currentAccuracy,
            datasetDiagnostics: cleanDiagnostics,
        },

        datasets: {
            trainSet: uiController.trainSet || [],
            testSet: uiController.testSet || []
        },

        topology: network.layers.map((layer, lIdx) => ({
            id: layer.id,
            layerIndex: lIdx,
            type: layer.type,
            activationId: layer.activationId,
            connectionTypeId: layer.connectionTypeId,
            neuronSeed: layer.neuronSeed,
            neurons: layer.neurons.map((neuron, nIdx) => ({
                neuronIndex: nIdx,
                id: neuron.id,
                activationId: neuron.activationId,
                bias: neuron.bias,
                value: neuron.value,
                netInput: neuron.netInput,
                errorSignal: neuron.errorSignal,
                biasVelocity: neuron.biasVelocity,
                isDropped: neuron.isDropped,

                inputs: neuron.inputs.map(conn => ({
                    id: conn.id,
                    weight: conn.weight,
                    velocity: conn.velocity,
                    isDropped: conn.isDropped,
                    fromNeuronId: conn.from.id
                }))
            }))
        }))
    };

    // Usamos un Blob en lugar de un DataURI crudo para soportar datasets grandes de forma robusta
    const blob = new Blob([JSON.stringify(modelState, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", url);
    downloadAnchor.setAttribute("download", `modelo_epoca_${network.epoch}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
    URL.revokeObjectURL(url); // Liberar memoria
}
