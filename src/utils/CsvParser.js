export class CsvParser {
    constructor() {
        this.rawLines = [];
        this.headers = [];
    }

    /**
     * Lee el archivo CSV crudo y extrae las filas limpias
     * @param {File} file - El objeto de archivo obtenido del input/dropzone
     * @returns {Promise<Object>} Resumen estadístico inicial para la UI
     */
    async scanFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = (e) => {
                const text = e.target.result;
                this.rawLines = text.split(/\r?\n/).map(line => line.trim()).filter(line => line.length > 0);

                if (this.rawLines.length < 2) {
                    return reject(new Error('El archivo CSV debe contener al menos una cabecera y una fila de datos.'));
                }

                const firstLine = this.rawLines[0];
                const separator = firstLine.includes(';') ? ';' : ',';

                this.headers = firstLine.split(separator).map(h => h.trim().replace(/^["']|["']$/g, ''));

                const diagnostics = this.runFullExploratoryAnalysis(separator);
                resolve(diagnostics);
            };

            reader.onerror = () => reject(new Error('Error al leer el archivo físico.'));
            reader.readAsText(file);
        });
    }

    /**
     * Escanea el 100% de las celdas del archivo para validar tipos y extraer rangos
     */
    runFullExploratoryAnalysis(separator) {
        const report = {};
        
        this.headers.forEach(header => {
            report[header] = {
                name: header,
                totalRows: this.rawLines.length - 1,
                isFullyNumeric: true,
                min: Infinity,
                max: -Infinity,
                uniqueValues: new Set(),
                valueCounts: {}, 
                frequencies: {},
                sampleRows: []
            };
        });

        for (let i = 1; i < this.rawLines.length; i++) {
            const cells = this.rawLines[i].split(separator).map(c => c.trim().replace(/^["']|["']$/g, ''));
            
            if (cells.length !== this.headers.length) continue;

            this.headers.forEach((header, colIdx) => {
                const cellValue = cells[colIdx];
                const colReport = report[header];

                if (i <= 5) colReport.sampleRows.push(cellValue);

                if (cellValue !== '') {
                    colReport.uniqueValues.add(cellValue);
                    colReport.valueCounts[cellValue] = (colReport.valueCounts[cellValue] || 0) + 1;
                }

                const numericValue = Number(cellValue);
                
                if (cellValue === '' || isNaN(numericValue)) {
                    colReport.isFullyNumeric = false;
                } else {
                    if (numericValue < colReport.min) colReport.min = numericValue;
                    if (numericValue > colReport.max) colReport.max = numericValue;
                }
            });
        }

        this.headers.forEach(header => {
            const colReport = report[header];
            colReport.uniqueValues = Array.from(colReport.uniqueValues);
            
            if (!colReport.isFullyNumeric) {
                colReport.uniqueValues.forEach(val => {
                    const count = colReport.valueCounts[val] || 0;
                    colReport.frequencies[val] = count / colReport.totalRows;
                });
            }
            delete colReport.valueCounts;

            if (!colReport.isFullyNumeric || colReport.min === Infinity) {
                colReport.min = null;
                colReport.max = null;
            }
        });

        return report;
    }

    /**
     * Transforma todo el CSV en un array de tensores/arrays nativos normalizados
     * según la configuración manual recolectada de la UI.
     * @param {Object} diagnostics - El objeto con los selectores del DOM incrustados
     * @returns {Array<Object>} Dataset listo para entrenamiento: [{ inputs: [], targets: [] }]
     */
    compileDataset(diagnostics) {
        const separator = this.rawLines[0].includes(';') ? ';' : ',';
        const compiledData = [];

        for (let i = 1; i < this.rawLines.length; i++) {
            const cells = this.rawLines[i].split(separator).map(c => c.trim().replace(/^["']|["']$/g, ''));

            if (cells.length !== this.headers.length) continue;

            const rowInputs = [];
            const rowTargets = [];

            this.headers.forEach((header, colIdx) => {
                const cellValue = cells[colIdx];
                const col = diagnostics[header];
                
                const role = col.role;
                if (role === 'ignore') return;

                let processedValues = [];

                //LA COLUMNA ES NUMÉRICA
                if (col.isFullyNumeric) {
                    const num = parseFloat(cellValue);
                    const normType = col.normalization;
                    
                    const range = col.max - col.min;
                    if (range === 0) {
                        processedValues.push(0);
                    } else if (normType === 'minmax_0_1') {
                        processedValues.push((num - col.min) / range);
                    } else if (normType === 'minmax_1_1') {
                        processedValues.push(2 * ((num - col.min) / range) - 1);
                    } else {
                        processedValues.push(num);
                    }
                } 
                // LA COLUMNA ES BINARIA
                else if (col.uniqueValues.length === 2) {

                    const numericRepresentation = (cellValue === col.uniqueValues[0]) ? 0 : 1;
                    processedValues.push(numericRepresentation);
                } 
                // LA COLUMNA ES CATEGÓRICA MULTICLASE
                else {
                    const oneHotVector = new Array(col.uniqueValues.length).fill(0);
                    const categoryIdx = col.uniqueValues.indexOf(cellValue);
                    if (categoryIdx !== -1) {
                        oneHotVector[categoryIdx] = 1;
                    }
                    processedValues = oneHotVector;
                }

                if (role === 'input') {
                    rowInputs.push(...processedValues);
                } else if (role === 'output') {
                    rowTargets.push(...processedValues);
                }
            });

            compiledData.push({
                inputs: rowInputs,
                targets: rowTargets
            });
        }

        return compiledData;
    }
}