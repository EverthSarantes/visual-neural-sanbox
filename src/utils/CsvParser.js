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
            report[header].uniqueValues = Array.from(report[header].uniqueValues);
            
            if (!report[header].isFullyNumeric || report[header].min === Infinity) {
                report[header].min = null;
                report[header].max = null;
            }
        });

        return report;
    }
}