export class MetricsChart {
    constructor() {
        this.lossChart = null;
        this.accuracyChart = null;
        
        this.initCharts();
    }

    /**
     * Inicializa las dos gráficas de Chart.js con estilos personalizados para el tema oscuro
     */
    initCharts() {
        const ctxLoss = document.getElementById('chart-loss');
        const ctxAcc = document.getElementById('chart-accuracy');

        if (!ctxLoss || !ctxAcc) {
            console.warn('No se encontraron los elementos canvas para las gráficas en el HTML.');
            return;
        }

        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    grid: { color: 'rgba(51, 65, 85, 0.1)' },
                    ticks: { color: '#64748b', font: { size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(51, 65, 85, 0.1)' },
                    ticks: { color: '#64748b', font: { size: 10 } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        };

        // Gráfica de Pérdida
        this.lossChart = new Chart(ctxLoss, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: '#f43f5e',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }]
            },
            options: commonOptions
        });

        // Gráfica de Precisión (Accuracy Chart)
        this.accuracyChart = new Chart(ctxAcc, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: '#10b981',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }]
            },
            options: commonOptions
        });
    }

    /**
     * Inyecta un nuevo punto de datos en ambas curvas y fuerza el redibujado
     */
    pushMetrics(epoch, loss, accuracy) {
        if (!this.lossChart || !this.accuracyChart) return;

        this.lossChart.data.labels.push(epoch);
        this.lossChart.data.datasets[0].data.push(loss);

        this.accuracyChart.data.labels.push(epoch);
        this.accuracyChart.data.datasets[0].data.push(accuracy);

        const maxPuntosVisibles = 300;
        
        if (this.lossChart.data.labels.length > maxPuntosVisibles) {
            this.lossChart.data.labels.shift();
            this.lossChart.data.datasets[0].data.shift();

            this.accuracyChart.data.labels.shift();
            this.accuracyChart.data.datasets[0].data.shift();
        }

        this.lossChart.update();
        this.accuracyChart.update();
    }

    /**
     * Limpia por completo el historial de las curvas (Para el botón Reset)
     */
    clear() {
        if (!this.lossChart || !this.accuracyChart) return;

        this.lossChart.data.labels = [];
        this.lossChart.data.datasets[0].data = [];
        
        this.accuracyChart.data.labels = [];
        this.accuracyChart.data.datasets[0].data = [];

        this.lossChart.update();
        this.accuracyChart.update();
    }
}