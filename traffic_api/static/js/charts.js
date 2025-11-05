// Chart.js visualization for traffic forecasts

let forecastChart = null;

/**
 * Initialize or update forecast chart with new data
 * @param {object} prediction - Prediction object with mean, std, horizons
 */
function updateForecastChart(prediction) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    // Prepare data from prediction
    const predictions = prediction.predictions || prediction;
    
    // Generate labels (time ahead)
    const labels = predictions.map((_, idx) => {
        const minutes = (idx + 1) * 15; // 15-min intervals
        const hours = Math.floor(minutes / 60);
        const mins = minutes % 60;
        
        if (hours === 0) {
            return `${mins}min`;
        } else if (mins === 0) {
            return `${hours}h`;
        } else {
            return `${hours}h${mins}`;
        }
    });
    
    // Extract mean predictions and confidence intervals
    const means = predictions.map(p => p.mean);
    const upperBounds = predictions.map(p => p.mean + p.std);
    const lowerBounds = predictions.map(p => Math.max(0, p.mean - p.std)); // Don't go negative
    
    // Destroy existing chart if it exists
    if (forecastChart) {
        forecastChart.destroy();
    }
    
    // Create new chart
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Predicted Speed',
                    data: means,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: false,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: '#3b82f6',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                },
                {
                    label: 'Upper Bound (80% CI)',
                    data: upperBounds,
                    borderColor: '#94a3b8',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: '+1',
                    backgroundColor: 'rgba(148, 163, 184, 0.1)',
                    pointRadius: 0,
                    tension: 0.4
                },
                {
                    label: 'Lower Bound (80% CI)',
                    data: lowerBounds,
                    borderColor: '#94a3b8',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '3-Hour Traffic Speed Forecast',
                    font: {
                        size: 14,
                        weight: 'bold'
                    }
                },
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 10,
                        font: {
                            size: 11
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(1) + ' km/h';
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Speed (km/h)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + ' km/h';
                        }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time Ahead',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
}

/**
 * Show loading state on chart
 */
function showChartLoading() {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    if (forecastChart) {
        forecastChart.destroy();
    }
    
    // Draw loading message
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.font = '14px Arial';
    ctx.fillStyle = '#6c757d';
    ctx.textAlign = 'center';
    ctx.fillText('Loading forecast...', ctx.canvas.width / 2, ctx.canvas.height / 2);
}

/**
 * Show error on chart
 * @param {string} message - Error message
 */
function showChartError(message) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    if (forecastChart) {
        forecastChart.destroy();
    }
    
    // Draw error message
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.font = '14px Arial';
    ctx.fillStyle = '#dc3545';
    ctx.textAlign = 'center';
    ctx.fillText('Error: ' + message, ctx.canvas.width / 2, ctx.canvas.height / 2);
}

// Export functions
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        updateForecastChart,
        showChartLoading,
        showChartError
    };
}
