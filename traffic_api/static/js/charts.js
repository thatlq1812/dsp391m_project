// Chart.js visualization for traffic forecasts

let forecastChart = null;

/**
 * Initialize or update forecast chart with new data
 * @param {object} prediction - Prediction object with mean, std, horizons
 */
function updateForecastChart(prediction) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    // Prepare data from prediction
    // Backend returns: { node_id, lat, lon, forecasts: [...], ... }
    const forecasts = prediction.forecasts || prediction.predictions || prediction;
    
    if (!Array.isArray(forecasts)) {
        console.error('Invalid forecast data:', prediction);
        showChartError('Invalid forecast data format');
        return;
    }
    
    // Generate labels (time ahead)
    const labels = forecasts.map((f, idx) => {
        // Use horizon_minutes if available, otherwise calculate
        const minutes = f.horizon_minutes || (f.horizon * 15);
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
    const means = forecasts.map(f => f.mean);
    const upperBounds = forecasts.map(f => f.upper_80 || (f.mean + f.std));
    const lowerBounds = forecasts.map(f => f.lower_80 || Math.max(0, f.mean - f.std));
    
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
