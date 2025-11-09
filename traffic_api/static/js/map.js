// Google Maps integration for traffic visualization

// Global variables
let map;
let markers = [];
let nodes = [];
let selectedNode = null;
let infoWindow;

// HCMC coordinates
const HCMC_CENTER = { lat: 10.772465, lng: 106.697794 };

/**
 * Get marker color based on speed
 * @param {number} speed - Speed in km/h
 * @returns {string} Color hex code
 */
function getMarkerColor(speed) {
    if (speed > 40) return '#28a745'; // Green - fast
    if (speed > 20) return '#ffc107'; // Yellow - moderate
    return '#dc3545'; // Red - congested
}

/**
 * Initialize Google Map (called by Google Maps API)
 */
async function initMap() {
    // Hide loading overlay once map starts loading
    const loadingOverlay = document.getElementById('loading-overlay');
    
    try {
        // Create map
        map = new google.maps.Map(document.getElementById('map'), {
            center: HCMC_CENTER,
            zoom: 13,
            mapTypeControl: true,
            mapTypeControlOptions: {
                style: google.maps.MapTypeControlStyle.DROPDOWN_MENU,
                position: google.maps.ControlPosition.TOP_RIGHT
            },
            streetViewControl: false,
            fullscreenControl: true,
            styles: [
                {
                    featureType: 'poi',
                    stylers: [{ visibility: 'off' }] // Hide points of interest for cleaner view
                }
            ]
        });
        
        // Create info window for node details
        infoWindow = new google.maps.InfoWindow();
        
        // Load nodes from API
        await loadNodes();
        
        // Initial prediction update
        await updatePredictions();
        
        // Hide loading overlay
        loadingOverlay.classList.add('hidden');
        
        // Auto-refresh every 15 minutes
        setInterval(updatePredictions, 15 * 60 * 1000);
        
        console.log('Map initialized successfully');
        
    } catch (error) {
        console.error('Map initialization failed:', error);
        loadingOverlay.innerHTML = `
            <div class="error-message">
                <h4>Failed to load map</h4>
                <p>${error.message}</p>
                <button class="btn btn-primary" onclick="location.reload()">Retry</button>
            </div>
        `;
    }
}

/**
 * Load traffic nodes from API
 */
async function loadNodes() {
    try {
        const data = await apiGetNodes();
        nodes = data.nodes || data;
        
        // Add 'id' alias for 'node_id' for easier access
        nodes = nodes.map(node => ({
            ...node,
            id: node.node_id
        }));
        
        console.log(`Loaded ${nodes.length} traffic nodes`);
        
        // Update stats
        document.getElementById('totalNodes').textContent = nodes.length;
        
        // Create markers for each node
        nodes.forEach((node, index) => {
            const marker = new google.maps.Marker({
                position: { lat: node.lat, lng: node.lon },
                map: map,
                title: `Node ${node.id}`,
                icon: {
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: 8,
                    fillColor: '#3b82f6',
                    fillOpacity: 0.8,
                    strokeColor: '#ffffff',
                    strokeWeight: 2
                }
            });
            
            // Add click listener
            marker.addListener('click', () => selectNode(node, marker));
            
            // Store marker with node reference
            markers.push({ node: node, marker: marker });
        });
        
    } catch (error) {
        console.error('Failed to load nodes:', error);
        alert('Failed to load traffic nodes. Please check API connection.');
        throw error;
    }
}

/**
 * Update predictions for all nodes
 */
async function updatePredictions() {
    try {
        console.log('Updating predictions...');
        
        // Get predictions for first timestep (current + 15min)
        const data = await apiGetPredictions([1]);
        
        const predictions = data.predictions || [];
        
        // Update marker colors based on current predicted speed
        markers.forEach(({ node, marker }) => {
            const pred = predictions.find(p => p.node_id === node.id);
            if (pred && pred.predictions && pred.predictions.length > 0) {
                const currentSpeed = pred.predictions[0].mean;
                
                marker.setIcon({
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: 8,
                    fillColor: getMarkerColor(currentSpeed),
                    fillOpacity: 0.8,
                    strokeColor: '#ffffff',
                    strokeWeight: 2
                });
            }
        });
        
        // Update last update time
        const now = new Date();
        document.getElementById('lastUpdate').textContent = 
            `Last update: ${now.toLocaleTimeString()}`;
        
        console.log('Predictions updated');
        
    } catch (error) {
        console.error('Failed to update predictions:', error);
        document.getElementById('lastUpdate').textContent = 
            `Update failed: ${error.message}`;
    }
}

/**
 * Select a node and show its forecast
 * @param {object} node - Node object
 * @param {object} marker - Google Maps marker
 */
async function selectNode(node, marker) {
    selectedNode = node;
    
    console.log(`Selected node ${node.id}`);
    
    // Highlight selected marker
    markers.forEach(m => {
        m.marker.setIcon({
            path: google.maps.SymbolPath.CIRCLE,
            scale: m.marker === marker ? 12 : 8, // Enlarge selected
            fillColor: m.marker.getIcon().fillColor,
            fillOpacity: m.marker === marker ? 1.0 : 0.8,
            strokeColor: '#ffffff',
            strokeWeight: m.marker === marker ? 3 : 2
        });
    });
    
    // Update panel title
    document.getElementById('nodeTitle').textContent = `Node ${node.id} Forecast`;
    
    // Update node details
    const detailsDiv = document.getElementById('nodeDetails');
    detailsDiv.innerHTML = `
        <p><strong>Location:</strong> ${node.lat.toFixed(6)}, ${node.lon.toFixed(6)}</p>
        <p><strong>Streets:</strong> ${node.street_names?.join(', ') || 'N/A'}</p>
        <p><strong>Road Types:</strong> ${node.road_types?.join(', ') || 'N/A'}</p>
    `;
    
    // Show loading on chart
    showChartLoading();
    
    try {
        // Fetch detailed forecast for this node
        const prediction = await apiGetNodePrediction(node.id);
        
        console.log('Prediction received:', prediction);
        
        // Update chart
        updateForecastChart(prediction);
        
        // Update inference time
        document.getElementById('inferenceTime').textContent = 
            prediction.inference_time_ms?.toFixed(1) || '--';
        
        // Update cache status
        const cacheStatus = document.getElementById('cacheStatus');
        if (prediction.cache_hit) {
            cacheStatus.textContent = 'HIT';
            cacheStatus.className = 'stat-value cache-hit';
        } else {
            cacheStatus.textContent = 'MISS';
            cacheStatus.className = 'stat-value cache-miss';
        }
        
    } catch (error) {
        console.error('Failed to get node prediction:', error);
        showChartError(error.message);
        
        detailsDiv.innerHTML += `
            <div class="error-message mt-2">
                <small>Failed to load forecast: ${error.message}</small>
            </div>
        `;
    }
}

/**
 * Center map on HCMC
 */
function centerMap() {
    if (map) {
        map.setCenter(HCMC_CENTER);
        map.setZoom(13);
    }
}

// Make initMap available globally for Google Maps callback
window.initMap = initMap;
window.centerMap = centerMap;
