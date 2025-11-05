// API Client for STMGT Traffic Forecast
// Handles all communication with FastAPI backend

// API Configuration
const API_BASE_URL = window.location.origin; // Same origin for simplicity

// API Key (for future authentication)
// const API_KEY = 'your-api-key-here';

/**
 * Generic API call wrapper with error handling
 * @param {string} endpoint - API endpoint path
 * @param {object} options - Fetch options
 * @returns {Promise} Response JSON
 */
async function apiCall(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                // 'X-API-Key': API_KEY,  // Uncomment when auth is enabled
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            if (response.status === 401) {
                throw new Error('Authentication required. Please provide valid API key.');
            } else if (response.status === 404) {
                throw new Error('Resource not found.');
            } else if (response.status === 500) {
                throw new Error('Server error. Please try again later.');
            } else {
                throw new Error(`API call failed: ${response.status} ${response.statusText}`);
            }
        }
        
        return await response.json();
        
    } catch (error) {
        console.error(`API Error [${endpoint}]:`, error);
        throw error;
    }
}

/**
 * Get all traffic nodes
 * @returns {Promise} Array of node objects
 */
async function apiGetNodes() {
    const data = await apiCall('/nodes');
    return data;
}

/**
 * Get predictions for all nodes
 * @param {Array} horizons - Forecast horizons (e.g., [1, 4, 8])
 * @returns {Promise} Prediction object
 */
async function apiGetPredictions(horizons = [1, 4, 8]) {
    const data = await apiCall('/predict', {
        method: 'POST',
        body: JSON.stringify({ horizons })
    });
    return data;
}

/**
 * Get prediction for specific node
 * @param {number} nodeId - Node ID
 * @param {Array} horizons - Forecast horizons (max 8 steps)
 * @returns {Promise} Prediction object for node
 */
async function apiGetNodePrediction(nodeId, horizons = [1, 2, 3, 4, 6, 8]) {
    const data = await apiCall(`/predict`, {
        method: 'POST',
        body: JSON.stringify({ 
            node_ids: [nodeId],
            horizons 
        })
    });
    
    // Extract prediction for this node
    if (data.nodes && data.nodes.length > 0) {
        const nodePred = data.nodes.find(p => p.node_id === nodeId);
        if (nodePred) {
            return {
                ...nodePred,
                inference_time_ms: data.inference_time_ms,
                timestamp: data.timestamp,
                forecast_time: data.forecast_time
            };
        }
    }
    
    throw new Error(`No prediction found for node ${nodeId}`);
}

/**
 * Get specific node metadata
 * @param {number} nodeId - Node ID
 * @returns {Promise} Node metadata
 */
async function apiGetNode(nodeId) {
    const data = await apiCall(`/nodes/${nodeId}`);
    return data;
}

/**
 * Health check
 * @returns {Promise} Health status
 */
async function apiHealthCheck() {
    const data = await apiCall('/health');
    return data;
}

/**
 * Get cache statistics (if admin endpoint available)
 * @returns {Promise} Cache stats
 */
async function apiGetCacheStats() {
    try {
        const data = await apiCall('/admin/cache/stats');
        return data;
    } catch (error) {
        // Admin endpoint might not be available
        console.warn('Cache stats not available');
        return null;
    }
}

// Export functions for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        apiGetNodes,
        apiGetPredictions,
        apiGetNodePrediction,
        apiGetNode,
        apiHealthCheck,
        apiGetCacheStats
    };
}
