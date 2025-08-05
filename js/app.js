/**
 * Main Application Entry Point
 * Initializes the RNN model, 3D visualization, and UI controller
 */

// Global variables
window.rnnModel = null;
window.visualization = null;
window.uiController = null;

/**
 * Initialize the application
 */
function initializeApp() {
    try {
        console.log('Initializing RNN 3D Visualization App...');
        
        // Initialize RNN Model
        window.rnnModel = new RNNModel(64, 1000, 50);
        console.log('✓ RNN Model initialized');
        
        // Initialize 3D Visualization
        const container = document.getElementById('three-container');
        if (!container) {
            throw new Error('3D container not found');
        }
        
        window.visualization = new Visualization3D(container);
        console.log('✓ 3D Visualization initialized');
        
        // Initialize UI Controller
        window.uiController = new UIController();
        console.log('✓ UI Controller initialized');
        
        // Auto-select and load Amazon model for better predictions
        setTimeout(async () => {
            try {
                console.log('Auto-loading Amazon model for better predictions...');
                const amazonRadio = document.getElementById('amazon-model');
                if (amazonRadio) {
                    amazonRadio.checked = true;
                    await window.uiController.loadAmazonModel();
                    console.log('✓ Amazon model auto-loaded');
                }
            } catch (error) {
                console.warn('Amazon model auto-load failed, using demo model:', error);
                window.uiController.updateModelStatus('Using enhanced demo model', 'success');
            }
        }, 1000);
        
        // Setup demo sentences
        setupDemoSentences();
        
        console.log('🚀 App fully initialized and ready!');
        
        // Show welcome message
        showWelcomeMessage();
        
    } catch (error) {
        console.error('❌ Failed to initialize app:', error);
        showInitializationError(error.message);
    }
}

/**
 * Setup demo sentences for quick testing
 */
function setupDemoSentences() {
    const demoSentences = [
        "I love this movie!",
        "This film is terrible",
        "The movie was okay",
        "Amazing performance by the actors",
        "Worst movie I've ever seen",
        "Not bad, could be better"
    ];
    
    // Add demo sentence buttons
    const controlPanel = document.querySelector('.control-panel');
    const demoContainer = document.createElement('div');
    demoContainer.className = 'demo-sentences';
    demoContainer.innerHTML = `
        <label>Quick Demo:</label>
        <div class="demo-buttons">
            ${demoSentences.map(sentence => 
                `<button class="demo-btn" data-sentence="${sentence}">${sentence}</button>`
            ).join('')}
        </div>
    `;
    
    // Add CSS for demo buttons
    const style = document.createElement('style');
    style.textContent = `
        .demo-sentences {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        .demo-sentences label {
            font-weight: bold;
            min-width: max-content;
        }
        .demo-buttons {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }
        .demo-btn {
            padding: 5px 10px;
            font-size: 12px;
            background: linear-gradient(45deg, #74b9ff, #0984e3);
            border: none;
            border-radius: 15px;
            color: white;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .demo-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        }
    `;
    document.head.appendChild(style);
    
    controlPanel.appendChild(demoContainer);
    
    // Add event listeners for demo buttons
    demoContainer.addEventListener('click', (e) => {
        if (e.target.classList.contains('demo-btn')) {
            const sentence = e.target.getAttribute('data-sentence');
            document.getElementById('sentence-input').value = sentence;
            
            // Auto-process if not currently processing
            if (window.uiController && !window.uiController.isProcessing) {
                window.uiController.processFullSentence();
            }
        }
    });
}

/**
 * Show welcome message
 */
function showWelcomeMessage() {
    const notification = document.createElement('div');
    notification.className = 'welcome-notification';
    notification.innerHTML = `
        <div class="welcome-content">
            <h3>🎉 Welcome to 3D RNN Visualizer!</h3>
            <p>Enter a sentence or try a demo to see how RNNs process language for sentiment analysis.</p>
            <p><strong>Tip:</strong> Use the demo buttons for quick examples!</p>
        </div>
    `;
    
    // Add CSS for welcome notification
    const style = document.createElement('style');
    style.textContent = `
        .welcome-notification {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 30px;
            border-radius: 15px;
            border: 2px solid #00f5ff;
            box-shadow: 0 10px 30px rgba(0, 245, 255, 0.3);
            z-index: 10000;
            max-width: 400px;
            text-align: center;
            backdrop-filter: blur(10px);
            animation: welcomeSlideIn 0.5s ease-out;
        }
        .welcome-content h3 {
            color: #00f5ff;
            margin-bottom: 15px;
        }
        .welcome-content p {
            margin-bottom: 10px;
            line-height: 1.5;
        }
        @keyframes welcomeSlideIn {
            from {
                opacity: 0;
                transform: translate(-50%, -50%) scale(0.8);
            }
            to {
                opacity: 1;
                transform: translate(-50%, -50%) scale(1);
            }
        }
        @keyframes welcomeSlideOut {
            from {
                opacity: 1;
                transform: translate(-50%, -50%) scale(1);
            }
            to {
                opacity: 0;
                transform: translate(-50%, -50%) scale(0.8);
            }
        }
    `;
    document.head.appendChild(style);
    
    document.body.appendChild(notification);
    
    // Auto-hide after 4 seconds or on click
    const hideNotification = () => {
        notification.style.animation = 'welcomeSlideOut 0.3s ease-in';
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 300);
    };
    
    setTimeout(hideNotification, 4000);
    notification.addEventListener('click', hideNotification);
}

/**
 * Show initialization error
 */
function showInitializationError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'init-error';
    errorDiv.innerHTML = `
        <div class="error-content">
            <h3>❌ Initialization Error</h3>
            <p>${message}</p>
            <p>Please refresh the page and try again.</p>
            <button onclick="location.reload()">🔄 Refresh Page</button>
        </div>
    `;
    
    // Add CSS for error notification
    const style = document.createElement('style');
    style.textContent = `
        .init-error {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 0, 0, 0.9);
            color: white;
            padding: 30px;
            border-radius: 15px;
            border: 2px solid #ff6b6b;
            box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
            z-index: 10000;
            max-width: 400px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .error-content h3 {
            color: #ffcccb;
            margin-bottom: 15px;
        }
        .error-content p {
            margin-bottom: 10px;
            line-height: 1.5;
        }
        .error-content button {
            margin-top: 15px;
            padding: 10px 20px;
            background: #fff;
            color: #ff6b6b;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
        }
    `;
    document.head.appendChild(style);
    
    document.body.appendChild(errorDiv);
}

/**
 * Handle application errors
 */
window.addEventListener('error', (event) => {
    console.error('Application Error:', event.error);
    
    // Show user-friendly error message
    const errorNotification = document.createElement('div');
    errorNotification.className = 'app-error';
    errorNotification.innerHTML = `
        <div class="error-content">
            <h4>⚠️ Something went wrong</h4>
            <p>An unexpected error occurred. Please try refreshing the page.</p>
            <button onclick="this.parentElement.parentElement.remove()">Dismiss</button>
        </div>
    `;
    
    errorNotification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(45deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        z-index: 10000;
        max-width: 300px;
    `;
    
    document.body.appendChild(errorNotification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (document.body.contains(errorNotification)) {
            document.body.removeChild(errorNotification);
        }
    }, 5000);
});

/**
 * Performance monitoring
 */
function monitorPerformance() {
    let frameCount = 0;
    let lastTime = performance.now();
    
    function checkFPS() {
        frameCount++;
        const currentTime = performance.now();
        
        if (currentTime >= lastTime + 1000) {
            const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
            
            if (fps < 30) {
                console.warn(`Low FPS detected: ${fps} fps`);
                
                // Suggest performance improvements
                if (window.visualization) {
                    console.log('💡 Try reducing animation speed or closing other browser tabs');
                }
            }
            
            frameCount = 0;
            lastTime = currentTime;
        }
        
        requestAnimationFrame(checkFPS);
    }
    
    requestAnimationFrame(checkFPS);
}

/**
 * Keyboard shortcuts
 */
document.addEventListener('keydown', (event) => {
    // Ctrl/Cmd + Enter to process sentence
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        if (window.uiController && !window.uiController.isProcessing) {
            window.uiController.processFullSentence();
        }
    }
    
    // Escape to reset
    if (event.key === 'Escape') {
        event.preventDefault();
        if (window.uiController) {
            window.uiController.resetVisualization();
        }
    }
    
    // Space to pause/resume (if implementing pause functionality)
    if (event.key === ' ' && event.target.tagName !== 'INPUT') {
        event.preventDefault();
        // Toggle pause functionality could be added here
    }
});

/**
 * Initialize when DOM is loaded
 */
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    // DOM already loaded
    initializeApp();
}

// Start performance monitoring
monitorPerformance();

// Export for debugging
window.app = {
    initialize: initializeApp,
    showWelcome: showWelcomeMessage,
    version: '1.0.0'
};
