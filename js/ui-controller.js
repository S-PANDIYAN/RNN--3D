/**
 * UI Controller
 * Manages the user interface and coordinates between the model and visualization
 */

class UIController {
    constructor() {
        this.currentStep = 'ready';
        this.currentProgress = 0;
        this.isProcessing = false;
        this.stepQueue = [];
        
        this.initializeElements();
        this.attachEventListeners();
        this.setupTooltips();
    }
    
    initializeElements() {
        // Model selection elements
        this.demoModelRadio = document.getElementById('demo-model');
        this.amazonModelRadio = document.getElementById('amazon-model');
        this.kerasModelRadio = document.getElementById('keras-model');
        this.kerasUpload = document.getElementById('keras-upload');
        this.modelFile = document.getElementById('model-file');
        this.tokenizerFile = document.getElementById('tokenizer-file');
        this.loadModelBtn = document.getElementById('load-model-btn');
        this.modelStatus = document.getElementById('model-status');
        
        // Input elements
        this.sentenceInput = document.getElementById('sentence-input');
        this.processBtn = document.getElementById('process-btn');
        this.resetBtn = document.getElementById('reset-btn');
        this.animationSpeed = document.getElementById('animation-speed');
        this.speedValue = document.getElementById('speed-value');
        
        // Phase control buttons
        this.tokenizeBtn = document.getElementById('tokenize-btn');
        this.embedBtn = document.getElementById('embed-btn');
        this.processRnnBtn = document.getElementById('process-rnn-btn');
        this.predictBtn = document.getElementById('predict-btn');
        this.backpropBtn = document.getElementById('backprop-btn');
        
        // Info display elements
        this.stepTitle = document.getElementById('step-title');
        this.stepDescription = document.getElementById('step-description');
        this.equationDisplay = document.getElementById('equation-display');
        this.tokenCount = document.getElementById('token-count');
        this.currentTimestep = document.getElementById('current-timestep');
        this.hiddenSize = document.getElementById('hidden-size');
        
        // Sentiment result elements
        this.positiveFill = document.getElementById('positive-fill');
        this.neutralFill = document.getElementById('neutral-fill');
        this.negativeFill = document.getElementById('negative-fill');
        this.positiveScore = document.getElementById('positive-score');
        this.neutralScore = document.getElementById('neutral-score');
        this.negativeScore = document.getElementById('negative-score');
        
        // Progress elements
        this.progressFill = document.getElementById('progress-fill');
        
        // Tooltip
        this.tooltip = document.getElementById('tooltip');
        this.tooltipTitle = document.getElementById('tooltip-title');
        this.tooltipText = document.getElementById('tooltip-text');
    }
    
    attachEventListeners() {
        // Model selection listeners
        this.demoModelRadio.addEventListener('change', () => this.toggleModelType());
        this.amazonModelRadio.addEventListener('change', () => this.toggleModelType());
        this.kerasModelRadio.addEventListener('change', () => this.toggleModelType());
        this.loadModelBtn.addEventListener('click', () => this.loadKerasModel());
        
        // Main control buttons
        this.processBtn.addEventListener('click', () => this.processFullSentence());
        this.resetBtn.addEventListener('click', () => this.resetVisualization());
        
        // Animation speed control
        this.animationSpeed.addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            this.speedValue.textContent = `${speed}x`;
            if (window.visualization) {
                window.visualization.setAnimationSpeed(speed);
            }
        });
        
        // Phase control buttons
        this.tokenizeBtn.addEventListener('click', () => this.executeStep('tokenize'));
        this.embedBtn.addEventListener('click', () => this.executeStep('embed'));
        this.processRnnBtn.addEventListener('click', () => this.executeStep('rnn'));
        this.predictBtn.addEventListener('click', () => this.executeStep('predict'));
        this.backpropBtn.addEventListener('click', () => this.executeStep('backprop'));
        
        // Enter key support for input
        this.sentenceInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.processFullSentence();
            }
        });
        
        // Prevent form submission
        this.sentenceInput.addEventListener('submit', (e) => e.preventDefault());
    }
    
    setupTooltips() {
        const tooltipElements = [
            { element: this.tokenizeBtn, content: { title: "Tokenization", text: "Break the sentence into individual words or tokens for processing." }},
            { element: this.embedBtn, content: { title: "Word Embeddings", text: "Convert tokens into dense vector representations that capture semantic meaning." }},
            { element: this.processRnnBtn, content: { title: "RNN Processing", text: "Process each word sequentially, updating the hidden state at each time step." }},
            { element: this.predictBtn, content: { title: "Prediction", text: "Use the final hidden state to predict sentiment classes with probabilities." }},
            { element: this.backpropBtn, content: { title: "Backpropagation", text: "Show how gradients flow backward through time to update network weights." }}
        ];
        
        tooltipElements.forEach(({ element, content }) => {
            element.addEventListener('mouseenter', (e) => this.showTooltip(e, content));
            element.addEventListener('mouseleave', () => this.hideTooltip());
            element.addEventListener('mousemove', (e) => this.updateTooltipPosition(e));
        });
    }
    
    /**
     * Toggle between demo and Keras model
     */
    toggleModelType() {
        if (this.kerasModelRadio.checked) {
            this.kerasUpload.style.display = 'flex';
        } else {
            this.kerasUpload.style.display = 'none';
            
            if (this.amazonModelRadio.checked) {
                // Switch to Amazon model
                this.loadAmazonModel();
            } else {
                // Reset to demo model
                window.rnnModel.useKerasModel = false;
                this.updateModelStatus('Using demo model', 'success');
            }
        }
    }

    /**
     * Load Amazon RNN model
     */
    async loadAmazonModel() {
        try {
            this.updateModelStatus('Loading Amazon RNN model...', 'loading');
            
            // Load the Amazon model through the RNN model
            await window.rnnModel.loadAmazonModel();
            
            this.updateModelStatus('Amazon RNN model loaded successfully!', 'success');
            console.log('Amazon model integration complete');
            
        } catch (error) {
            console.error('Failed to load Amazon model:', error);
            this.updateModelStatus('Failed to load Amazon model', 'error');
            
            // Fall back to demo model
            this.demoModelRadio.checked = true;
            this.amazonModelRadio.checked = false;
            window.rnnModel.useKerasModel = false;
        }
    }

    /**
     * Load Keras model from files
     */
    async loadKerasModel() {
        if (!this.modelFile.files[0]) {
            this.showError('Please select a model file (.json or .h5)');
            return;
        }

        try {
            this.updateModelStatus('Loading model...', 'loading');
            this.loadModelBtn.disabled = true;

            // Create URLs for the files
            const modelFile = this.modelFile.files[0];
            const tokenizerFile = this.tokenizerFile.files[0];

            let modelUrl, tokenizerUrl;

            // Handle different file types
            if (modelFile.name.endsWith('.json')) {
                // TensorFlow.js model
                modelUrl = URL.createObjectURL(modelFile);
            } else if (modelFile.name.endsWith('.h5')) {
                // Keras H5 model (needs conversion)
                this.showError('H5 models need to be converted to TensorFlow.js format first. Please use tfjs-converter.');
                this.updateModelStatus('H5 format not supported', 'error');
                return;
            } else {
                this.showError('Unsupported model format. Please use .json (TensorFlow.js) format.');
                this.updateModelStatus('Unsupported format', 'error');
                return;
            }

            if (tokenizerFile) {
                tokenizerUrl = URL.createObjectURL(tokenizerFile);
            }

            // Load the model
            await window.rnnModel.loadKerasModel(modelUrl, tokenizerUrl);
            
            this.updateModelStatus('✅ Model loaded successfully!', 'success');
            
            // Clean up URLs
            URL.revokeObjectURL(modelUrl);
            if (tokenizerUrl) URL.revokeObjectURL(tokenizerUrl);

        } catch (error) {
            console.error('Model loading error:', error);
            this.updateModelStatus('❌ Failed to load model', 'error');
            this.showError(`Failed to load model: ${error.message}`);
        } finally {
            this.loadModelBtn.disabled = false;
        }
    }

    /**
     * Update model status display
     */
    updateModelStatus(message, type = 'info') {
        this.modelStatus.textContent = message;
        this.modelStatus.className = `model-status ${type}`;
    }

    /**
     * Process the entire sentence through all steps
     */
    async processFullSentence() {
        if (this.isProcessing) return;
        
        const sentence = this.sentenceInput.value.trim();
        if (!sentence) {
            this.showError('Please enter a sentence to analyze.');
            return;
        }
        
        this.isProcessing = true;
        this.processBtn.disabled = true;
        this.disablePhaseButtons();
        
        try {
            // Store the current sentence for step-by-step processing
            this.currentSentence = sentence;
            this.currentPrediction = null;
            
            // Execute all steps in sequence
            await this.executeStep('tokenize');
            await this.delay(500);
            await this.executeStep('embed');
            await this.delay(500);
            await this.executeStep('rnn');
            await this.delay(500);
            await this.executeStep('predict');
            await this.delay(500);
            await this.executeStep('backprop');
            
            this.updateProgress(100);
            this.updateStepInfo('complete', 'Analysis Complete!', 'The RNN has successfully processed your sentence and predicted the sentiment.');
            
        } catch (error) {
            console.error('Error processing sentence:', error);
            this.showError('An error occurred while processing the sentence.');
        } finally {
            this.isProcessing = false;
            this.processBtn.disabled = false;
            this.enablePhaseButtons();
        }
    }
    
    /**
     * Execute a specific step of the RNN process
     */
    async executeStep(step) {
        if (!this.currentSentence && step !== 'tokenize') {
            this.showError('Please process a sentence first.');
            return;
        }
        
        try {
            switch (step) {
                case 'tokenize':
                    await this.executeTokenizeStep();
                    break;
                case 'embed':
                    await this.executeEmbedStep();
                    break;
                case 'rnn':
                    await this.executeRNNStep();
                    break;
                case 'predict':
                    await this.executePredictStep();
                    break;
                case 'backprop':
                    await this.executeBackpropStep();
                    break;
            }
            
            this.markStepComplete(step);
        } catch (error) {
            console.error(`Error executing ${step} step:`, error);
            this.showError(`Error in ${step} step: ${error.message}`);
        }
    }
    
    async executeTokenizeStep() {
        const sentence = this.sentenceInput.value.trim();
        if (!sentence) {
            throw new Error('Please enter a sentence to tokenize.');
        }
        
        this.currentSentence = sentence;
        const tokens = window.rnnModel.tokenize(sentence);
        this.currentTokens = tokens;
        
        this.updateStepInfo('tokenize', 'Tokenization', `Breaking "${sentence}" into ${tokens.length} tokens: [${tokens.join(', ')}]`);
        this.updateEquation('sentence → [token₁, token₂, ..., tokenₙ]');
        this.updateTokenCount(tokens.length);
        this.updateProgress(20);
        
        if (window.visualization) {
            await window.visualization.visualizeTokenization(tokens);
        }
    }
    
    async executeEmbedStep() {
        if (!this.currentTokens) {
            throw new Error('Please tokenize a sentence first.');
        }
        
        const indices = window.rnnModel.tokensToIndices(this.currentTokens);
        const embeddings = window.rnnModel.indicesToEmbeddings(indices);
        this.currentEmbeddings = embeddings;
        
        this.updateStepInfo('embed', 'Word Embeddings', `Converting tokens to ${embeddings[0].length}-dimensional embedding vectors.`);
        this.updateEquation('token → embedding_vector ∈ ℝᵈ');
        this.updateProgress(40);
        
        if (window.visualization) {
            await window.visualization.visualizeEmbedding(this.currentTokens, embeddings);
        }
    }
    
    async executeRNNStep() {
        if (!this.currentEmbeddings) {
            throw new Error('Please generate embeddings first.');
        }
        
        const result = window.rnnModel.forward(this.currentEmbeddings);
        this.currentComputationHistory = result.computationHistory;
        
        this.updateStepInfo('rnn', 'RNN Processing', `Processing ${this.currentTokens.length} time steps through the recurrent network.`);
        this.updateEquation('hₜ = tanh(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁ + bₕ)');
        this.updateProgress(60);
        
        if (window.visualization) {
            await window.visualization.visualizeRNNProcessing(result.computationHistory);
        }
        
        // Update current timestep during processing
        for (let t = 0; t < result.computationHistory.length; t++) {
            this.updateCurrentTimestep(t + 1);
            await this.delay(300);
        }
    }
    
    async executePredictStep() {
        if (!this.currentSentence) {
            throw new Error('Please process a sentence first.');
        }

        // Debug: Check which model is being used
        console.log('=== Prediction Debug Info ===');
        console.log('Amazon model radio checked:', this.amazonModelRadio.checked);
        console.log('Demo model radio checked:', this.demoModelRadio.checked);
        console.log('RNN model useKerasModel:', window.rnnModel.useKerasModel);
        console.log('Amazon loader exists:', !!window.rnnModel.amazonLoader);
        console.log('Amazon loader loaded:', window.rnnModel.amazonLoader?.isLoaded);
        
        // Force load Amazon model if selected but not loaded
        if (this.amazonModelRadio.checked && (!window.rnnModel.amazonLoader || !window.rnnModel.amazonLoader.isLoaded)) {
            console.log('Force loading Amazon model...');
            await this.loadAmazonModel();
        }

        // Use the enhanced prediction method that handles both Amazon and demo models
        this.currentPrediction = await window.rnnModel.predictSentiment(this.currentSentence);
        
        console.log('Prediction result:', this.currentPrediction);
        
        // Convert the prediction result to the expected format
        let probabilities;
        let predictedLabel;
        let confidence;
        
        if (this.currentPrediction.modelType === 'amazon_rnn') {
            // Amazon model returns single sentiment score (0-1)
            const score = this.currentPrediction.score;
            const sentiment = this.currentPrediction.sentiment;
            const conf = this.currentPrediction.confidence;
            
            // Ensure confidence is at least 60% for clear results
            const adjustedConf = Math.max(0.6, conf);
            
            if (sentiment === 'positive') {
                probabilities = [adjustedConf, (1 - adjustedConf) / 2, (1 - adjustedConf) / 2];
                predictedLabel = 'Positive';
            } else if (sentiment === 'negative') {
                probabilities = [(1 - adjustedConf) / 2, (1 - adjustedConf) / 2, adjustedConf];
                predictedLabel = 'Negative';
            } else {
                probabilities = [(1 - adjustedConf) / 2, adjustedConf, (1 - adjustedConf) / 2];
                predictedLabel = 'Neutral';
            }
            
            confidence = adjustedConf;
        } else {
            // Demo model format - give more realistic results
            const score = this.currentPrediction.score;
            console.log('Demo model score:', score);
            
            if (score > 0.6) {
                const conf = Math.min(0.9, score + 0.2);
                probabilities = [conf, (1 - conf) / 2, (1 - conf) / 2];
                predictedLabel = 'Positive';
                confidence = conf;
            } else if (score < 0.4) {
                const conf = Math.min(0.9, (1 - score) + 0.2);
                probabilities = [(1 - conf) / 2, (1 - conf) / 2, conf];
                predictedLabel = 'Negative';
                confidence = conf;
            } else {
                probabilities = [0.25, 0.6, 0.15];
                predictedLabel = 'Neutral';
                confidence = 0.6;
            }
        }
        
        console.log('Final probabilities:', probabilities);
        console.log('Predicted label:', predictedLabel);
        
        this.updateStepInfo('predict', 'Sentiment Prediction', `Predicted: ${predictedLabel} (${(confidence * 100).toFixed(1)}% confidence)`);
        this.updateEquation('y = softmax(Wₕᵧ·hₜ + bᵧ)');
        this.updateProgress(80);
        
        this.updateSentimentScores(probabilities);
        
        if (window.visualization) {
            await window.visualization.visualizePrediction(probabilities);
        }
    }
    
    async executeBackpropStep() {
        if (!this.currentPrediction) {
            throw new Error('Please generate a prediction first.');
        }
        
        // Simulate backpropagation with assumed target (opposite of prediction for demo)
        const targetSentiment = this.currentPrediction.predictedClass === 0 ? 2 : 0;
        const backpropResult = window.rnnModel.backpropagateDemo(this.currentSentence, targetSentiment);
        
        this.updateStepInfo('backprop', 'Backpropagation Through Time', `Computing gradients and propagating errors backward through ${this.currentTokens.length} time steps.`);
        this.updateEquation('∂L/∂W = Σₜ ∂L/∂hₜ · ∂hₜ/∂W');
        this.updateProgress(100);
        
        if (window.visualization) {
            await window.visualization.visualizeBackpropagation(backpropResult.gradients);
        }
    }
    
    /**
     * Reset the visualization
     */
    resetVisualization() {
        this.currentStep = 'ready';
        this.currentProgress = 0;
        this.currentSentence = null;
        this.currentTokens = null;
        this.currentEmbeddings = null;
        this.currentPrediction = null;
        this.currentComputationHistory = null;
        
        this.updateStepInfo('ready', 'Ready to Start', 'Enter a sentence and click "Process Sentence" to begin the RNN visualization.');
        this.updateEquation('Ready to process...');
        this.updateTokenCount(0);
        this.updateCurrentTimestep(0);
        this.updateProgress(0);
        this.clearSentimentScores();
        this.resetPhaseButtons();
        
        if (window.visualization) {
            window.visualization.clearScene();
        }
        
        this.isProcessing = false;
        this.processBtn.disabled = false;
    }
    
    /**
     * Update UI elements
     */
    updateStepInfo(step, title, description) {
        this.currentStep = step;
        this.stepTitle.textContent = title;
        this.stepDescription.textContent = description;
        
        // Add animation class
        this.stepTitle.classList.add('animate-in');
        this.stepDescription.classList.add('animate-in');
        
        setTimeout(() => {
            this.stepTitle.classList.remove('animate-in');
            this.stepDescription.classList.remove('animate-in');
        }, 500);
    }
    
    updateEquation(equation) {
        this.equationDisplay.textContent = equation;
        this.equationDisplay.classList.add('highlight');
        setTimeout(() => {
            this.equationDisplay.classList.remove('highlight');
        }, 2000);
    }
    
    updateTokenCount(count) {
        this.tokenCount.textContent = count;
        this.tokenCount.classList.add('highlight');
        setTimeout(() => {
            this.tokenCount.classList.remove('highlight');
        }, 1000);
    }
    
    updateCurrentTimestep(step) {
        this.currentTimestep.textContent = step;
        this.currentTimestep.classList.add('highlight');
        setTimeout(() => {
            this.currentTimestep.classList.remove('highlight');
        }, 1000);
    }
    
    updateProgress(percent) {
        this.currentProgress = percent;
        this.progressFill.style.width = `${percent}%`;
    }
    
    updateSentimentScores(probabilities) {
        const [positive, neutral, negative] = probabilities;
        const sentimentLabels = ['Positive', 'Neutral', 'Negative'];
        const sentimentEmojis = ['😊', '😐', '😠'];
        
        // Find the dominant sentiment
        const maxIndex = probabilities.indexOf(Math.max(...probabilities));
        const dominantSentiment = sentimentLabels[maxIndex];
        const dominantScore = probabilities[maxIndex];
        const dominantEmoji = sentimentEmojis[maxIndex];
        
        // Get sentiment item elements
        const sentimentItems = document.querySelectorAll('.sentiment-item');
        
        // Reset all bars and remove previous highlights
        this.positiveFill.style.width = '0%';
        this.neutralFill.style.width = '0%';
        this.negativeFill.style.width = '0%';
        
        this.positiveScore.textContent = '0%';
        this.neutralScore.textContent = '0%';
        this.negativeScore.textContent = '0%';
        
        // Remove all previous dominant classes and highlights
        sentimentItems.forEach(item => {
            item.classList.remove('dominant');
        });
        this.positiveScore.classList.remove('highlight');
        this.neutralScore.classList.remove('highlight');
        this.negativeScore.classList.remove('highlight');
        
        // Show only the dominant sentiment
        if (maxIndex === 0) { // Positive
            this.positiveFill.style.width = '100%';
            this.positiveScore.textContent = `${(dominantScore * 100).toFixed(1)}%`;
            this.positiveScore.classList.add('highlight');
            sentimentItems[0].classList.add('dominant');
        } else if (maxIndex === 1) { // Neutral
            this.neutralFill.style.width = '100%';
            this.neutralScore.textContent = `${(dominantScore * 100).toFixed(1)}%`;
            this.neutralScore.classList.add('highlight');
            sentimentItems[1].classList.add('dominant');
        } else { // Negative
            this.negativeFill.style.width = '100%';
            this.negativeScore.textContent = `${(dominantScore * 100).toFixed(1)}%`;
            this.negativeScore.classList.add('highlight');
            sentimentItems[2].classList.add('dominant');
        }
        
        // Update step description to show the result
        this.stepDescription.textContent = `Prediction complete! The sentence is classified as: ${dominantEmoji} ${dominantSentiment} (${(dominantScore * 100).toFixed(1)}% confidence)`;
    }
    
    clearSentimentScores() {
        // Reset all bars
        this.positiveFill.style.width = '0%';
        this.neutralFill.style.width = '0%';
        this.negativeFill.style.width = '0%';
        
        this.positiveScore.textContent = '0%';
        this.neutralScore.textContent = '0%';
        this.negativeScore.textContent = '0%';
        
        // Remove all highlights and dominant classes
        const sentimentItems = document.querySelectorAll('.sentiment-item');
        sentimentItems.forEach(item => {
            item.classList.remove('dominant');
        });
        
        this.positiveScore.classList.remove('highlight');
        this.neutralScore.classList.remove('highlight');
        this.negativeScore.classList.remove('highlight');
    }
    
    /**
     * Phase button management
     */
    disablePhaseButtons() {
        [this.tokenizeBtn, this.embedBtn, this.processRnnBtn, this.predictBtn, this.backpropBtn]
            .forEach(btn => btn.disabled = true);
    }
    
    enablePhaseButtons() {
        [this.tokenizeBtn, this.embedBtn, this.processRnnBtn, this.predictBtn, this.backpropBtn]
            .forEach(btn => btn.disabled = false);
    }
    
    markStepComplete(step) {
        const stepButtons = {
            tokenize: this.tokenizeBtn,
            embed: this.embedBtn,
            rnn: this.processRnnBtn,
            predict: this.predictBtn,
            backprop: this.backpropBtn
        };
        
        const button = stepButtons[step];
        if (button) {
            button.classList.add('active');
            setTimeout(() => {
                button.classList.remove('active');
            }, 2000);
        }
    }
    
    resetPhaseButtons() {
        [this.tokenizeBtn, this.embedBtn, this.processRnnBtn, this.predictBtn, this.backpropBtn]
            .forEach(btn => {
                btn.classList.remove('active');
                btn.disabled = false;
            });
    }
    
    /**
     * Tooltip management
     */
    showTooltip(event, content) {
        this.tooltipTitle.textContent = content.title;
        this.tooltipText.textContent = content.text;
        this.tooltip.classList.add('show');
        this.updateTooltipPosition(event);
    }
    
    hideTooltip() {
        this.tooltip.classList.remove('show');
    }
    
    updateTooltipPosition(event) {
        const rect = this.tooltip.getBoundingClientRect();
        let x = event.clientX + 10;
        let y = event.clientY - rect.height - 10;
        
        // Keep tooltip within viewport
        if (x + rect.width > window.innerWidth) {
            x = event.clientX - rect.width - 10;
        }
        if (y < 0) {
            y = event.clientY + 10;
        }
        
        this.tooltip.style.left = `${x}px`;
        this.tooltip.style.top = `${y}px`;
    }
    
    /**
     * Error handling
     */
    showError(message) {
        // Create error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.textContent = message;
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(45deg, #ff6b6b, #ee5a52);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            z-index: 10000;
            animation: slideInRight 0.3s ease-out;
        `;
        
        document.body.appendChild(errorDiv);
        
        // Remove after 3 seconds
        setTimeout(() => {
            errorDiv.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => {
                document.body.removeChild(errorDiv);
            }, 300);
        }, 3000);
    }
    
    /**
     * Utility functions
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    /**
     * Get current processing state
     */
    getCurrentState() {
        return {
            step: this.currentStep,
            progress: this.currentProgress,
            sentence: this.currentSentence,
            tokens: this.currentTokens,
            embeddings: this.currentEmbeddings,
            prediction: this.currentPrediction,
            isProcessing: this.isProcessing
        };
    }
}

// Export for use in other modules
window.UIController = UIController;
