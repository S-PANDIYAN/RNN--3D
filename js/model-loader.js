/**
 * Model Loader for Keras Models
 * Handles loading and interfacing with TensorFlow.js converted Keras models
 */

class ModelLoader {
    constructor() {
        this.model = null;
        this.tokenizer = null;
        this.maxSequenceLength = 100;
        this.vocabSize = 10000;
        this.isLoaded = false;
        this.modelMetadata = null;
    }

    /**
     * Load a TensorFlow.js model from URL or file
     */
    async loadModel(modelPath, tokenizerPath = null) {
        try {
            console.log('Loading TensorFlow.js model...');
            
            // Load TensorFlow.js if not already loaded
            if (typeof tf === 'undefined') {
                await this.loadTensorFlowJS();
            }

            // Load the model
            this.model = await tf.loadLayersModel(modelPath);
            console.log('✓ Model loaded successfully');
            
            // Load tokenizer if provided
            if (tokenizerPath) {
                await this.loadTokenizer(tokenizerPath);
            } else {
                // Create a simple tokenizer
                this.createSimpleTokenizer();
            }

            // Extract model metadata
            this.extractModelMetadata();
            
            this.isLoaded = true;
            return true;

        } catch (error) {
            console.error('❌ Error loading model:', error);
            throw new Error(`Failed to load model: ${error.message}`);
        }
    }

    /**
     * Load TensorFlow.js library dynamically
     */
    async loadTensorFlowJS() {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js';
            script.onload = () => {
                console.log('✓ TensorFlow.js loaded');
                resolve();
            };
            script.onerror = () => {
                reject(new Error('Failed to load TensorFlow.js'));
            };
            document.head.appendChild(script);
        });
    }

    /**
     * Load tokenizer configuration
     */
    async loadTokenizer(tokenizerPath) {
        try {
            const response = await fetch(tokenizerPath);
            const tokenizerConfig = await response.json();
            
            this.tokenizer = {
                wordIndex: tokenizerConfig.word_index || {},
                indexWord: {},
                maxSequenceLength: tokenizerConfig.max_sequence_length || 100,
                vocabSize: tokenizerConfig.vocab_size || 10000
            };

            // Create reverse mapping
            Object.entries(this.tokenizer.wordIndex).forEach(([word, index]) => {
                this.tokenizer.indexWord[index] = word;
            });

            this.maxSequenceLength = this.tokenizer.maxSequenceLength;
            this.vocabSize = this.tokenizer.vocabSize;

            console.log('✓ Tokenizer loaded');
        } catch (error) {
            console.warn('Could not load tokenizer, using simple tokenizer');
            this.createSimpleTokenizer();
        }
    }

    /**
     * Create a simple tokenizer for demo purposes
     */
    createSimpleTokenizer() {
        // Common words for sentiment analysis
        const commonWords = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'good', 'bad', 'great', 'terrible', 'awesome', 'awful', 'amazing', 'horrible', 'excellent', 'poor',
            'love', 'hate', 'like', 'dislike', 'enjoy', 'despise', 'adore', 'loathe',
            'movie', 'film', 'book', 'product', 'service', 'item', 'place', 'restaurant', 'hotel',
            'best', 'worst', 'better', 'worse', 'perfect', 'terrible', 'wonderful', 'disappointing',
            'recommend', 'avoid', 'buy', 'purchase', 'waste', 'money', 'time', 'quality',
            'fast', 'slow', 'quick', 'delayed', 'prompt', 'late', 'early', 'on-time',
            'helpful', 'useless', 'effective', 'ineffective', 'useful', 'worthless',
            'beautiful', 'ugly', 'pretty', 'gorgeous', 'hideous', 'attractive', 'unattractive',
            'easy', 'difficult', 'hard', 'simple', 'complex', 'straightforward', 'complicated',
            'clean', 'dirty', 'fresh', 'stale', 'new', 'old', 'modern', 'outdated'
        ];

        this.tokenizer = {
            wordIndex: { '<PAD>': 0, '<UNK>': 1 },
            indexWord: { 0: '<PAD>', 1: '<UNK>' },
            maxSequenceLength: this.maxSequenceLength,
            vocabSize: this.vocabSize
        };

        // Add common words to tokenizer
        commonWords.forEach((word, index) => {
            const id = index + 2;
            this.tokenizer.wordIndex[word.toLowerCase()] = id;
            this.tokenizer.indexWord[id] = word.toLowerCase();
        });
    }

    /**
     * Extract model metadata for visualization
     */
    extractModelMetadata() {
        if (!this.model) return;

        this.modelMetadata = {
            inputShape: this.model.inputs[0].shape,
            outputShape: this.model.outputs[0].shape,
            layers: this.model.layers.map(layer => ({
                name: layer.name,
                className: layer.getClassName(),
                units: layer.units || null,
                activation: layer.activation ? layer.activation.name : null
            })),
            trainableParams: this.model.countParams()
        };

        console.log('Model Metadata:', this.modelMetadata);
    }

    /**
     * Tokenize and preprocess text for the model
     */
    preprocessText(text) {
        // Clean text
        const cleanedText = text.toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();

        // Tokenize
        const words = cleanedText.split(' ');
        const tokens = words.map(word => 
            this.tokenizer.wordIndex[word] || this.tokenizer.wordIndex['<UNK>']
        );

        // Pad or truncate to max sequence length
        let paddedTokens = new Array(this.maxSequenceLength).fill(0);
        for (let i = 0; i < Math.min(tokens.length, this.maxSequenceLength); i++) {
            paddedTokens[i] = tokens[i];
        }

        return {
            originalText: text,
            cleanedText: cleanedText,
            words: words,
            tokens: tokens,
            paddedTokens: paddedTokens,
            sequenceLength: Math.min(tokens.length, this.maxSequenceLength)
        };
    }

    /**
     * Make prediction using the loaded model
     */
    async predict(text) {
        if (!this.isLoaded || !this.model) {
            throw new Error('Model not loaded');
        }

        try {
            // Preprocess the text
            const preprocessed = this.preprocessText(text);
            
            // Create tensor
            const inputTensor = tf.tensor2d([preprocessed.paddedTokens], [1, this.maxSequenceLength]);
            
            // Make prediction
            const prediction = this.model.predict(inputTensor);
            const probabilitiesArray = await prediction.data();
            
            // Clean up tensors
            inputTensor.dispose();
            prediction.dispose();

            // Convert to array and get predicted class
            const probabilities = Array.from(probabilitiesArray);
            const predictedClass = probabilities.indexOf(Math.max(...probabilities));
            
            // Map to sentiment labels (assuming binary or 3-class classification)
            let sentimentLabels;
            if (probabilities.length === 2) {
                sentimentLabels = ['Negative', 'Positive'];
            } else if (probabilities.length === 3) {
                sentimentLabels = ['Negative', 'Neutral', 'Positive'];
            } else {
                sentimentLabels = probabilities.map((_, i) => `Class ${i}`);
            }

            return {
                preprocessed: preprocessed,
                probabilities: probabilities,
                predictedClass: predictedClass,
                predictedLabel: sentimentLabels[predictedClass],
                confidence: probabilities[predictedClass],
                sentimentLabels: sentimentLabels
            };

        } catch (error) {
            console.error('Prediction error:', error);
            throw new Error(`Prediction failed: ${error.message}`);
        }
    }

    /**
     * Get intermediate layer outputs for visualization
     */
    async getLayerOutputs(text, layerNames = null) {
        if (!this.isLoaded || !this.model) {
            throw new Error('Model not loaded');
        }

        try {
            const preprocessed = this.preprocessText(text);
            const inputTensor = tf.tensor2d([preprocessed.paddedTokens], [1, this.maxSequenceLength]);

            const outputs = {};
            
            if (layerNames) {
                // Get specific layer outputs
                for (const layerName of layerNames) {
                    const layer = this.model.getLayer(layerName);
                    if (layer) {
                        const layerModel = tf.model({
                            inputs: this.model.inputs,
                            outputs: layer.output
                        });
                        const output = layerModel.predict(inputTensor);
                        outputs[layerName] = await output.data();
                        output.dispose();
                        layerModel.dispose();
                    }
                }
            } else {
                // Get outputs from all layers
                for (const layer of this.model.layers) {
                    if (layer.output) {
                        try {
                            const layerModel = tf.model({
                                inputs: this.model.inputs,
                                outputs: layer.output
                            });
                            const output = layerModel.predict(inputTensor);
                            outputs[layer.name] = await output.data();
                            output.dispose();
                            layerModel.dispose();
                        } catch (e) {
                            console.warn(`Could not get output for layer ${layer.name}`);
                        }
                    }
                }
            }

            inputTensor.dispose();
            return outputs;

        } catch (error) {
            console.error('Error getting layer outputs:', error);
            throw error;
        }
    }

    /**
     * Get model summary
     */
    getModelSummary() {
        if (!this.model) return null;

        return {
            ...this.modelMetadata,
            isLoaded: this.isLoaded,
            vocabSize: this.vocabSize,
            maxSequenceLength: this.maxSequenceLength,
            hasTokenizer: !!this.tokenizer
        };
    }

    /**
     * Export model predictions in a format compatible with the visualization
     */
    async getVisualizationData(text) {
        const prediction = await this.predict(text);
        const layerOutputs = await this.getLayerOutputs(text);

        return {
            tokens: prediction.preprocessed.words,
            indices: prediction.preprocessed.tokens,
            embeddings: this.generateEmbeddingVisualization(prediction.preprocessed.paddedTokens),
            probabilities: prediction.probabilities,
            predictedClass: prediction.predictedClass,
            predictedLabel: prediction.predictedLabel,
            confidence: prediction.confidence,
            hiddenStates: this.extractHiddenStates(layerOutputs),
            computationHistory: this.generateComputationHistory(prediction.preprocessed, layerOutputs)
        };
    }

    /**
     * Generate embedding visualization data
     */
    generateEmbeddingVisualization(tokens) {
        // For visualization, create simplified embedding vectors
        return tokens.slice(0, tokens.findIndex(t => t === 0) || tokens.length).map(token => {
            const embedding = new Array(50).fill(0);
            for (let i = 0; i < 50; i++) {
                embedding[i] = Math.sin(token * 0.1 + i * 0.2) * 0.5;
            }
            return embedding;
        });
    }

    /**
     * Extract hidden states from layer outputs
     */
    extractHiddenStates(layerOutputs) {
        const hiddenStates = [];
        
        // Look for LSTM, GRU, or similar layer outputs
        Object.entries(layerOutputs).forEach(([layerName, output]) => {
            if (layerName.includes('lstm') || layerName.includes('gru') || layerName.includes('rnn')) {
                // Convert flat array to sequence of hidden states
                const outputArray = Array.from(output);
                const hiddenSize = outputArray.length / this.maxSequenceLength;
                
                for (let t = 0; t < this.maxSequenceLength; t++) {
                    const start = t * hiddenSize;
                    const end = start + hiddenSize;
                    hiddenStates.push(outputArray.slice(start, end));
                }
            }
        });

        return hiddenStates;
    }

    /**
     * Generate computation history for visualization
     */
    generateComputationHistory(preprocessed, layerOutputs) {
        const history = [];
        const sequenceLength = preprocessed.sequenceLength;

        for (let t = 0; t < sequenceLength; t++) {
            history.push({
                timeStep: t,
                input: this.generateEmbeddingVisualization([preprocessed.tokens[t]])[0] || [],
                hiddenState: layerOutputs.lstm ? Array.from(layerOutputs.lstm).slice(t * 64, (t + 1) * 64) : [],
                inputContribution: [],
                hiddenContribution: [],
                combined: []
            });
        }

        return history;
    }
}

// Export for use in other modules
window.ModelLoader = ModelLoader;
