/**
 * RNN Model Implementation for Sentiment Analysis
 * This class handles the mathematical operations and data flow of the RNN
 */

class RNNModel {
    constructor(hiddenSize = 64, vocabSize = 1000, embeddingSize = 50) {
        this.hiddenSize = hiddenSize;
        this.vocabSize = vocabSize;
        this.embeddingSize = embeddingSize;
        
        // Model loader for Keras models
        this.modelLoader = new ModelLoader();
        this.useKerasModel = false;
        
        // Initialize weights (simplified for educational purposes)
        this.weights = {
            // Input to hidden weights
            Wxh: this.randomMatrix(hiddenSize, embeddingSize),
            // Hidden to hidden weights  
            Whh: this.randomMatrix(hiddenSize, hiddenSize),
            // Hidden to output weights
            Why: this.randomMatrix(3, hiddenSize), // 3 sentiment classes
            // Bias vectors
            bh: this.randomVector(hiddenSize),
            by: this.randomVector(3)
        };
        
        // Vocabulary for simple tokenization
        this.vocabulary = new Map();
        this.reverseVocab = new Map();
        this.initializeVocabulary();
        
        // Training data for educational examples
        this.trainingExamples = [
            { text: "I love this movie", sentiment: 0 }, // positive
            { text: "This is amazing", sentiment: 0 },
            { text: "Great film", sentiment: 0 },
            { text: "Okay movie", sentiment: 1 }, // neutral
            { text: "It was fine", sentiment: 1 },
            { text: "Average film", sentiment: 1 },
            { text: "I hate this", sentiment: 2 }, // negative
            { text: "Terrible movie", sentiment: 2 },
            { text: "Worst film ever", sentiment: 2 }
        ];
        
        // Store computation history for visualization
        this.computationHistory = [];
        this.currentTimeStep = 0;
    }
    
    /**
     * Load a Keras model
     */
    async loadKerasModel(modelPath, tokenizerPath = null) {
        try {
            console.log('Loading Keras model for RNN visualization...');
            await this.modelLoader.loadModel(modelPath, tokenizerPath);
            this.useKerasModel = true;
            
            // Update model parameters from loaded model
            const summary = this.modelLoader.getModelSummary();
            if (summary) {
                this.vocabSize = summary.vocabSize;
                this.hiddenSize = summary.layers.find(l => l.className === 'LSTM')?.units || this.hiddenSize;
            }
            
            console.log('✅ Keras model loaded successfully!');
            console.log('Model summary:', summary);
            
            return true;
        } catch (error) {
            console.error('❌ Failed to load Keras model:', error);
            this.useKerasModel = false;
            throw error;
        }
    }

    /**
     * Load and integrate Amazon RNN model
     */
    async loadAmazonModel() {
        try {
            console.log('Integrating Amazon RNN model...');
            
            // Create Amazon model loader
            this.amazonLoader = new AmazonModelLoader();
            
            // Load the Amazon model
            await this.amazonLoader.loadModel();
            
            // Update model parameters to match Amazon model
            this.useKerasModel = true;
            this.vocabSize = this.amazonLoader.vocabSize;
            this.hiddenSize = this.amazonLoader.rnnUnits;
            this.embeddingSize = this.amazonLoader.embeddingDim;
            
            console.log('Amazon model integrated successfully!');
            console.log('Model info:', this.amazonLoader.getModelInfo());
            
            return true;
            
        } catch (error) {
            console.error('Failed to integrate Amazon model:', error);
            this.useKerasModel = false;
            throw error;
        }
    }

    /**
     * Check if Keras model is available
     */
    isKerasModelLoaded() {
        return this.useKerasModel && this.modelLoader.isLoaded;
    }

    /**
     * Enhanced predict method that uses Amazon model when available
     */
    async predictSentiment(sentence) {
        try {
            console.log(`Analyzing sentiment: "${sentence}"`);
            
            let result;
            
            if (this.useKerasModel && this.amazonLoader && this.amazonLoader.isLoaded) {
                // Use Amazon model for prediction
                console.log('Using Amazon RNN model for prediction...');
                result = await this.amazonLoader.predict(sentence);
                
            } else {
                // Fall back to demo model
                console.log('Using demo model for prediction...');
                result = await this.predictWithDemoModel(sentence);
            }
            
            // Store result for visualization
            this.lastPrediction = result;
            
            console.log('Prediction result:', result);
            return result;
            
        } catch (error) {
            console.error('Sentiment prediction failed:', error);
            throw error;
        }
    }

    /**
     * Demo model prediction (existing functionality)
     */
    async predictWithDemoModel(sentence) {
        // Your existing prediction logic here
        const tokens = this.tokenize(sentence);
        const indices = this.tokensToIndices(tokens);
        const embeddings = this.indicesToEmbeddings(indices);
        
        // Simulate processing
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Simple demo prediction
        const positiveWords = ['love', 'great', 'amazing', 'excellent', 'wonderful'];
        const negativeWords = ['hate', 'terrible', 'awful', 'horrible', 'bad'];
        
        const text = sentence.toLowerCase();
        let score = 0.5; // neutral
        
        positiveWords.forEach(word => {
            if (text.includes(word)) score += 0.2;
        });
        
        negativeWords.forEach(word => {
            if (text.includes(word)) score -= 0.2;
        });
        
        score = Math.max(0, Math.min(1, score));
        
        return {
            text: sentence,
            score: score,
            sentiment: score > 0.6 ? 'positive' : score < 0.4 ? 'negative' : 'neutral',
            confidence: Math.abs(score - 0.5) * 2,
            modelType: 'demo'
        };
    }

    /**
     * Initialize a simple vocabulary
     */
    initializeVocabulary() {
        const commonWords = [
            'I', 'love', 'hate', 'this', 'movie', 'film', 'great', 'amazing', 
            'terrible', 'worst', 'best', 'good', 'bad', 'okay', 'fine', 'average',
            'the', 'a', 'an', 'is', 'was', 'are', 'were', 'very', 'really',
            'so', 'too', 'not', 'never', 'always', 'sometimes', 'often',
            'beautiful', 'ugly', 'nice', 'awful', 'wonderful', 'boring', 'exciting'
        ];
        
        // Add special tokens
        this.vocabulary.set('<UNK>', 0);
        this.vocabulary.set('<PAD>', 1);
        this.reverseVocab.set(0, '<UNK>');
        this.reverseVocab.set(1, '<PAD>');
        
        // Add common words
        commonWords.forEach((word, index) => {
            const id = index + 2;
            this.vocabulary.set(word.toLowerCase(), id);
            this.reverseVocab.set(id, word.toLowerCase());
        });
    }
    
    /**
     * Tokenize input sentence
     */
    tokenize(sentence) {
        // Simple tokenization (split by spaces and punctuation)
        const tokens = sentence.toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(token => token.length > 0);
        
        return tokens;
    }
    
    /**
     * Convert tokens to indices
     */
    tokensToIndices(tokens) {
        return tokens.map(token => {
            return this.vocabulary.get(token) || this.vocabulary.get('<UNK>');
        });
    }
    
    /**
     * Convert token indices to embeddings
     */
    indicesToEmbeddings(indices) {
        return indices.map(index => {
            // Simple embedding: random vector based on word index (for demo)
            return this.getEmbedding(index);
        });
    }
    
    /**
     * Get embedding vector for a word index
     */
    getEmbedding(wordIndex) {
        // Create deterministic "embeddings" based on word index
        const embedding = new Array(this.embeddingSize);
        const seed = wordIndex * 123; // Simple seed for deterministic random
        
        for (let i = 0; i < this.embeddingSize; i++) {
            // Pseudo-random values based on word index
            embedding[i] = Math.sin(seed + i) * 0.5;
        }
        
        return embedding;
    }
    
    /**
     * Forward pass through RNN
     */
    forward(embeddings) {
        this.computationHistory = [];
        let hiddenState = new Array(this.hiddenSize).fill(0);
        const outputs = [];
        
        embeddings.forEach((embedding, t) => {
            // h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)
            const inputContribution = this.matrixVectorMultiply(this.weights.Wxh, embedding);
            const hiddenContribution = this.matrixVectorMultiply(this.weights.Whh, hiddenState);
            
            const combined = this.vectorAdd(
                this.vectorAdd(inputContribution, hiddenContribution),
                this.weights.bh
            );
            
            hiddenState = combined.map(x => Math.tanh(x));
            
            // Store computation for visualization
            this.computationHistory.push({
                timeStep: t,
                input: [...embedding],
                hiddenState: [...hiddenState],
                inputContribution: [...inputContribution],
                hiddenContribution: [...hiddenContribution],
                combined: [...combined]
            });
        });
        
        // Final output: y = softmax(Why * h_final + by)
        const finalOutput = this.vectorAdd(
            this.matrixVectorMultiply(this.weights.Why, hiddenState),
            this.weights.by
        );
        
        const probabilities = this.softmax(finalOutput);
        
        return {
            hiddenStates: this.computationHistory.map(step => step.hiddenState),
            finalOutput: probabilities,
            computationHistory: this.computationHistory
        };
    }
    
    /**
     * Predict sentiment for a sentence
     */
    async predictSentiment(sentence) {
        // Check Amazon model first (highest priority)
        if (this.amazonLoader && this.amazonLoader.isLoaded) {
            try {
                console.log('Using Amazon RNN model for prediction...');
                const result = await this.amazonLoader.predict(sentence);
                return result;
            } catch (error) {
                console.warn('Amazon model prediction failed, falling back:', error);
            }
        }
        
        // Check Keras model second
        if (this.useKerasModel && this.modelLoader.isLoaded) {
            try {
                console.log('Using Keras model for prediction...');
                const result = await this.modelLoader.getVisualizationData(sentence);
                return result;
            } catch (error) {
                console.warn('Keras model prediction failed, falling back to demo model:', error);
            }
        }
        
        // Fall back to demo model
        console.log('Using demo model for prediction...');
        return this.predictSentimentDemo(sentence);
    }

    /**
     * Demo prediction method with enhanced sentiment analysis
     */
    predictSentimentDemo(sentence) {
        const tokens = this.tokenize(sentence);
        const indices = this.tokensToIndices(tokens);
        const embeddings = this.indicesToEmbeddings(indices);

        const result = this.forward(embeddings);
        
        // Enhanced sentiment analysis for demo
        const enhancedSentiment = this.analyzeTextSentimentDemo(sentence);
        
        const sentimentLabels = ['Positive', 'Neutral', 'Negative'];
        
        // Use enhanced sentiment analysis instead of random forward pass
        let probabilities;
        let predictedClass;
        
        if (enhancedSentiment.sentiment === 'positive') {
            predictedClass = 0;
            probabilities = [enhancedSentiment.confidence, (1 - enhancedSentiment.confidence) / 2, (1 - enhancedSentiment.confidence) / 2];
        } else if (enhancedSentiment.sentiment === 'negative') {
            predictedClass = 2;
            probabilities = [(1 - enhancedSentiment.confidence) / 2, (1 - enhancedSentiment.confidence) / 2, enhancedSentiment.confidence];
        } else {
            predictedClass = 1;
            probabilities = [(1 - enhancedSentiment.confidence) / 2, enhancedSentiment.confidence, (1 - enhancedSentiment.confidence) / 2];
        }

        return {
            tokens: tokens,
            indices: indices,
            embeddings: embeddings,
            probabilities: probabilities,
            predictedClass: predictedClass,
            predictedLabel: sentimentLabels[predictedClass],
            confidence: enhancedSentiment.confidence,
            hiddenStates: result.hiddenStates,
            computationHistory: result.computationHistory,
            score: enhancedSentiment.score,
            modelType: 'demo'
        };
    }
    
    /**
     * Enhanced sentiment analysis for demo model
     */
    analyzeTextSentimentDemo(text) {
        const positiveWords = [
            'love', 'like', 'great', 'amazing', 'excellent', 'wonderful', 'fantastic', 
            'awesome', 'brilliant', 'perfect', 'good', 'nice', 'beautiful', 'happy',
            'best', 'favorite', 'outstanding', 'superb', 'magnificent'
        ];
        
        const negativeWords = [
            'hate', 'dislike', 'terrible', 'awful', 'horrible', 'bad', 'worst',
            'disgusting', 'annoying', 'disappointed', 'sad', 'boring', 'dull', 
            'poor', 'weak', 'pathetic', 'useless', 'stupid', 'waste', 'disaster'
        ];
        
        // Strong patterns
        const negativePatterns = [
            /worst.*ever/i, /hate.*this/i, /terrible.*movie/i, /awful.*film/i
        ];
        
        const positivePatterns = [
            /best.*ever/i, /love.*this/i, /amazing.*movie/i, /fantastic.*film/i
        ];
        
        let score = 0.5;
        
        // Check for strong patterns first
        for (const pattern of negativePatterns) {
            if (pattern.test(text)) {
                return { sentiment: 'negative', confidence: 0.9, score: 0.1 };
            }
        }
        
        for (const pattern of positivePatterns) {
            if (pattern.test(text)) {
                return { sentiment: 'positive', confidence: 0.9, score: 0.9 };
            }
        }
        
        // Word analysis
        const words = text.toLowerCase().replace(/[^a-zA-Z\s]/g, '').split(/\s+/);
        let positiveCount = 0;
        let negativeCount = 0;
        
        for (const word of words) {
            if (positiveWords.includes(word)) positiveCount++;
            if (negativeWords.includes(word)) negativeCount++;
        }
        
        if (positiveCount > negativeCount) {
            score = 0.7 + (positiveCount * 0.1);
            return { sentiment: 'positive', confidence: Math.min(0.95, score), score };
        } else if (negativeCount > positiveCount) {
            score = 0.3 - (negativeCount * 0.1);
            return { sentiment: 'negative', confidence: Math.min(0.95, 1 - score), score: Math.max(0.05, score) };
        } else {
            return { sentiment: 'neutral', confidence: 0.65, score: 0.5 };
        }
    }    /**
     * Simulate backpropagation through time
     */
    async backpropagateDemo(sentence, targetSentiment) {
        const prediction = await this.predictSentiment(sentence);
        const timeSteps = prediction.tokens.length;
        
        // Calculate loss (cross-entropy)
        const targetVector = new Array(3).fill(0);
        targetVector[targetSentiment] = 1;
        
        const loss = this.crossEntropyLoss(prediction.probabilities, targetVector);
        
        // Simulate gradient flow (simplified for visualization)
        const gradients = {
            outputGradients: this.vectorSubtract(prediction.probabilities, targetVector),
            hiddenGradients: [],
            inputGradients: []
        };
        
        // Backward pass through time
        let deltaH = this.matrixVectorMultiply(
            this.transpose(this.weights.Why), 
            gradients.outputGradients
        );
        
        for (let t = timeSteps - 1; t >= 0; t--) {
            const currentState = prediction.computationHistory[t];
            
            // Gradient of tanh
            const tanhGrad = currentState.hiddenState.map(h => 1 - h * h);
            deltaH = this.vectorMultiply(deltaH, tanhGrad);
            
            gradients.hiddenGradients.unshift([...deltaH]);
            
            // Input gradients
            const inputGrad = this.matrixVectorMultiply(this.transpose(this.weights.Wxh), deltaH);
            gradients.inputGradients.unshift(inputGrad);
            
            // Propagate to previous time step
            if (t > 0) {
                deltaH = this.matrixVectorMultiply(this.transpose(this.weights.Whh), deltaH);
            }
        }
        
        return {
            loss: loss,
            gradients: gradients,
            prediction: prediction
        };
    }
    
    // Utility functions for matrix operations
    randomMatrix(rows, cols) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix[i] = [];
            for (let j = 0; j < cols; j++) {
                matrix[i][j] = (Math.random() - 0.5) * 0.1; // Small random weights
            }
        }
        return matrix;
    }
    
    randomVector(size) {
        return new Array(size).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    }
    
    matrixVectorMultiply(matrix, vector) {
        return matrix.map(row => 
            row.reduce((sum, val, i) => sum + val * vector[i], 0)
        );
    }
    
    vectorAdd(a, b) {
        return a.map((val, i) => val + b[i]);
    }
    
    vectorSubtract(a, b) {
        return a.map((val, i) => val - b[i]);
    }
    
    vectorMultiply(a, b) {
        return a.map((val, i) => val * b[i]);
    }
    
    transpose(matrix) {
        return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
    }
    
    softmax(vector) {
        const max = Math.max(...vector);
        const exp = vector.map(x => Math.exp(x - max));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sum);
    }
    
    crossEntropyLoss(predicted, target) {
        return -target.reduce((sum, t, i) => sum + t * Math.log(predicted[i] + 1e-15), 0);
    }
    
    /**
     * Get educational explanations for each step
     */
    getStepExplanation(step) {
        const explanations = {
            tokenize: {
                title: "Tokenization",
                description: "Breaking down the input sentence into individual words or tokens. Each token will be processed sequentially by the RNN.",
                equation: "sentence → [token₁, token₂, ..., tokenₙ]"
            },
            embed: {
                title: "Word Embeddings",
                description: "Converting each token into a dense vector representation. These embeddings capture semantic meaning of words in a continuous space.",
                equation: "token → embedding_vector ∈ ℝᵈ"
            },
            rnn: {
                title: "RNN Processing",
                description: "The RNN processes each word sequentially, updating its hidden state at each time step. The hidden state acts as the network's memory.",
                equation: "hₜ = tanh(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁ + bₕ)"
            },
            predict: {
                title: "Output Prediction",
                description: "Using the final hidden state to predict sentiment classes. The softmax function converts raw scores to probabilities.",
                equation: "y = softmax(Wₕᵧ·hₜ + bᵧ)"
            },
            backprop: {
                title: "Backpropagation Through Time",
                description: "Computing gradients by unrolling the RNN and propagating errors backward through all time steps to update weights.",
                equation: "∂L/∂W = Σₜ ∂L/∂hₜ · ∂hₜ/∂W"
            }
        };
        
        return explanations[step] || { title: "Unknown Step", description: "", equation: "" };
    }
}

// Export for use in other modules
window.RNNModel = RNNModel;
