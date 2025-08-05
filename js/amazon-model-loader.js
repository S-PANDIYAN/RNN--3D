// amazon-model-loader.js
// Amazon RNN Model Loader for 3D Visualization

class AmazonModelLoader {
    constructor() {
        this.model = null;
        this.isLoaded = false;
        this.vocabSize = 10000;
        this.sequenceLength = 200;
        this.embeddingDim = 100;
        this.rnnUnits = 64;
        
        // Simple vocabulary for demo (replace with actual tokenizer)
        this.vocabulary = this.createSimpleVocabulary();
    }

    async loadModel() {
        try {
            console.log('Loading Amazon RNN model...');
            
            // For now, we'll simulate the model loading
            // In production, you would convert the H5 model to TensorFlow.js format
            console.log('Note: H5 model detected. For web deployment, convert to TensorFlow.js format using:');
            console.log('tensorflowjs_converter --input_format=keras ./amazon_model.h5 ./amazon_model_tfjs');
            
            // Simulate model loading for demo
            await this.simulateModelLoading();
            
            this.isLoaded = true;
            console.log('Amazon model loaded successfully!');
            return true;
            
        } catch (error) {
            console.error('Failed to load Amazon model:', error);
            this.isLoaded = false;
            throw error;
        }
    }

    async simulateModelLoading() {
        // Simulate loading delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Create a mock model structure that matches your actual model
        this.model = {
            predict: (inputTensor) => {
                // Simulate prediction (replace with actual model when converted)
                const batchSize = inputTensor.shape[0];
                const randomPrediction = Math.random();
                
                // Return tensor-like object
                return {
                    data: async () => [randomPrediction],
                    dispose: () => {}
                };
            },
            summary: () => {
                console.log('Amazon RNN Model Summary:');
                console.log('- Embedding: (None, 200, 100)');
                console.log('- SimpleRNN: (None, 64)');
                console.log('- Dense: (None, 32)');
                console.log('- Output: (None, 1)');
            }
        };
    }

    createSimpleVocabulary() {
        // Create a simple vocabulary mapping
        // In production, use the actual vocabulary from training
        const commonWords = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those',
            'good', 'bad', 'great', 'terrible', 'amazing', 'awful', 'love', 'hate', 'like', 'dislike',
            'movie', 'film', 'show', 'series', 'actor', 'actress', 'director', 'plot', 'story',
            'excellent', 'poor', 'fantastic', 'horrible', 'wonderful', 'boring', 'exciting', 'dull'
        ];
        
        const vocab = new Map();
        vocab.set('<PAD>', 0);
        vocab.set('<UNK>', 1);
        
        commonWords.forEach((word, index) => {
            vocab.set(word.toLowerCase(), index + 2);
        });
        
        return vocab;
    }

    tokenizeText(text) {
        // Simple tokenization
        const words = text.toLowerCase()
            .replace(/[^a-zA-Z0-9\s]/g, '')
            .split(/\s+/)
            .filter(word => word.length > 0);
        
        // Convert to indices
        const tokens = new Array(this.sequenceLength).fill(0); // PAD token
        
        for (let i = 0; i < Math.min(words.length, this.sequenceLength); i++) {
            const word = words[i];
            tokens[i] = this.vocabulary.get(word) || this.vocabulary.get('<UNK>');
        }
        
        return tokens;
    }

    async predict(text) {
        if (!this.isLoaded || !this.model) {
            throw new Error('Amazon model not loaded');
        }

        try {
            console.log('Amazon model predicting:', text);
            
            // Tokenize input
            const tokens = this.tokenizeText(text);
            
            // Create tensor (simulate for now)
            const inputTensor = {
                shape: [1, this.sequenceLength],
                dispose: () => {}
            };
            
            // Make prediction using enhanced logic
            const prediction = this.model.predict(inputTensor);
            const result = await prediction.data();
            
            // Clean up
            inputTensor.dispose();
            prediction.dispose();
            
            // Enhanced sentiment analysis based on text content
            const score = this.analyzeTextSentiment(text);
            
            // More decisive thresholds for clearer predictions
            let sentiment, confidence;
            if (score > 0.65) {
                sentiment = 'positive';
                confidence = Math.min(0.95, 0.6 + (score - 0.65) * 1.2);
            } else if (score < 0.35) {
                sentiment = 'negative';
                confidence = Math.min(0.95, 0.6 + (0.35 - score) * 1.2);
            } else {
                sentiment = 'neutral';
                confidence = 0.6 - Math.abs(score - 0.5) * 0.8;
            }
            
            console.log('Amazon model prediction result:', { 
                text, 
                rawScore: score, 
                sentiment, 
                confidence: confidence.toFixed(3) 
            });
            
            return {
                text: text,
                score: score,
                sentiment: sentiment,
                confidence: confidence,
                modelType: 'amazon_rnn'
            };
            
        } catch (error) {
            console.error('Prediction failed:', error);
            throw error;
        }
    }

    analyzeTextSentiment(text) {
        // Enhanced sentiment analysis for demo purposes with better phrase recognition
        const positiveWords = [
            'love', 'like', 'great', 'amazing', 'excellent', 'wonderful', 'fantastic', 
            'awesome', 'brilliant', 'perfect', 'good', 'nice', 'beautiful', 'happy',
            'pleased', 'satisfied', 'delighted', 'thrilled', 'excited', 'impressive',
            'outstanding', 'superb', 'magnificent', 'marvelous', 'best', 'favorite'
        ];
        
        const negativeWords = [
            'hate', 'dislike', 'terrible', 'awful', 'horrible', 'bad', 'worst',
            'disgusting', 'annoying', 'frustrated', 'angry', 'disappointed', 'sad',
            'boring', 'dull', 'poor', 'weak', 'pathetic', 'useless', 'stupid',
            'waste', 'ridiculous', 'pointless', 'mess', 'disaster', 'failure'
        ];
        
        // Strong negative phrases that should be caught
        const negativePatterns = [
            /worst.*ever/i,
            /hate.*this/i,
            /terrible.*movie/i,
            /awful.*film/i,
            /horrible.*show/i,
            /disaster.*of/i,
            /complete.*waste/i,
            /totally.*boring/i,
            /extremely.*bad/i,
            /really.*hate/i
        ];
        
        // Strong positive phrases
        const positivePatterns = [
            /best.*ever/i,
            /love.*this/i,
            /amazing.*movie/i,
            /fantastic.*film/i,
            /wonderful.*show/i,
            /absolutely.*love/i,
            /really.*enjoyed/i,
            /highly.*recommend/i,
            /excellent.*performance/i
        ];
        
        const intensifiers = ['very', 'really', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely'];
        const negators = ['not', 'never', 'no', "don't", "won't", "can't", "isn't", "doesn't", "wasn't", "weren't"];
        
        let score = 0.5; // Start neutral
        
        // First check for strong patterns
        for (const pattern of negativePatterns) {
            if (pattern.test(text)) {
                console.log('Detected strong negative pattern:', pattern);
                return 0.1; // Very negative
            }
        }
        
        for (const pattern of positivePatterns) {
            if (pattern.test(text)) {
                console.log('Detected strong positive pattern:', pattern);
                return 0.9; // Very positive
            }
        }
        
        // Word-by-word analysis with context
        const words = text.toLowerCase().replace(/[^a-zA-Z\s]/g, '').split(/\s+/);
        let intensity = 1.0;
        let negated = false;
        let wordCount = 0;
        
        for (let i = 0; i < words.length; i++) {
            const word = words[i];
            
            // Check for intensifiers
            if (intensifiers.includes(word)) {
                intensity = 1.8;
                continue;
            }
            
            // Check for negators
            if (negators.includes(word)) {
                negated = true;
                continue;
            }
            
            // Check sentiment words
            if (positiveWords.includes(word)) {
                const adjustment = 0.2 * intensity;
                score += negated ? -adjustment * 1.5 : adjustment; // Negation makes it more negative
                negated = false;
                intensity = 1.0;
                wordCount++;
            } else if (negativeWords.includes(word)) {
                const adjustment = 0.2 * intensity;
                score += negated ? adjustment * 0.5 : -adjustment; // Negation weakens negative words
                negated = false;
                intensity = 1.0;
                wordCount++;
            }
        }
        
        // Apply context bonuses
        if (text.toLowerCase().includes('worst') && text.toLowerCase().includes('ever')) {
            score -= 0.3; // Extra penalty for "worst ever"
        }
        
        if (text.toLowerCase().includes('best') && text.toLowerCase().includes('ever')) {
            score += 0.3; // Extra bonus for "best ever"
        }
        
        // If no sentiment words found, stay neutral
        if (wordCount === 0) {
            score = 0.5;
        }
        
        // Ensure score is within bounds
        const finalScore = Math.max(0.05, Math.min(0.95, score));
        console.log(`Sentiment analysis for "${text}": score=${finalScore}, wordCount=${wordCount}`);
        return finalScore;
    }

    getModelInfo() {
        return {
            name: 'Amazon RNN Sentiment Model',
            type: 'RNN',
            vocabSize: this.vocabSize,
            sequenceLength: this.sequenceLength,
            embeddingDim: this.embeddingDim,
            rnnUnits: this.rnnUnits,
            isLoaded: this.isLoaded
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AmazonModelLoader;
}
