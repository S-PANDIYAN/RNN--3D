# 🧠 3D RNN Sentiment Analysis Visualizer

An interactive 3D educational web application that visualizes how Recurrent Neural Networks (RNNs) process natural language for sentiment analysis. Perfect for students learning deep learning, NLP, and RNNs!

## ✨ Features

### 🎓 Educational Visualization
- **Tokenization**: See how sentences are broken into individual tokens
- **Word Embeddings**: Visualize how words are converted to vector representations
- **RNN Processing**: Watch the sequential processing through RNN cells
- **Hidden States**: Observe how hidden states evolve over time
- **Sentiment Prediction**: See final classification with probabilities
- **Backpropagation**: Understand gradient flow through time

### 🎮 Interactive 3D Experience
- **3D Visualization**: Full Three.js powered 3D environment
- **Real-time Animation**: Smooth animations showing data flow
- **Interactive Controls**: Orbit controls for exploring the 3D scene
- **Step-by-step Mode**: Process each phase individually
- **Speed Control**: Adjust animation speed

### 🤖 Model Support
- **Demo Model**: Built-in simplified RNN for educational purposes
- **Keras Integration**: Load your own trained TensorFlow/Keras models
- **Real Predictions**: Use actual trained models for sentiment analysis

## 🚀 Quick Start

### Option 1: Demo Mode (No Setup Required)
1. Open `index.html` in a modern web browser
2. Enter a sentence like "I love this movie!"
3. Click "🚀 Process Sentence"
4. Watch the 3D visualization!

### Option 2: With Your Keras Model
1. Convert your Keras model to TensorFlow.js format (see below)
2. Open the web application
3. Select "Load Keras Model"
4. Upload your model files
5. Start analyzing with real predictions!

## 📁 Project Structure

```
RNN--3D/
├── index.html                 # Main application page
├── styles.css                # Application styling
├── js/
│   ├── app.js                # Main application logic
│   ├── rnn-model.js          # RNN model implementation
│   ├── 3d-visualization.js   # 3D rendering and animations
│   ├── ui-controller.js      # User interface management
│   └── model-loader.js       # Keras model loading
├── keras_to_tfjs_converter.py # Python script for model conversion
└── README.md                 # This file
```

## 🔧 Converting Your Keras Model

### Prerequisites
```bash
pip install tensorflow tensorflowjs
```

### Method 1: Using the Provided Converter Script
```bash
# Convert your Keras model
python keras_to_tfjs_converter.py --model your_model.h5 --output ./tfjs_model --tokenizer your_tokenizer.pkl

# Create a sample model for testing
python keras_to_tfjs_converter.py --sample
```

### Method 2: Manual Conversion
```python
import tensorflowjs as tfjs

# Convert Keras model to TensorFlow.js
tfjs.converters.save_keras_model(your_keras_model, 'tfjs_model_directory')
```

### Expected Model Format
Your Keras model should:
- Accept integer sequences (tokenized text) as input
- Output probabilities for sentiment classes
- Be trained for sentiment analysis (binary or 3-class)

## 📝 Usage Instructions

### Loading Your Model
1. **Convert**: Use the converter script to convert your Keras model
2. **Upload**: In the web app, select "Load Keras Model"
3. **Files**: Upload the `model.json` file and optionally `tokenizer.json`
4. **Load**: Click "📂 Load Model" and wait for confirmation

### Using the Visualizer
1. **Enter Text**: Type a sentence in the input field
2. **Process**: Click "🚀 Process Sentence" for full animation
3. **Step Mode**: Use individual step buttons for detailed exploration
4. **Speed**: Adjust animation speed with the slider
5. **Reset**: Click "🔄 Reset" to start over

### Demo Sentences
Try these example sentences:
- "I love this movie!" (Positive)
- "This film is terrible" (Negative) 
- "The movie was okay" (Neutral)
- "Amazing performance by the actors" (Positive)
- "Worst movie I've ever seen" (Negative)

## 🎯 Educational Concepts Covered

### 1. **Tokenization** 🔤
- Breaking sentences into individual words/tokens
- Handling punctuation and special characters
- Vocabulary mapping

### 2. **Word Embeddings** 📊
- Converting tokens to dense vector representations
- Semantic meaning in continuous space
- Dimensionality and vector operations

### 3. **RNN Processing** 🔄
- Sequential data processing
- Hidden state updates: `h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)`
- Temporal dependencies and memory

### 4. **Sentiment Classification** 🎭
- Final prediction: `y = softmax(W_hy * h_t + b_y)`
- Probability distributions
- Multi-class classification

### 5. **Backpropagation Through Time** ⏮️
- Gradient computation: `∂L/∂W = Σ_t ∂L/∂h_t * ∂h_t/∂W`
- Unrolling RNN through time steps
- Weight updates and learning

## 🛠️ Technical Details

### Technologies Used
- **Three.js**: 3D graphics and animations
- **TensorFlow.js**: Model loading and inference
- **HTML5/CSS3**: Modern web standards
- **Vanilla JavaScript**: No framework dependencies

### Browser Requirements
- Modern browser with WebGL support
- JavaScript enabled
- Minimum 2GB RAM recommended for smooth animations

### Performance Tips
- Close other browser tabs for better performance
- Reduce animation speed if experiencing lag
- Use Chrome or Firefox for best WebGL performance

## 🔍 Troubleshooting

### Model Loading Issues
- **File Format**: Ensure model is in TensorFlow.js format (.json + .bin files)
- **CORS**: If loading from file://, use a local server (e.g., `python -m http.server`)
- **Memory**: Large models may require more RAM

### Visualization Issues
- **WebGL**: Check if WebGL is enabled in your browser
- **Performance**: Try reducing animation speed or closing other applications
- **Controls**: Use mouse to orbit around the 3D scene

### Common Errors
- **"Model not loaded"**: Ensure model files are correctly formatted
- **"Prediction failed"**: Check input text length and model compatibility
- **"OrbitControls not found"**: Check internet connection for CDN resources

## 🎓 Educational Use Cases

### For Students
- **Understanding RNNs**: Visual representation of sequential processing
- **NLP Concepts**: Tokenization, embeddings, and classification
- **Deep Learning**: Backpropagation and gradient flow

### For Educators
- **Interactive Demos**: Engage students with 3D visualizations
- **Step-by-step Learning**: Break down complex concepts
- **Real Models**: Connect theory with practical applications

### For Researchers
- **Model Debugging**: Visualize model behavior
- **Concept Explanation**: Communicate research to broader audiences
- **Educational Outreach**: Make AI/ML more accessible

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Bug Reports**: Found an issue? Please report it!
2. **Feature Requests**: Ideas for new visualizations?
3. **Model Integration**: Help support more model types
4. **Documentation**: Improve explanations and examples

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Three.js** community for excellent 3D graphics library
- **TensorFlow.js** team for making ML accessible in browsers
- **Educational AI** community for inspiration and feedback

## 📞 Support

Having issues? Here's how to get help:

1. **Check the troubleshooting section** above
2. **Look at browser console** for error messages
3. **Verify model format** using the converter script
4. **Try the demo mode** first to ensure basic functionality

---

**Happy Learning! 🎉**

Transform your understanding of RNNs with this interactive 3D visualization tool!
