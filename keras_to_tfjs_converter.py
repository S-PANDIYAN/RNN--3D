"""
Keras to TensorFlow.js Model Converter
This script helps convert your Keras sentiment analysis model to TensorFlow.js format
for use in the 3D RNN visualization web application.

Requirements:
- tensorflow
- tensorflowjs

Install with: pip install tensorflow tensorflowjs
"""

import tensorflow as tf
import tensorflowjs as tfjs
import json
import pickle
import argparse
import os
from pathlib import Path

def convert_keras_model(model_path, output_dir, tokenizer_path=None):
    """
    Convert Keras model to TensorFlow.js format
    
    Args:
        model_path: Path to the Keras model file (.h5 or SavedModel directory)
        output_dir: Output directory for TensorFlow.js model
        tokenizer_path: Optional path to tokenizer pickle file
    """
    
    print(f"Loading Keras model from: {model_path}")
    
    # Load the model
    try:
        if model_path.endswith('.h5') or model_path.endswith('.keras'):
            model = tf.keras.models.load_model(model_path)
        else:
            model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to TensorFlow.js
    print(f"\nConverting model to TensorFlow.js format...")
    try:
        tfjs.converters.save_keras_model(model, output_dir)
        print(f"✅ Model converted successfully to: {output_dir}")
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return False
    
    # Handle tokenizer if provided
    if tokenizer_path and os.path.exists(tokenizer_path):
        print(f"\nProcessing tokenizer from: {tokenizer_path}")
        try:
            # Load tokenizer
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            
            # Extract tokenizer configuration
            tokenizer_config = {
                'word_index': getattr(tokenizer, 'word_index', {}),
                'vocab_size': len(getattr(tokenizer, 'word_index', {})) + 1,
                'max_sequence_length': getattr(tokenizer, 'maxlen', 100),
                'oov_token': getattr(tokenizer, 'oov_token', '<UNK>'),
                'filters': getattr(tokenizer, 'filters', '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'),
                'lower': getattr(tokenizer, 'lower', True),
                'split': getattr(tokenizer, 'split', ' ')
            }
            
            # Save tokenizer configuration
            tokenizer_output_path = os.path.join(output_dir, 'tokenizer.json')
            with open(tokenizer_output_path, 'w') as f:
                json.dump(tokenizer_config, f, indent=2)
            
            print(f"✅ Tokenizer configuration saved to: {tokenizer_output_path}")
            print(f"   Vocabulary size: {tokenizer_config['vocab_size']}")
            print(f"   Max sequence length: {tokenizer_config['max_sequence_length']}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not process tokenizer: {e}")
            print("   You can still use the model with the built-in simple tokenizer")
    
    # Create usage instructions
    instructions_path = os.path.join(output_dir, 'usage_instructions.md')
    with open(instructions_path, 'w') as f:
        f.write(f"""# Model Usage Instructions

## Files Generated:
- `model.json`: Model architecture and weights metadata
- `group1-shard1of1.bin`: Model weights
- `tokenizer.json`: Tokenizer configuration (if provided)

## How to use in the 3D RNN Visualizer:

1. Open the 3D RNN Visualization web application
2. Select "Load Keras Model" radio button
3. Upload the `model.json` file as the model file
4. Upload the `tokenizer.json` file as the tokenizer file (optional)
5. Click "Load Model"
6. Enter sentences to analyze!

## Model Details:
- Input shape: {model.input_shape}
- Output shape: {model.output_shape}
- Total parameters: {model.count_params():,}
- Layers: {len(model.layers)}

## Example sentences to try:
- "This movie is absolutely amazing!"
- "The product quality is terrible"
- "It's an okay experience, nothing special"
- "I love this so much!"
- "Worst purchase I've ever made"

## Troubleshooting:
- Make sure both model.json and the .bin files are in the same directory
- Check browser console for any loading errors
- Ensure your model has compatible input/output shapes for sentiment analysis
""")
    
    print(f"\n📋 Usage instructions saved to: {instructions_path}")
    print(f"\n🎉 Conversion completed successfully!")
    print(f"Upload the files in '{output_dir}' to your web application.")
    
    return True

def create_sample_model_and_data():
    """
    Create a sample sentiment analysis model for demonstration
    """
    print("Creating sample sentiment analysis model...")
    
    # Model parameters
    vocab_size = 1000
    embedding_dim = 50
    max_length = 100
    
    # Create a simple LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: negative, neutral, positive
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the model
    model.save('sample_sentiment_model.h5')
    print("✅ Sample model saved as 'sample_sentiment_model.h5'")
    
    # Create sample tokenizer configuration
    sample_tokenizer = {
        'word_index': {
            'the': 1, 'and': 2, 'a': 3, 'to': 4, 'of': 5, 'i': 6, 'it': 7, 'in': 8, 'you': 9, 'is': 10,
            'movie': 11, 'film': 12, 'good': 13, 'bad': 14, 'great': 15, 'terrible': 16, 'love': 17, 'hate': 18,
            'amazing': 19, 'awful': 20, 'excellent': 21, 'horrible': 22, 'wonderful': 23, 'disappointing': 24,
            'best': 25, 'worst': 26, 'fantastic': 27, 'boring': 28, 'interesting': 29, 'exciting': 30
        },
        'vocab_size': 1000,
        'max_sequence_length': 100,
        'oov_token': '<UNK>',
        'filters': '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        'lower': True,
        'split': ' '
    }
    
    with open('sample_tokenizer.json', 'w') as f:
        json.dump(sample_tokenizer, f, indent=2)
    print("✅ Sample tokenizer saved as 'sample_tokenizer.json'")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert Keras model to TensorFlow.js')
    parser.add_argument('--model', '-m', help='Path to Keras model file (.h5)')
    parser.add_argument('--output', '-o', default='./tfjs_model', help='Output directory for TensorFlow.js model')
    parser.add_argument('--tokenizer', '-t', help='Path to tokenizer pickle file')
    parser.add_argument('--sample', action='store_true', help='Create a sample model for testing')
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_model_and_data()
        print("\nTo convert the sample model, run:")
        print("python keras_to_tfjs_converter.py --model sample_sentiment_model.h5 --output tfjs_model")
        return
    
    if not args.model:
        print("Please provide a model file with --model or use --sample to create a test model")
        print("Usage: python keras_to_tfjs_converter.py --model your_model.h5 --output ./tfjs_model")
        return
    
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return
    
    success = convert_keras_model(args.model, args.output, args.tokenizer)
    
    if success:
        print(f"\n🎯 Next steps:")
        print(f"1. Copy the files from '{args.output}' to your web server")
        print(f"2. Open the 3D RNN Visualizer in your browser")
        print(f"3. Select 'Load Keras Model' and upload the model.json file")
        print(f"4. Upload tokenizer.json if available")
        print(f"5. Start analyzing sentiment with real predictions!")

if __name__ == "__main__":
    main()
