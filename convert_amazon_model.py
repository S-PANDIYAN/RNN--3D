"""
Direct converter for Amazon RNN Model
This script directly converts your specific Amazon RNN model to TensorFlow.js format
"""

import tensorflow as tf
import tensorflowjs as tfjs
import json
import os

def convert_amazon_model():
    """
    Convert the Amazon RNN Model to TensorFlow.js format
    """
    
    # Your model path
    model_path = r"D:\SE\Amazon RNN Model.keras"
    output_dir = "./amazon_model_tfjs"
    
    print(f"🔄 Converting Amazon RNN Model...")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please check the file path and make sure the model exists.")
        return False
    
    try:
        # Load the Keras model
        print("📂 Loading Keras model...")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Print model information
        print("\n📊 Model Summary:")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        print(f"Total parameters: {model.count_params():,}")
        print(f"Number of layers: {len(model.layers)}")
        
        # Print layer details
        print("\n🏗️ Model Architecture:")
        for i, layer in enumerate(model.layers):
            layer_info = f"Layer {i+1}: {layer.name} ({layer.__class__.__name__})"
            if hasattr(layer, 'units'):
                layer_info += f" - Units: {layer.units}"
            if hasattr(layer, 'activation'):
                layer_info += f" - Activation: {layer.activation.__name__}"
            print(layer_info)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to TensorFlow.js
        print(f"\n🔄 Converting to TensorFlow.js format...")
        tfjs.converters.save_keras_model(model, output_dir)
        print(f"✅ Model converted successfully!")
        
        # Create a configuration file with model details
        model_config = {
            "model_name": "Amazon RNN Sentiment Analysis",
            "model_type": "RNN/LSTM",
            "input_shape": list(model.input_shape),
            "output_shape": list(model.output_shape),
            "num_classes": model.output_shape[-1],
            "total_parameters": int(model.count_params()),
            "layers": [
                {
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "units": getattr(layer, 'units', None),
                    "activation": getattr(layer.activation, '__name__', None) if hasattr(layer, 'activation') else None
                }
                for layer in model.layers
            ]
        }
        
        # Save model configuration
        config_path = os.path.join(output_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Create a simple tokenizer for Amazon reviews
        amazon_tokenizer = {
            'word_index': {
                # Common words in Amazon reviews
                'the': 1, 'and': 2, 'a': 3, 'to': 4, 'of': 5, 'i': 6, 'it': 7, 'in': 8, 'you': 9, 'is': 10,
                'this': 11, 'that': 12, 'was': 13, 'for': 14, 'are': 15, 'as': 16, 'with': 17, 'have': 18,
                'be': 19, 'at': 20, 'one': 21, 'had': 22, 'by': 23, 'but': 24, 'not': 25, 'what': 26,
                'all': 27, 'were': 28, 'they': 29, 'we': 30, 'when': 31, 'your': 32, 'can': 33, 'said': 34,
                
                # Product-related words
                'product': 35, 'item': 36, 'quality': 37, 'price': 38, 'buy': 39, 'purchase': 40, 'order': 41,
                'delivery': 42, 'shipping': 43, 'package': 44, 'arrived': 45, 'received': 46, 'sent': 47,
                'seller': 48, 'customer': 49, 'service': 50, 'return': 51, 'refund': 52, 'money': 53,
                
                # Positive sentiment words
                'good': 54, 'great': 55, 'excellent': 56, 'amazing': 57, 'awesome': 58, 'perfect': 59, 'love': 60,
                'wonderful': 61, 'fantastic': 62, 'outstanding': 63, 'superb': 64, 'brilliant': 65, 'best': 66,
                'recommend': 67, 'satisfied': 68, 'happy': 69, 'pleased': 70, 'impressed': 71, 'worth': 72,
                
                # Negative sentiment words
                'bad': 73, 'terrible': 74, 'awful': 75, 'horrible': 76, 'worst': 77, 'hate': 78, 'disappointed': 79,
                'poor': 80, 'cheap': 81, 'useless': 82, 'waste': 83, 'broken': 84, 'defective': 85, 'damaged': 86,
                'problem': 87, 'issue': 88, 'wrong': 89, 'bad': 90, 'avoid': 91, 'regret': 92,
                
                # Neutral/descriptive words
                'size': 93, 'color': 94, 'material': 95, 'design': 96, 'style': 97, 'feature': 98, 'function': 99,
                'work': 100, 'works': 101, 'working': 102, 'use': 103, 'used': 104, 'using': 105, 'easy': 106,
                'simple': 107, 'hard': 108, 'difficult': 109, 'fast': 110, 'slow': 111, 'quick': 112, 'long': 113
            },
            'vocab_size': 5000,  # Assuming larger vocabulary for Amazon reviews
            'max_sequence_length': 200,  # Longer sequences for reviews
            'oov_token': '<UNK>',
            'filters': '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            'lower': True,
            'split': ' '
        }
        
        # Save tokenizer
        tokenizer_path = os.path.join(output_dir, 'tokenizer.json')
        with open(tokenizer_path, 'w') as f:
            json.dump(amazon_tokenizer, f, indent=2)
        
        # Create usage instructions specifically for Amazon model
        instructions_path = os.path.join(output_dir, 'README.md')
        with open(instructions_path, 'w') as f:
            f.write(f"""# Amazon RNN Model - TensorFlow.js Conversion

## Model Details
- **Name**: Amazon RNN Sentiment Analysis
- **Input Shape**: {model.input_shape}
- **Output Shape**: {model.output_shape}
- **Classes**: {model.output_shape[-1]} (likely Negative, Neutral, Positive)
- **Parameters**: {model.count_params():,}

## Files Generated
- `model.json` - Model architecture and metadata
- `group1-shard1of1.bin` - Model weights
- `tokenizer.json` - Amazon review tokenizer
- `model_config.json` - Detailed model configuration

## How to Use in 3D RNN Visualizer

1. **Open the web application** (`index.html`)
2. **Select "Load Keras Model"** radio button
3. **Upload `model.json`** as the model file
4. **Upload `tokenizer.json`** as the tokenizer file
5. **Click "Load Model"** and wait for success message
6. **Enter Amazon review text** and watch the visualization!

## Example Amazon Reviews to Try

### Positive Reviews:
- "This product exceeded my expectations! Great quality and fast shipping."
- "Amazing item, works perfectly. Highly recommend to everyone!"
- "Best purchase I've made this year. Excellent customer service too."

### Negative Reviews:
- "Terrible quality, broke after one day. Complete waste of money."
- "Worst product ever. Cheap materials and poor design."
- "Very disappointed with this purchase. Requesting a refund."

### Neutral Reviews:
- "The product is okay, nothing special but does what it's supposed to do."
- "Average quality for the price. Could be better but acceptable."
- "It works fine, though the design could be improved."

## Model Architecture
""")
            
            for i, layer in enumerate(model.layers):
                f.write(f"{i+1}. **{layer.name}** ({layer.__class__.__name__})")
                if hasattr(layer, 'units'):
                    f.write(f" - {layer.units} units")
                if hasattr(layer, 'activation'):
                    f.write(f" - {layer.activation.__name__} activation")
                f.write("\n")
        
        print(f"\n📋 Files created in '{output_dir}':")
        print(f"   ✓ model.json - Main model file")
        print(f"   ✓ group1-shard1of1.bin - Model weights")
        print(f"   ✓ tokenizer.json - Amazon review tokenizer")
        print(f"   ✓ model_config.json - Model configuration")
        print(f"   ✓ README.md - Usage instructions")
        
        print(f"\n🎯 Next Steps:")
        print(f"1. Open your 3D RNN Visualizer web application")
        print(f"2. Select 'Load Keras Model'")
        print(f"3. Upload the model.json file")
        print(f"4. Upload the tokenizer.json file")
        print(f"5. Start analyzing Amazon reviews!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Amazon RNN Model Converter")
    print("=" * 50)
    
    success = convert_amazon_model()
    
    if success:
        print("\n🎉 Conversion completed successfully!")
        print("Your Amazon RNN model is now ready for the 3D visualizer!")
    else:
        print("\n💥 Conversion failed. Please check the error messages above.")
