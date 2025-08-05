#!/usr/bin/env python3
"""
Simple Keras to TensorFlow.js Converter
This script converts your Amazon RNN model to TensorFlow.js format
"""

import os
import sys

def convert_keras_model():
    """Convert Keras model to TensorFlow.js format"""
    
    # Check if model file exists
    model_path = r"D:\SE\Amazon RNN Model.keras"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"✅ Found model file: {model_path}")
    
    try:
        # Import TensorFlow
        print("📦 Loading TensorFlow...")
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Load the Keras model
        print("🔄 Loading Keras model...")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Print model summary
        print("\n📊 Model Summary:")
        model.summary()
        
        # Save as SavedModel format first (more reliable)
        saved_model_path = "./amazon_model_savedmodel"
        print(f"\n💾 Saving as SavedModel format to: {saved_model_path}")
        tf.saved_model.save(model, saved_model_path)
        print("✅ SavedModel saved successfully!")
        
        # Now try to convert to TensorFlow.js using command line
        print("\n🔄 Converting to TensorFlow.js format...")
        tfjs_output_path = "./amazon_model_tfjs"
        
        # Use tensorflowjs_converter command
        import subprocess
        cmd = [
            "tensorflowjs_converter",
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            "--signature_name=serving_default",
            "--saved_model_tags=serve",
            saved_model_path,
            tfjs_output_path
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ TensorFlow.js conversion successful!")
            print(f"✅ Converted model saved to: {tfjs_output_path}")
            
            # List generated files
            if os.path.exists(tfjs_output_path):
                files = os.listdir(tfjs_output_path)
                print(f"\n📁 Generated files:")
                for file in files:
                    print(f"  - {file}")
                    
                # Check for model.json
                model_json_path = os.path.join(tfjs_output_path, "model.json")
                if os.path.exists(model_json_path):
                    print(f"\n🎯 Main model file: {model_json_path}")
                    print("✅ This is the file you need to load in your JavaScript code!")
                    
            return True
        else:
            print("❌ TensorFlow.js conversion failed!")
            print("Error output:", result.stderr)
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try installing missing packages:")
        print("   pip install tensorflow tensorflowjs")
        return False
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Keras to TensorFlow.js Converter")
    print("=" * 50)
    
    success = convert_keras_model()
    
    if success:
        print("\n🎉 Conversion completed successfully!")
        print("\n📝 Next steps:")
        print("1. Copy the 'amazon_model_tfjs' folder to your web project")
        print("2. Update your JavaScript code to load 'model.json'")
        print("3. The model will work with your 3D RNN visualization!")
    else:
        print("\n💔 Conversion failed. Please check the errors above.")
        
if __name__ == "__main__":
    main()
