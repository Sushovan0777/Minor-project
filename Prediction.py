import cv2
import numpy as np
from tensorflow.keras.models import load_model #type: ignore

def predict_image(image, model):
    try:
        # Make prediction
        mask = model.predict(image, verbose=0)[0]
         # Process the prediction
        prediction = np.squeeze(mask)  # Remove batch dimension

        mask = np.squeeze(mask, axis=-1)
        mask = mask >= 0.7
        mask = mask.astype(np.int32)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate([mask, mask, mask], axis=-1)
        mask = mask * 255
        
        # # Convert mask to heatmap
        prediction = (prediction * 255).astype(np.uint8)  # Scale to 0-255 range
        heatmap = cv2.applyColorMap(prediction, cv2.COLORMAP_JET)
        
        # Get the original image (first image from batch, remove normalization)
        original = (image[0] * 255).astype(np.uint8)

        # # Overlay heatmap on original image
        heatmap_op = cv2.addWeighted(original, 0.8, heatmap, 0.6, 0)

        # concatinate mask and heatmap
        output = np.hstack([mask, heatmap_op])
        return output

    except Exception as e:
        print(f"Error in mask: {str(e)}")
        return None