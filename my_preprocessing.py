import cv2
import numpy as np
from PIL import Image
import os

# Configuration
TARGET_SIZE = (128, 128)  # Resize all images to 128x128
NORMALIZE = True          # Normalize pixel values to [0, 1]

def preprocess_image(image_path):
    """
    Preprocess a single image
    """
    # Load image (handles both JPG and TIFF)
    img = Image.open(image_path)
    
    # Convert to grayscale
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Apply histogram equalization (improve contrast)
    img_array = cv2.equalizeHist(img_array)
    
    # Resize to standard size
    img_resized = cv2.resize(img_array, TARGET_SIZE)
    
    # Normalize if needed
    if NORMALIZE:
        img_resized = img_resized.astype(np.float32) / 255.0
    
    return img_resized

def augment_image(image):
    """
    Create augmented versions of an image
    Returns: list of augmented images
    """
    augmented = []
    
    # Convert back to uint8 for augmentation
    if image.dtype == np.float32:
        img = (image * 255).astype(np.uint8)
    else:
        img = image
    
    # 1. Horizontal flip
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)
    
    # 2. Rotate +10 degrees
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
    rotated1 = cv2.warpAffine(img, M, (cols, rows))
    augmented.append(rotated1)
    
    # 3. Rotate -10 degrees
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -10, 1)
    rotated2 = cv2.warpAffine(img, M, (cols, rows))
    augmented.append(rotated2)
    
    # 4. Darker
    darker = cv2.convertScaleAbs(img, alpha=0.8, beta=0)
    augmented.append(darker)
    
    # 5. Brighter
    brighter = cv2.convertScaleAbs(img, alpha=1.2, beta=0)
    augmented.append(brighter)
    
    return augmented

def process_dataset(input_folder, output_folder, augment_classes=None):
    """
    Process entire dataset
    
    Args:
        input_folder: Path to input dataset (e.g., 'CK_dataset')
        output_folder: Path to save processed dataset
        augment_classes: List of class names to augment (e.g., ['fear'])
    """
    for split in ['train', 'test']:
        split_path = os.path.join(input_folder, split)
        output_split_path = os.path.join(output_folder, split)
        
        # Get all emotion folders
        emotions = os.listdir(split_path)
        
        for emotion in emotions:
            emotion_path = os.path.join(split_path, emotion)
            output_emotion_path = os.path.join(output_split_path, emotion)
            os.makedirs(output_emotion_path, exist_ok=True)
            
            # Get all images
            images = [f for f in os.listdir(emotion_path) 
                     if f.endswith(('.jpg', '.png', '.tiff', '.jpeg'))]
            
            print(f"Processing {split}/{emotion}: {len(images)} images")
            
            for img_name in images:
                img_path = os.path.join(emotion_path, img_name)
                
                # Preprocess
                processed_img = preprocess_image(img_path)
                
                # Save original processed image
                base_name = os.path.splitext(img_name)[0]
                save_path = os.path.join(output_emotion_path, f"{base_name}.png")
                
                # Convert back to uint8 for saving
                save_img = (processed_img * 255).astype(np.uint8) if NORMALIZE else processed_img
                cv2.imwrite(save_path, save_img)
                
                # Augment if this class needs it (only for training)
                if split == 'train' and augment_classes and emotion in augment_classes:
                    augmented = augment_image(processed_img)
                    for idx, aug_img in enumerate(augmented, 1):
                        aug_save_path = os.path.join(output_emotion_path, 
                                                    f"{base_name}_aug{idx}.png")
                        cv2.imwrite(aug_save_path, aug_img)
            
            # Count final images
            final_count = len(os.listdir(output_emotion_path))
            print(f"  → Saved {final_count} images")

# Main execution
if __name__ == "__main__":
    print("Starting preprocessing...")
    
    # Process CK dataset
    print("\n" + "="*50)
    print("Processing CK Dataset")
    print("="*50)
    process_dataset(
        input_folder='CK_dataset',
        output_folder='processed_CK_dataset',
        augment_classes=['fear']  # Only augment fear class
    )
    
    # Process JAFFE dataset
    print("\n" + "="*50)
    print("Processing JAFFE Dataset")
    print("="*50)
    process_dataset(
        input_folder='JAFFE-[70,30]',
        output_folder='processed_JAFFE_dataset',
        augment_classes=['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']  # Augment all
    )
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE!")
    print("="*50)