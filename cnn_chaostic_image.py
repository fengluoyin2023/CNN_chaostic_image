
# coding: utf-8

# In[ ]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Step 1: Generate Chaotic Image as Key
def generate_chaotic_image(image_shape, iterations=1000):
    # Initialize random chaotic image (can be replaced with other chaotic functions)
    chaotic_image = np.random.rand(*image_shape) * 255

    # Logistic map parameters
    r = 3.99
    x = 0.5

    # Logistic map function
    def logistic_map(r, x):
        return r * x * (1 - x)

    # Generate chaotic image
    for i in range(iterations):
        chaotic_image = logistic_map(r, chaotic_image)

    return chaotic_image

# Step 2: Image Encryption using Chaotic Image
def encrypt_image(original_image, chaotic_image):
    encrypted_image = original_image ^ chaotic_image  # XOR operation between original and chaotic images
    return encrypted_image

# Step 3: Image Decryption using Chaotic Image
def decrypt_image(encrypted_image, chaotic_image):
    decrypted_image = encrypted_image ^ chaotic_image  # XOR operation with chaotic image to retrieve original image
    return decrypted_image

# Step 4: Image Encryption using CNN (This is a simplified example)
def encrypt_image_cnn(encrypted_image):
    # Implement your CNN model for image encryption here
    # Example: Use TensorFlow/Keras to create an encryption CNN model
    model = tf.keras.Sequential([
        # Add your convolutional layers here
        # ...
        # Add your dense layers here
        # ...
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(encrypted_image, encrypted_image, epochs=10, batch_size=32)

    # Use the trained CNN model to encrypt the image
    encrypted_image_cnn = model.predict(encrypted_image)

    return encrypted_image_cnn

# 改动
if __name__ == "__main__":
    # Load original image
    original_image_path = "remote_image.jpg"
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    # Generate chaotic image as a key
    chaotic_image = generate_chaotic_image(original_image.shape)

    # Encrypt the original image
    encrypted_image = encrypt_image(original_image, chaotic_image)

    # Encrypt the encrypted image using CNN
    encrypted_image_cnn = encrypt_image_cnn(encrypted_image)

    # Display the original and encrypted images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(encrypted_image, cmap="gray")
    plt.title("Encrypted Image")

    plt.subplot(1, 3, 3)
    plt.imshow(encrypted_image_cnn, cmap="gray")
    plt.title("Encrypted Image with CNN")

    plt.show()

