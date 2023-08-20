Neural Network Classifier for Shapes Recognition




This is a simple neural network classifier implemented in Python using the NumPy library. The goal of this project is to recognize different shapes (ellipses, triangles, and circles) from images and classify them accordingly.





Features

a. Multi-layer neural network architecture.
b. Sigmoid activation function for hidden and output layers.
c. Backpropagation for training the network.
d. Shape recognition using image convolution and feature extraction.
e. Ability to adjust the number of hidden layers during training.


****git clone https://github.com/your-username/neural-network-shape-recognition.git

****cd neural-network-shape-recognition

****pip install numpy pillow




Usage
Prepare your training images:

Organize your ellipse, triangle, and circle images into separate directories named Ellipses, Triangles, and Circles. Place these directories in the project folder.

Run the neural network training and testing:

Open the neural_network_shapes.py script in a code editor or IDE, and execute the script. This will train the neural network using the provided image data and test its performance on the same data.

Customization
You can customize the neural network's behavior by adjusting the following parameters in the neural_network_shapes.py script:

times: Number of hidden layers in the neural network.
iterations: Number of training iterations.
N: Learning rate.
M: Momentum rate.
Feel free to experiment with these parameters to observe how they affect the neural network's performance.

Notes
The images used for training and testing should be 28x28 pixels in size and grayscale.
The neural network uses a sigmoid activation function, which might not be optimal for more complex tasks. Consider experimenting with other activation functions and architectures for improved results.
