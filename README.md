# Ancient-Telugu-Character-Recognition


Document Model Architecture
The hybrid model for Telugu Alphabet Recognition combines Convolutional Neural Network (CNN) layers for feature extraction, a Bidirectional Long Short-Term Memory (BiLSTM) layer for sequential context, and a Transformer Encoder for attention-based feature refinement.

Model Architecture Breakdown:
Input Layer: The model accepts grayscale images of size (128, 128, 1).

CNN Feature Extractor:

Layer 1: Conv2D with 32 filters, (3,3) kernel size, relu activation, and same padding. Followed by MaxPooling2D with default pool size (2,2).
Layer 2: Conv2D with 64 filters, (3,3) kernel size, relu activation, and same padding. Followed by MaxPooling2D with default pool size (2,2).
Layer 3: Conv2D with 128 filters, (3,3) kernel size, relu activation, and same padding. Followed by MaxPooling2D with default pool size (2,2).
The repeated application of Conv2D and MaxPooling2D layers progressively extracts hierarchical features while reducing the spatial dimensions of the input image. After these layers, an input of (128, 128, 1) is reduced to an output feature map of (16, 16, 128).
Reshaping for Sequence Model: The output from the CNN feature extractor, which is (Batch_Size, 16, 16, 128), is Reshaped into a sequence (Batch_Size, 16*16, 128) = (Batch_Size, 256, 128). This transforms the spatial features into a sequence of 256 vectors, each of dimension 128, suitable for sequential processing layers.

BiLSTM Component: A Bidirectional(LSTM(128, return_sequences=True)) layer processes the reshaped sequence. It uses 128 LSTM units in each direction and return_sequences=True ensures that the layer outputs a sequence of the same length as the input, allowing the Transformer Encoder to process each step of the sequence.

Transformer Encoder (transformer_encoder function):

Multi-Head Self-Attention: MultiHeadAttention with num_heads=4 and key_dim=64. This layer allows the model to jointly attend to information from different representation subspaces at different positions.
Dropout: A Dropout layer with a rate of 0.1 is applied after attention to prevent overfitting.
Add & Normalize (1): The output of the attention layer is added to its input (inputs + x) and then passed through a LayerNormalization layer. This is a common residual connection pattern in Transformers.
Feed-Forward Network: This consists of two Dense layers:
First Dense layer with ff_dim=256 units and relu activation.
Second Dense layer which projects back to the original input dimension (inputs.shape[-1]).
Add & Normalize (2): The output of the feed-forward network is added to its input (x + f) and then passed through another LayerNormalization layer, forming another residual connection.
Global Pooling: A GlobalAveragePooling1D() layer is applied to the output of the Transformer Encoder. This condenses the sequence of features into a single feature vector by taking the average across the sequence dimension.

Classification Head: A final Dense layer with num_classes units and softmax activation produces the probability distribution over the different Telugu alphabet classes.
