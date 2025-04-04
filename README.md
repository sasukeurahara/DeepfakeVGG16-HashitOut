# 🧠 Deepfake Detection Using VGG16

This project is focused on detecting deepfake images by training a binary classifier using transfer learning with the VGG16 model. It was developed as part of the "Hash It Out" hackathon, where the goal was to address digital misinformation and its consequences using AI.

Deepfakes are synthetic media where a person in an image or video is replaced with someone else’s likeness using AI. This technology, while impressive, poses serious risks in terms of misinformation, identity theft, and privacy violations. The core aim of this project is to provide a lightweight, accurate image-based detection mechanism to distinguish between real and fake (AI-generated) facial images.

## 📁 Dataset

The dataset used contains two directories:
- `training_real`: Real facial images
- `training_fake`: AI-generated facial images

All images were resized to `224x224` and normalized for compatibility with the VGG16 model's input expectations. Some basic visualization and grayscale inspection were done to understand the structure and patterns in both classes.

## ⚙️ Model Details

We utilized the pre-trained VGG16 convolutional neural network available in Keras, removing the top (output) layer and adding a Dense layer with 2 output units and softmax activation for binary classification. All base layers were frozen to prevent their weights from being updated during training, making the process faster and reducing the risk of overfitting.

- Loss Function: `sparse_categorical_crossentropy` (because labels are integer encoded)
- Optimizer: `Adam`
- Metrics: `Accuracy`
- Epochs: `50`
- Batch Size: `20`
- Validation Split: `0.1` (from training set)
- Test Size: `0.2` (held out from the full dataset)

## 📊 Results

The model performed exceptionally well given the dataset size and task:

- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~96%
- **Test Accuracy**: ~94%

The use of transfer learning helped the model converge quickly and achieve strong generalization without extensive tuning.

## 🧪 Inference Pipeline

For predictions, any input image is resized to 224x224 and passed to the model through `model.predict()`. The softmax output is then thresholded:
- Class 0 → Real
- Class 1 → Fake

An example visualization pipeline was built to show the input image alongside predicted and actual labels.

## 📦 Dependencies

- `TensorFlow / Keras`: Model creation and training
- `OpenCV`: Image loading and processing
- `NumPy`, `Pandas`: Data manipulation
- `Matplotlib`: Visualization
- `scikit-learn`: Train/test splitting, metrics

## 🔬 Key Learnings

- VGG16, though a slightly older architecture, still performs very well on image classification tasks when fine-tuned properly.
- Transfer learning is highly effective when dataset size is limited.
- Even simple deep learning pipelines can detect subtle synthetic image features if properly trained.

## 🚀 Future Improvements

- Expand dataset to include more diverse real/fake images
- Add video frame-level detection for deepfake videos
- Include face landmark tracking and motion consistency as additional cues
- Deploy using Flask or FastAPI and convert to TensorFlow Lite for on-device inference
