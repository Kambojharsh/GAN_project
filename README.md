# Anime Faces DCGAN

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate anime-style faces using the Anime Face Dataset. The implementation leverages PyTorch and other essential Python libraries.

---

## Features

- Downloads and preprocesses the Anime Face dataset from Kaggle.
- Implements a DCGAN model for image generation.
- Includes data augmentation and model training pipelines.

---

## Setup and Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- PyTorch (compatible with your system's CUDA version)
- COLAB Notebook (optional, for running the `.ipynb` file)

### Required Libraries

The notebook uses the following Python libraries:

- `opendatasets` for downloading the dataset.
- `torch`, `torchvision` for deep learning and dataset handling.
- `os`, `shutil`, `numpy`, and other standard libraries.

Install the dependencies using the following command:

```bash
pip install opendatasets torch torchvision

```
### Downloading the Dataset

You need Kaggle API credentials to download the Anime Face dataset.

1. Create an account on [Kaggle](https://www.kaggle.com/).
2. Navigate to "My Account" and generate an API key (`kaggle.json`).
3. Add your Kaggle credentials in the notebook when prompted.
4. The dataset will be downloaded automatically when running the notebook.

---

### File Structure

- **`DCGAN_anime_faces.ipynb`**: Main notebook containing the dataset download, preprocessing, model definition, and training.
- **`animefacedataset/`**: Contains the downloaded dataset images (automatically created).

---

### Usage

#### Run the Notebook

Open `DCGAN_anime_faces.ipynb` and execute each cell in order.

#### Dataset Preprocessing

The dataset is automatically downloaded and extracted into the `animefacedataset/` directory. Images are prepared for training using `torchvision.transforms`.

#### Model Training

- The DCGAN model is defined in PyTorch with separate Generator and Discriminator modules.
- Adjust training hyperparameters (epochs, learning rate, batch size) in the notebook as needed.
- Start training the GAN by running the relevant cells.

#### Generate Anime Faces

After training, generate new anime faces using the trained Generator. Save or visualize the output using the provided utilities.

---

### Output

- The notebook produces a trained DCGAN model.
- Generated images are saved during the training process or can be visualized inline.
