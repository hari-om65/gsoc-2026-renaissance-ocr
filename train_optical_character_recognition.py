from tqdm import tqdm
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optimization
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Disable the image size limit to prevent Decompression Bomb Errors
Image.MAX_IMAGE_PIXELS = None
import fitz  # Library for Portable Document Format conversion

# ---------------------------------------------------------
# 1. ARCHITECTURE: Convolutional Recurrent Neural Network
# ---------------------------------------------------------
class HistoricalConvolutionalRecurrentNeuralNetwork(nn.Module):
    def __init__(self, total_character_classes, hidden_layer_size=256):
        super(HistoricalConvolutionalRecurrentNeuralNetwork, self).__init__()
        
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.recurrent_layers = nn.LSTM(
            input_size=256, hidden_size=hidden_layer_size, 
            num_layers=2, bidirectional=True, batch_first=False
        )
        
        self.linear_output = nn.Linear(hidden_layer_size * 2, total_character_classes)

    def forward(self, image_input_tensor):
        extracted_features = self.convolutional_layers(image_input_tensor)
        batch_size, channels, height, width = extracted_features.size()
        extracted_features = extracted_features.view(batch_size, channels * height, width)
        extracted_features = extracted_features.permute(2, 0, 1) 
        recurrent_output, _ = self.recurrent_layers(extracted_features)
        return self.linear_output(recurrent_output)

# ---------------------------------------------------------
# 2. DATASET: Loading and Augmentation
# ---------------------------------------------------------
vocabulary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ſçáéíóúñ.,;:'"
character_to_index = {char: index + 1 for index, char in enumerate(vocabulary)}
character_to_index['-'] = 0 

class HistoricalDocumentDataset(Dataset):
    def __init__(self, image_directory, text_labels_dictionary):
        self.image_directory = image_directory
        self.image_filenames = list(text_labels_dictionary.keys())
        self.text_labels = text_labels_dictionary
        
        self.augmentation_pipeline = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(degrees=5), 
            transforms.GaussianBlur(kernel_size=3), 
            transforms.Resize((64, 256)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_name = self.image_filenames[index]
        image_path = os.path.join(self.image_directory, image_name)
        
        document_image = Image.open(image_path).convert('RGB')
        processed_image = self.augmentation_pipeline(document_image)
        
        transcript = self.text_labels[image_name]
        numerical_target = [character_to_index.get(char, 1) for char in transcript]
        target_tensor = torch.tensor(numerical_target, dtype=torch.long)
        
        return processed_image, target_tensor

# ---------------------------------------------------------
# 3. EXECUTION: Conversion and Training Loop
# ---------------------------------------------------------
def convert_portable_document_format_to_images(document_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    document = fitz.open(document_path)
    for page_number in range(len(document)):
        # Include the document name in the image name to prevent overwriting files with the same page number
        image_filename = f"{output_folder}/{os.path.basename(document_path)}_page_{page_number + 1}.jpg"
        
        if os.path.exists(image_filename):
            print(f"Skipping {image_filename} (already exists)")
            continue
            
        page = document.load_page(page_number)
        pixel_map = page.get_pixmap(dpi=300) 
        pixel_map.save(image_filename)
        
    print(f"Successfully processed {document_path}.")

# Custom collation function to handle variable-length text labels
def historical_collate_function(batch):
    batch_images = [item[0] for item in batch]
    batch_labels = [item[1] for item in batch]
    batch_images = torch.stack(batch_images)
    return batch_images, batch_labels

if __name__ == "__main__":
    source_directory = "historical_sources"
    output_image_directory = "dataset_images"

    # 1. Convert ALL 6 Portable Document Format files in the folder
    portable_document_format_files = glob.glob(f"{source_directory}/*.pdf")
    for document_path in portable_document_format_files:
        convert_portable_document_format_to_images(document_path, output_image_directory)
    
    # 2. Define the EXACT ground truth from your reference document
    ground_truth_labels = {
        # Note the updated naming convention to match the conversion function above
        "Covarrubias - Tesoro lengua.pdf_page_7.jpg": "CENSVRA DE PEDRO DE VALENCIA CORONISTA General de el Rey nueftro Señor.",
        "Covarrubias - Tesoro lengua.pdf_page_8.jpg": "POr mandado del Confejo Supremo de Caftilla, he vifto el libro in-",
        # YOU MUST FILL IN THE REST OF THE TRANSCRIBED LINES HERE FOR ALL DOCUMENTS
    }
    
    # Filter ground truth labels to only include files that actually exist in the image directory
    valid_ground_truth_labels = {k: v for k, v in ground_truth_labels.items() if os.path.exists(os.path.join(output_image_directory, k))}

    # 3. Setup Data Loader
    historical_dataset = HistoricalDocumentDataset(output_image_directory, valid_ground_truth_labels)
    training_data_loader = DataLoader(
        historical_dataset, 
        batch_size=4, 
        shuffle=True,
        collate_fn=historical_collate_function
    )
    
    # 4. Initialize Model and Loss
    total_unique_characters = len(vocabulary) + 1 
    historical_model = HistoricalConvolutionalRecurrentNeuralNetwork(total_unique_characters)
    
    hardware_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    historical_model.to(hardware_device)
    
    connectionist_temporal_classification_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    learning_optimizer = optimization.Adam(historical_model.parameters(), lr=0.001)
    
    # 5. Training Loop
    print(f"Starting training on {hardware_device} for all documents...")
    historical_model.train()
    
    for current_epoch in range(50):
        accumulated_loss = 0.0
        
        if len(training_data_loader) == 0:
            print("Error: No training data found. Please check your ground_truth_labels mapping.")
            break

        for image_batch, text_labels in training_data_loader:
            image_batch = image_batch.to(hardware_device)
            
            learning_optimizer.zero_grad()
            prediction_output = historical_model(image_batch)
            
            input_lengths = torch.full(size=(image_batch.size(0),), fill_value=prediction_output.size(0), dtype=torch.long)
            target_lengths = torch.tensor([len(label) for label in text_labels], dtype=torch.long)
            
            # Send the flattened labels to the Graphics Processing Unit
            flattened_labels = torch.cat([label.to(hardware_device) for label in text_labels])
            
            batch_loss = connectionist_temporal_classification_loss(prediction_output, flattened_labels, input_lengths, target_lengths)
            batch_loss.backward()
            learning_optimizer.step()
            
            accumulated_loss += batch_loss.item()
            
        if (current_epoch + 1) % 10 == 0:
            print(f"Epoch {current_epoch + 1}/50 completed. Loss: {accumulated_loss / len(training_data_loader)}")

    torch.save(historical_model.state_dict(), "historical_optical_character_recognition_weights.pth")
    print("Training complete. Model weights saved successfully.")