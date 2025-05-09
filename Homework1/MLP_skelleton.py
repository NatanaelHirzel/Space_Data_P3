#TODO: all the neccessary imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return self.fc6(x)


#TODO: define and fill the normalize_data function
def normalize_data(data_whole):
    normalized_data = np.zeros(data_whole.shape)
    for i in range(data_whole.shape[0]):
        data = data_whole[i] 
        data_min = data.min()
        data_max = data.max()
        data_range = data_max - data_min
        normalized_data[i] = (data - data_min) / data_range

    return normalized_data

#TODO define and fill the MSE function
def mean_squared_error(data1, data2):
    # Ensure the two datasets have the same shape
    if data1.shape != data2.shape:
        raise ValueError("Input datasets must have the same shape.")
    # Calculate the mean squared error
    mse = ((data1 - data2) ** 2).mean()
    return mse

#TODO fill the main training function
def train_model(model, train_loader, test_loader, num_epochs=1000):
    loss_curve_train = []
    loss_curve_val = []

    criterion = mean_squared_error
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Average training loss for the epoch
        train_loss /= len(train_loader)
        loss_curve_train.append(train_loss)

        # Evaluate on the test set every epoch
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()

        # Average validation loss for the epoch
        test_loss /= len(test_loader)
        loss_curve_val.append(test_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    return model, loss_curve_train, loss_curve_val

#TODO: fill the function to plot a clean & noisy sample as well as the corresponding NN prediction
def plot_noisy_clean_predicted(noisy_sample, prediction_sample, clean_sample, noisy_picture, clean_picture):
    plt.figure(figsize=(15, 5))
    noisy_picture = noisy_picture - noisy_picture.min()
    noisy_picture = noisy_picture / noisy_picture.max()
    clean_picture = clean_picture - clean_picture.min()
    clean_picture = clean_picture / clean_picture.max()
    plt.plot(noisy_sample, label="Noisy", color="red")
    plt.plot(clean_sample, label="Clean", color="green")    
    plt.plot(prediction_sample, label="Predicted", color="blue")
    plt.legend()
    plt.title("Profiles")
    plt.savefig("Homework1/Images/profiles.png")
    plt.close()

    ratio = (prediction_sample - noisy_sample)/6
    prediction_picture = np.zeros(noisy_picture.shape)
    print(ratio.max(), ratio.min(), ratio)

    for i in range(ratio.shape[0]):
        prediction_picture[i,:] = ratio[i] + noisy_picture[i,:]
    plt.subplot(1, 3, 1)
    plt.imshow(noisy_picture, cmap='gray')
    plt.title("Noisy Image")
    plt.subplot(1, 3, 2)
    plt.imshow(clean_picture, cmap='gray')
    plt.title("Clean Image")
    plt.subplot(1, 3, 3)
    plt.imshow(prediction_picture, cmap='gray')
    plt.title("Predicted Image")
    plt.savefig("Homework1/Images/prediction.png")
    return True


if __name__ == "__main__":

    # (A) Load data (example: shape (N, H, W))
    noisy_data = np.load("Homework1/Data/noisy_images_small_1k.npy").astype(np.float32)
    clean_data = np.load("Homework1/Data/clean_images_small_1k.npy").astype(np.float32)
    print("noisy_data", noisy_data.shape)
    # Convert to row-wise brightness profiles, shape (N, H)
    noisy_profiles = noisy_data.mean(axis=2)
    clean_profiles = clean_data.mean(axis=2)
    print("noisy_profiles", noisy_profiles.shape)

    #TODO: call the normalize function
    noisy_profiles_norm = normalize_data(noisy_profiles)
    clean_profiles_norm = normalize_data(clean_profiles)

    #TODO convert to torch tensors and create the TensorDataset
    noisy_tensor = torch.from_numpy(noisy_profiles_norm).float()
    clean_tensor = torch.from_numpy(clean_profiles_norm).float()

    # Split into training and test sets
    noisy_train, noisy_test, clean_train, clean_test = train_test_split(
        noisy_tensor, clean_tensor, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(noisy_train, clean_train)
    test_dataset = TensorDataset(noisy_test, clean_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = noisy_tensor.shape[1]
    hidden_size = 4*128
    output_size = clean_tensor.shape[1]
    model = MLP(input_size, hidden_size, output_size)


    model, loss_curve_train, loss_curve_val = train_model(model, train_loader, test_loader, num_epochs=180)

    #TODO set the model to eval mode
    model.eval()

    #TODO: call your noisy_clean_predicted function with a sample from the training and a sample from the testing dataset
    index = 100
    #convert to numpy
    noisy_sample = test_dataset[index][0]
    clean_sample = test_dataset[index][1]
    prediction_sample = model(noisy_sample.unsqueeze(0))[0].detach().numpy()
    noisy_sample = noisy_sample.numpy()
    plot_noisy_clean_predicted(noisy_sample, prediction_sample, clean_sample, noisy_data[index], clean_data[index])
    
    plt.figure()
    plt.plot(loss_curve_train[1:])
    plt.title("Training Loss")
    plt.plot(loss_curve_val[1:])
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("Homework1/Images/loss_curve.png")