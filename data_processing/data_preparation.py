import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms

class pytorch_lightning_dataset(torch.utils.data.Dataset):
    """
    Custom dataset class for PyTorch Lightning.

    This class represents a dataset used for training or evaluating a model with PyTorch Lightning.

    Args:
        sequenced_data (numpy.ndarray): The input data for the dataset.
        transforms (callable, optional): Optional data transformations to be applied to the data.

    Attributes:
        sequenced_data (numpy.ndarray): The input data for the dataset.
        transforms (callable): Optional data transformations.
        len (int): The number of samples in the dataset.

    """

    def __init__(self, sequenced_data, transforms=None):
        # Store the input data and transformations
        self.sequenced_data = sequenced_data
        self.transforms = transforms

        # Apply transformations to the data if provided
        if transforms is not None:
            self.sequenced_data = self.transforms(self.sequenced_data.astype('float32')).squeeze()

        # Set the number of samples in the dataset
        self.len = sequenced_data.shape[0]

    def __getitem__(self, index):
        # Get the data sample at the specified index
        item = self.sequenced_data[index, :]

        # Return the data sample
        return item

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.len
    
class pytorch_lightning_module(pl.LightningModule):
    """
    Lightning module class for PyTorch Lightning.

    This class represents a PyTorch Lightning module used for defining and training a model.

    Args:
        pytorch_model (torch.nn.Module): The PyTorch model to be used.
        input_dim (int): The dimensionality of the input data.
        output_dim (int): The dimensionality of the output data.

    Attributes:
        pytorch_model (torch.nn.Module): The PyTorch model.
        input_dim (int): The dimensionality of the input data.
        output_dim (int): The dimensionality of the output data.
    """

    def __init__(self, pytorch_model, input_dim, output_dim):
        super().__init__()

        # Store the PyTorch model, input dimension, and output dimension
        self.pytorch_model = pytorch_model
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.pytorch_model(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (torch.Tensor): The input batch.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The computed loss value.
        """
        data = batch
        x = data[:, 0:self.input_dim]
        y = data[:, self.input_dim::]
        y_predicted = self.forward(x)
        loss = F.mse_loss(y_predicted, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (torch.Tensor): The input batch.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The computed loss value.
        """
        data = batch
        x = data[:, 0:self.input_dim]
        y = data[:, self.input_dim::]
        y_predicted = self.forward(x)
        loss = F.mse_loss(y_predicted, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
class pytorch_lightning_dataModule(pl.LightningDataModule):
    """
    Data module class for PyTorch Lightning.

    This class represents a data module used for organizing and preparing data for model training with PyTorch Lightning.

    Args:
        sequenced_data (numpy.ndarray): The input data for the data module.
        batch_size (int): The batch size for the data loaders.

    Attributes:
        transforms (callable): Data transformations.
        sequenced_data (numpy.ndarray): The input data for the data module.
        batch_size (int): The batch size for the data loaders.
        dataset (pytorch_lightning_dataset): The dataset used for training and validation.
        train (torch.utils.data.Subset): The training subset of the dataset.
        val (torch.utils.data.Subset): The validation subset of the dataset.
    """

    def __init__(self, sequenced_data, batch_size):
        super().__init__()

        # Define data transformations
        self.transforms = transforms.ToTensor()

        # Store the input data and batch size
        self.sequenced_data = sequenced_data
        self.batch_size = batch_size

    def setup(self, stage=None, part_training=0.8):
        """
        Setup the data module.

        Args:
            stage (str, optional): The current stage, either 'fit' or None. Default is None.
            part_training (float, optional): The fraction of the data to be used for training. Default is 0.8.
        """

        if stage == 'fit' or stage is None:
            # Split the dataset into training and validation sets          
            self.dataset = pytorch_lightning_dataset(self.sequenced_data, self.transforms)
            
            data_size = len(self.dataset)
            train_size = int(part_training * data_size)
            val_size = int(data_size - train_size)
            # self.train, self.val = torch.utils.data.random_split(self.dataset, [train_size, val_size])
            # self.train, self.val = train_test_split(self.dataset, test_size=val_size, train_size=train_size,shuffle=False)
            self.train = self.dataset[:train_size]
            self.val = self.dataset[train_size:]

    def train_dataloader(self):
        """
        Create and return the data loader for training.

        Returns:
            torch.utils.data.DataLoader: The data loader for training.
        """

        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=os.cpu_count() - 2,
                          persistent_workers=True,
                          pin_memory=True)

    def val_dataloader(self):
        """
        Create and return the data loader for validation.

        Returns:
            torch.utils.data.DataLoader: The data loader for validation.
        """

        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=os.cpu_count() - 2,
                          persistent_workers=True,
                          pin_memory=True)