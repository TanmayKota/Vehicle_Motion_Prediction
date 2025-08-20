from torch import nn
import torch
import torch.nn.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class run():
    def __init__(self,model):
        self.model=model
    
    def train_one_epoch(self,train_loader):
        self.model.train(True)
        print(f'Epoch: {epoch + 1}')
        running_loss = 0.0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        for batch_index, batch in enumerate(train_loader):
            if batch_index == len(train_loader) - 1:
                break
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            output = self.model(x_batch)
            loss = F.mse_loss(output, y_batch)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_index % 100 == 99:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                        avg_loss_across_batches))
                running_loss = 0.0
        print("Train Loss", loss)
        # log("train_loss", loss)
        print()

    def validate_one_epoch(self,test_loader):
        self.model.train(False)
        running_loss = 0.0

        for batch_index, batch in enumerate(test_loader):
            if batch_index == len(test_loader) - 1:
                break
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            
            with torch.no_grad():
    
                output = self.model(x_batch)
                loss = F.mse_loss(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)

        print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
        print('***************************************************')
        print("Val Loss", loss)
        print()