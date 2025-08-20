import torch


class acceleration():

    def __init__(self) -> None:
        
        pass

    def forward(x,y):
        x=torch.tensor(x)
        y=torch.tensor(y)
        positions = y[:, :]
        predictions = []
        velocities = x[:, 0:1]
        accelerations = x[:, 2:3]
        
        positions = positions + 10/25*velocities + 10/(25^2) *0.5 * accelerations
        velocities = velocities + 10/25*accelerations
        positions_squeezed = positions.unsqueeze(1)
        predictions.append(positions_squeezed)
        predictions = torch.cat(predictions, dim=1)

        # Reshape predictions to have approximately 625,000 rows and 2 columns
        predictions_2d = predictions.view(-1, 2)

        return predictions_2d


    def predictionmatrix (pred, y_act ):
        y_act=torch.tensor(y_act)
        pred_10 = pred[:10]
        pred_values = torch.cat((pred_10,y_act))
        pred_values = pred_values[:-10]
        return pred_values

    def loss_function( y_act, y_hat):    
        y_act=torch.tensor(y_act)
        diff = y_act-y_hat
        loss = torch.mean(torch.square(diff))
        return loss