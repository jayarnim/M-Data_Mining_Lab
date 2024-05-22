import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Model(nn.Module):
    def __init__(
            self, 
            input_dim, 
            latent_dim, 
            dropout_rate=0.5
            ):
        super(Model, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        self._layer_generator()

    def forward(self, x):
        encoded = self._encoder(x)
        decoded = self._decoder(encoded)
        return encoded, decoded

    def _encoder(self, x):
        x = torch.relu(self.encoder_dense1(x))
        x = self.dropout(x)
        x = torch.relu(self.encoder_dense2(x))
        x = self.dropout(x)
        encoded = torch.relu(self.encoder_dense3(x))
        return encoded

    def _decoder(self, x):
        x = torch.relu(self.decoder_dense1(x))
        x = self.dropout(x)
        x = torch.relu(self.decoder_dense2(x))
        x = self.dropout(x)
        decoded = self.decoder_dense3(x)
        return decoded

    def _layer_generator(self):
        # Encoder layers
        self.encoder_dense1 = nn.Linear(self.input_dim, self.latent_dim * 4)
        self.encoder_dense2 = nn.Linear(self.latent_dim * 4, self.latent_dim * 2)
        self.encoder_dense3 = nn.Linear(self.latent_dim * 2, self.latent_dim)

        # Decoder layers
        self.decoder_dense1 = nn.Linear(self.latent_dim, self.latent_dim * 2)
        self.decoder_dense2 = nn.Linear(self.latent_dim * 2, self.latent_dim * 4)
        self.decoder_dense3 = nn.Linear(self.latent_dim * 4, self.input_dim)

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)


class Trainer:
    def __init__(
            self, 
            model, 
            criterion='mse', 
            learning_rate=0.001
            ):
        self.model = model
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        if criterion=='mse':
            self.criterion = self._masked_mse_loss

    def train(
            self,
            data, 
            mask,
            num_epochs=50, 
            batch_size=32, 
            learning_rate=0.001, 
            refeeding=True
            ):
        # 데이터 세트 올리기
        dataloader = self._dataset_loader(data=data, mask=mask)
        # 모형을 학습 가능한 상태로 전환
        self.model.train()

        for epoch in range(num_epochs):
            for batch_data, batch_mask in tqdm(dataloader):
                # First Step
                # Initialize Params
                self.optimizer.zero_grad()
                # Forward Pass
                _, decoded = self.model(batch_data)
                # Loss
                loss = self.criterion(decoded, batch_data, batch_mask)
                # Backward Pass
                loss.backward(retain_graph=True)
                # Update Params
                self.optimizer.step()

                # Second Step(Dense re-Feeding Step)
                if refeeding==True:
                    self.optimizer.zero_grad()
                    _, re_decoded = self.model(decoded.detach())
                    refeeding_loss = self.criterion(re_decoded, batch_data, batch_mask)
                    refeeding_loss.backward()
                    self.optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Refeeding Loss: {refeeding_loss.item():.4f}')

    def _dataset_loader(self, data, mask):
        masked = (data != mask).float()
        dataset = torch.utils.data.TensorDataset(data, masked)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def _masked_mse_loss(self, pred, target, masked):
        loss = (pred - target) ** 2
        loss = loss * masked
        return loss.sum() / masked.sum()
