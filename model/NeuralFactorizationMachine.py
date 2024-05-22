import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(
            self,
            num_numeric_features,
            num_categoric_features,
            num_latent_factor,
            hidden_layers_unit
            ):
        super(Model, self).__init__()
        self.num_numeric_features = num_numeric_features
        self.num_categoric_features = num_categoric_features
        self.num_latent_factor = num_latent_factor
        self.hidden_layers_unit = hidden_layers_unit

        self._layer_generator()

    def forward(self, numeric_inputs, categorical_inputs):
        linear_part_output = self._linear_part(numeric_inputs, categorical_inputs)
        interaction_part_output = self._interaction_part(numeric_inputs, categorical_inputs)
        output = linear_part_output + interaction_part_output
        return output

    def _linear_part(self, numeric_inputs, categorical_inputs):
        # One-Hot Encoding for Categorical Features
        one_hot_categorical = self.one_hot_encoder(categorical_inputs)
        # Linear Input
        linear_part_inputs = torch.cat([numeric_inputs.float(), one_hot_categorical], dim=1)
        # Compute
        linear_part_output = self.linear_layer(linear_part_inputs)
        return linear_part_output

    def _interaction_part(self, numeric_inputs, categorical_inputs):
        # Create Latent Factor
        # Embedding for numeric features
        numeric_embeds = self.numeric_embeddings(numeric_inputs)
        # Embedding for categorical features
        category_embeds = [self.category_embeddings[i](categorical_inputs[:, i]) for i in range(len(self.category_embeddings))]
        category_embeds = torch.stack(category_embeds, dim=1)
        # Concatenate all embeddings
        combined_embeds = torch.cat([numeric_embeds, category_embeds], dim=1)
        
        # Compute interaction term
        summed_features_embeds = torch.sum(combined_embeds, dim=1)
        summed_features_embeds_square = torch.pow(summed_features_embeds, 2)
        squared_sum_features_embeds = torch.pow(combined_embeds, 2)
        squared_sum_features_embeds = torch.sum(squared_sum_features_embeds, dim=1)
        interaction_part = 0.5 * (summed_features_embeds_square - squared_sum_features_embeds)
        
        # Deep Neural Networks part
        x = interaction_part
        for layer in self.interaction_layer:
            x = F.relu(layer(x))
        
        # Output layer
        interaction_part_output = self.interaction_output(x)

        return interaction_part_output

    def _layer_generator(self):
        # Linear Part
        total_linear_features = self.num_numeric_features + sum(self.num_categoric_features)
        self.linear_layer = nn.Linear(total_linear_features, 1, bias=True)

        # Interaction part
        # Embedding layer
        self.numeric_embeddings = nn.Embedding(num_numeric_features, embedding_dim)
        self.category_embeddings = nn.ModuleList([nn.Embedding(cat_size, embedding_dim) for cat_size in num_categories])
        # Interaction layer
        input_dim = self.num_latent_factor * (self.num_numeric_features + len(self.num_categoric_features))

        self.interaction_layer = nn.ModuleList()
        for hidden_dim in self.hidden_layers_unit:
            self.interaction_layer.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.interaction_output = nn.Linear(input_dim, 1)

    def _one_hot_encoder(self, categorical_inputs):
        num_features = categorical_inputs.size(1)
        one_hot_categorical = []

        for i in range(num_features):
            one_hot = F.one_hot(categorical_inputs[:, i], num_classes=categorical_inputs[:, i].max()+1)
            one_hot_categorical.append(one_hot)

        one_hot_categorical = torch.cat(one_hot_categorical, dim=1).float()

        return one_hot_categorical
