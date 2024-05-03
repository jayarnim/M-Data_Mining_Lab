import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(
            self, 
            num_users: int, 
            num_items: int, 
            num_factors: int, 
            units_mlp: list, 
            units_neumf: list, 
            dropout: float=0.5
            ):
        """
        Arguments
        num_users           : 사용자 수
        num_items           : 아이템 수
        num_factors         : 잠재요인 갯수
        units_mlp           : MLP 출력 노드 갯수
        units_neumf         : NeuMF 출력 노드 갯수
        dropout             : MLP, NeuMF Hidden Layer Dropout %
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.units_mlp = units_mlp
        self.units_neumf = units_neumf
        self.dropout = dropout

        self._error_generator()
        self._layer_initializer()


    def call(
            self, 
            inputs: list
            ):
        """
        Arguments
        inputs            : [user, item]
        """
        user_ids, item_ids = inputs

        user_factors = self.user_embedding(user_ids)
        item_factors = self.item_embedding(item_ids)

        gmf_output = self.GMF(user_factors, item_factors)
        mlp_output = self.MLP(user_factors, item_factors)
        neumf_output = self.NeuMF(gmf_output, mlp_output)

        final_output = self.output_layer(neumf_output)

        return final_output


    def GMF(self, user_factors, item_factors):
        gmf_output = tf.multiply(user_factors, item_factors)
        gmf_output = tf.nn.l2_normalize(gmf_output, axis=1)
        return gmf_output


    def MLP(self, user_factors, item_factors):
        mlp_input = tf.concat([user_factors, item_factors], axis=1)
        mlp_output = self.mlp_layers(mlp_input)
        return mlp_output


    def NeuMF(self, gmf_output, mlp_output):
        neumf_input = tf.concat([gmf_output, mlp_output], axis=1)
        neumf_output = self.neumf_layers(neumf_input)
        return neumf_output


    def _error_generator(self):
        CONDITION_00 = (type(self.num_users)==int) & (type(self.num_items)==int) & (type(self.num_factors)==int) & (type(self.units_mlp)==list) & (type(self.units_neumf)==list) & (type(self.dropout)==float)
        ERROR_MESSAGE_00 = "파라미터에 아규먼트가 올바르게 입력되지 않았습니다."
        assert CONDITION_00, ERROR_MESSAGE_00

        CONDITION_01 = self.units_mlp[-1]==self.num_factors
        ERROR_MESSAGE_01 = "MLP PART 의 마지막 출력 노드 갯수는 NUM_FACTORS 와 동일해야 합니다."
        assert CONDITION_01, ERROR_MESSAGE_01
        
        CONDITION_02 = (len(self.units_mlp)<=4) & (len(self.units_neumf)<=3)
        ERROR_MESSAGE_02 = "MLP PART 의 Hidden Layers 는 최대 4, NeuMF PART 의 Hidden Layers 는 최대 3 이 적정합니다."
        assert CONDITION_02, ERROR_MESSAGE_02


    def _layer_initializer(self):
        # Latent Factor
        self.user_embedding = tf.keras.layers.Embedding(input_dim=self.num_users, output_dim=self.num_factors)
        self.item_embedding = tf.keras.layers.Embedding(input_dim=self.num_items, output_dim=self.num_factors)

        # MLP
        user_factor_shape = self.num_factors
        item_factor_shape = self.num_factors
        mlp_input_shape = user_factor_shape+item_factor_shape
        self.mlp_layers = self._layer_generator(input_shape=mlp_input_shape, n_hidden_units=self.units_mlp, dropout=self.dropout)

        # NeuMF
        gmf_output_shape = self.num_factors
        mlp_output_shape = self.num_factors
        neumf_input_shape = gmf_output_shape+mlp_output_shape
        self.neumf_layers = self._layer_generator(input_shape=neumf_input_shape, n_hidden_units=self.units_neumf, dropout=self.dropout)

        # Final Output
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')


    def _layer_generator(self, input_shape, n_hidden_units, dropout):
        layers = []
        for idx in range(len(n_hidden_units)):
            if idx == 0: layers.append(tf.keras.layers.Dense(input_shape=(input_shape,), units=n_hidden_units[idx], activation='relu'))
            else: layers.append(tf.keras.layers.Dense(units=n_hidden_units[idx], activation='relu'))
            layers.append(tf.keras.layers.BatchNormalization())
            layers.append(tf.keras.layers.Dropout(dropout))

        sequential_layers = tf.keras.Sequential(layers)

        return sequential_layers
