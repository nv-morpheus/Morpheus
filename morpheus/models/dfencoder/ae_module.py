import torch
from collections import OrderedDict


def compute_embedding_size(n_categories):
    """
    Applies a standard formula to choose the number of feature embeddings
    to use in a given embedding layers.

    n_categories is the number of unique categories in a column.
    """
    val = min(600, round(1.6 * n_categories**0.56))
    return int(val)


class CompleteLayer(torch.nn.Module):
    """
    Impliments a layer with linear transformation
    and optional activation and dropout."""

    def __init__(self, in_dim, out_dim, activation=None, dropout=None, *args, **kwargs):
        super(CompleteLayer, self).__init__(*args, **kwargs)
        self.layers = []
        linear = torch.nn.Linear(in_dim, out_dim)
        self.layers.append(linear)
        self.add_module("linear_layer", linear)

        self.activation = activation
        if activation is not None:
            act = self.interpret_activation(activation)
            self.layers.append(act)
        if dropout is not None:
            dropout_layer = torch.nn.Dropout(dropout)
            self.layers.append(dropout_layer)
            self.add_module("dropout", dropout_layer)

    def interpret_activation(self, act=None):
        if act is None:
            act = self.activation
        activations = {
            "leaky_relu": torch.nn.functional.leaky_relu,
            "relu": torch.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "selu": torch.selu,
            "hardtanh": torch.nn.functional.hardtanh,
            "relu6": torch.nn.functional.relu6,
            "elu": torch.nn.functional.elu,
            "celu": torch.nn.functional.celu,
            "rrelu": torch.nn.functional.rrelu,
            "hardshrink": torch.nn.functional.hardshrink,
            "tanhshrink": torch.nn.functional.tanhshrink,
            "softsign": torch.nn.functional.softsign,
        }
        try:
            return activations[act]
        except:
            msg = f"activation {act} not understood. \n"
            msg += "please use one of: \n"
            msg += str(list(activations.keys()))
            raise Exception(msg)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AEModule(torch.nn.Module): 
    """ Auto Encoder Pytorch Module."""
    def __init__(
        self,
        verbose,
        encoder_layers=None,
        decoder_layers=None,
        encoder_dropout=None,
        decoder_dropout=None,
        encoder_activations=None,
        decoder_activations=None,
        activation="relu",
        device=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_activations = encoder_activations
        self.decoder_activations = decoder_activations
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.activation = activation
        self.device = device

        # mapping cat features to the embedding layer
        self.categorical_embedding = OrderedDict()
        self.encoder = []
        self.decoder = []
        self.numeric_output = None
        self.binary_output = None
        # mapping cat features to the output layer
        self.categorical_output = OrderedDict()

    def build(self, numeric_fts, binary_fts, categorical_fts):
        if self.verbose:
            print("Building model...")

        cat_input_dim = self.build_categorical_input_layers(categorical_fts)

        # compute input dimension
        num_ft_cnt, bin_ft_cnt = len(numeric_fts), len(binary_fts)
        input_dim = cat_input_dim + num_ft_cnt + bin_ft_cnt

        dim = self.build_layers(input_dim)

        # set up predictive outputs
        self.build_outputs(dim, num_ft_cnt, bin_ft_cnt, categorical_fts)

        self.to(self.device)

    def build_categorical_input_layers(self, categorical_fts):
        # will compute total number of inputs
        input_dim = 0

        # create categorical variable embedding layers
        for ft, feature in categorical_fts.items():
            n_cats = len(feature["cats"]) + 1
            embed_dim = compute_embedding_size(n_cats)
            embed_layer = torch.nn.Embedding(n_cats, embed_dim)
            self.categorical_embedding[ft] = embed_layer
            self.add_module(f"{ft}_embedding", embed_layer)
            # track embedding inputs
            input_dim += embed_dim

        return input_dim

    def build_layers(self, input_dim):
        """
        Constructs the encoder and decoder layers for the autoencoder model.

        Args:
            input_dim (int): The input dimension of the autoencoder model.

        Returns:
            The output dimension of the encoder layers (int).
        """
        # construct a canned denoising autoencoder architecture
        if self.encoder_layers is None:
            self.encoder_layers = [int(4 * input_dim) for _ in range(3)]

        if self.decoder_layers is None:
            self.decoder_layers = []

        if self.encoder_activations is None:
            self.encoder_activations = [self.activation for _ in self.encoder_layers]

        if self.decoder_activations is None:
            self.decoder_activations = [self.activation for _ in self.decoder_layers]

        if self.encoder_dropout is None or type(self.encoder_dropout) == float:
            drp = self.encoder_dropout
            self.encoder_dropout = [drp for _ in self.encoder_layers]

        if self.decoder_dropout is None or type(self.decoder_dropout) == float:
            drp = self.decoder_dropout
            self.decoder_dropout = [drp for _ in self.decoder_layers]

        for i, dim in enumerate(self.encoder_layers):
            activation = self.encoder_activations[i]
            layer = CompleteLayer(
                input_dim, dim, activation=activation, dropout=self.encoder_dropout[i]
            )
            input_dim = dim
            self.encoder.append(layer)
            self.add_module(f"encoder_{i}", layer)

        for i, dim in enumerate(self.decoder_layers):
            activation = self.decoder_activations[i]
            layer = CompleteLayer(
                input_dim, dim, activation=activation, dropout=self.decoder_dropout[i]
            )
            input_dim = dim
            self.decoder.append(layer)
            self.add_module(f"decoder_{i}", layer)

        return input_dim

    def build_outputs(self, dim, num_ft_cnt, bin_ft_cnt, categorical_fts):
        self.numeric_output = torch.nn.Linear(dim, num_ft_cnt)
        self.binary_output = torch.nn.Linear(dim, bin_ft_cnt)

        for ft, feature in categorical_fts.items():
            cats = feature["cats"]
            layer = torch.nn.Linear(dim, len(cats) + 1)
            self.categorical_output[ft] = layer
            self.add_module(f"{ft}_output", layer)

    def forward(self, input):
        encoding = self.encode(input)
        num, bin, cat = self.decode(encoding)
        return num, bin, cat

    def encode(self, x, layers=None):
        if layers is None:
            layers = len(self.encoder)
        for i in range(layers):
            layer = self.encoder[i]
            x = layer(x)
        return x

    def decode(self, x, layers=None):
        if layers is None:
            layers = len(self.decoder)
        for i in range(layers):
            layer = self.decoder[i]
            x = layer(x)
        num, bin, cat = self.compute_outputs(x)
        return num, bin, cat

    def compute_outputs(self, x):
        num = self.numeric_output(x)
        bin = self.binary_output(x)
        bin = torch.sigmoid(bin)
        cat = []
        for output_layer in self.categorical_output.values():
            out = output_layer(x)
            cat.append(out)
        return num, bin, cat
