from .transformer_encoder import TransformerEncoder
from .rnn_encoder import RnnEncoder
from .rnn_encoder import LstmEncoder
from .rnn_encoder import GruEncoder
from .rnn_encoder import BirnnEncoder
from .rnn_encoder import BilstmEncoder
from .rnn_encoder import BigruEncoder
from .cnn_encoder import GatedcnnEncoder


str2encoder = {"transformer": TransformerEncoder, "rnn": RnnEncoder, "lstm": LstmEncoder,
               "gru": GruEncoder, "birnn": BirnnEncoder, "bilstm": BilstmEncoder, "bigru": BigruEncoder,
               "gatedcnn": GatedcnnEncoder}

__all__ = ["TransformerEncoder", "RnnEncoder", "LstmEncoder", "GruEncoder", "BirnnEncoder",
           "BilstmEncoder", "BigruEncoder", "GatedcnnEncoder", "str2encoder"]

