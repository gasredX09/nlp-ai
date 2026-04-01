from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 100
    d_model: int = 128
    num_heads: int = 8
    d_ff: int = 512
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    max_seq_len: int = 64
    dropout: float = 0.1
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads: ({self.num_heads})."
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads
