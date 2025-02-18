from music_df.dedouble import dedouble
from music_df.detremolo import detremolo
from music_df.quantize_df import quantize_df
from music_df.read import read
from music_df.salami_slice import salami_slice


def read_symbolic_score(path: str, quantize: int = 16):
    df = read(path)
    if quantize > 0:
        df = quantize_df(df, tpq=quantize)
    df = detremolo(df)
    df = salami_slice(df)

    # We don't need to quantize here because we already did that above
    df = dedouble(df, quantize=False)
    return df
