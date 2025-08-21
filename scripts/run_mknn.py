from pu.metrics import mknn
import polars as pl

if __name__ == "__main__":
    df = pl.read_parquet("data/legacysurvey_convnext.parquet")
    print(mknn(df["astropt_nano_legacysurvey"], df["astropt_tiny_legacysurvey"]))
    print(mknn(df["astropt_tiny_legacysurvey"], df["astropt_base_legacysurvey"]))
    print(mknn(df["astropt_base_legacysurvey"], df["astropt_large_legacysurvey"]))
    print("=========")
    print(mknn(df["astropt_nano_hsc"], df["astropt_tiny_hsc"]))
    print(mknn(df["astropt_tiny_hsc"], df["astropt_base_hsc"]))
    print(mknn(df["astropt_base_hsc"], df["astropt_large_hsc"]))
