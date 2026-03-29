from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\hsu96\OneDrive\Desktop\mysql_csv_exports")
DIR_MLP   = ROOT / "mlp_player_classifier"

DIR_TEST  = ROOT / "test"
OUT_DIR   = DIR_MLP / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_KFEATS   = DIR_TEST / "kmeans_features.xlsx"
FILE_LABELED  = DIR_TEST / "labeled.xlsx"
FILE_KRESULTS = DIR_TEST / "kmeans_results.xlsx"

def read_xlsx(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_excel(path)

def has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)

def main():
    dfX = read_xlsx(FILE_KFEATS)
    dfY = read_xlsx(FILE_LABELED)

    prefer_keys = ["server","player_id","row_idx"]
    time_keys_x = ["player_id","time"]
    time_keys_y = ["player_id","time"]

    if has_cols(dfX, prefer_keys) and has_cols(dfY, prefer_keys):
        join_keys = prefer_keys
    elif has_cols(dfX, time_keys_x) and has_cols(dfY, time_keys_y):
        join_keys = time_keys_x  # (= time_keys_y)
    elif has_cols(dfX, ["player_id"]) and has_cols(dfY, ["player_id"]):
        join_keys = ["player_id"]
    else:
        raise RuntimeError(f"cant conbine, [check] X  {list(dfX.columns)[:10]}..., Y  {list(dfY.columns)[:10]}...")

    label_col = "final_label" if "final_label" in dfY.columns else ("label" if "label" in dfY.columns else None)
    if label_col is None:
        raise RuntimeError("labeled.xlsx cant find final_label or label ")

    keep_y = list(dict.fromkeys(join_keys + [label_col]))
    df = pd.merge(dfX, dfY[keep_y].drop_duplicates(), on=join_keys, how="inner")

    if FILE_KRESULTS.exists():
        dfK = read_xlsx(FILE_KRESULTS)
        if has_cols(dfK, join_keys):
            keep_k = list(dict.fromkeys(join_keys + [c for c in dfK.columns if c.startswith(("_cluster","kmeans","dist_","km_"))]))
            if len(keep_k) > len(join_keys):
                df = pd.merge(df, dfK[keep_k].drop_duplicates(), on=join_keys, how="left")

    if "afktime" in df.columns and "AFK_ratio" not in df.columns:
        df["AFK_ratio"] = df["afktime"] / 1800.0
    if "afktime" in df.columns and "active_minutes" not in df.columns:
        df["active_minutes"] = np.maximum(0.0, 30.0 - df["afktime"]/60.0)


    feature_cols = [c for c in df.columns if c.startswith("rate_")]
    feature_cols += [c for c in df.columns if c.startswith("dist_")]
    for extra in ["_cluster","pmax","dist_to_assigned","min_dist","AFK_ratio","active_minutes"]:
        if extra in df.columns and extra not in feature_cols:
            feature_cols.append(extra)

    id_cols = [c for c in ["server","player_id","row_idx","time"] if c in join_keys or c in df.columns]
    base_cols = list(dict.fromkeys(["player_id"] + id_cols))
    out_cols = list(dict.fromkeys(base_cols + [label_col] + feature_cols))

    df_out = df[out_cols].copy()
    df_out.rename(columns={label_col: "label"}, inplace=True)

    out_path = OUT_DIR / "samples.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("finish: ", out_path)
    print(f"    {len(df_out)} rows、{len(df_out.columns)} colums")
    print("   join keys =", join_keys)
    print("   label  =", label_col)
    print("   numbers of features =", len(feature_cols))

if __name__ == "__main__":
    main()
