import os
import argparse
from omegaconf import OmegaConf
import pandas as pd


def strip_trailing_zeros(x):
    s = f"{x:.5f}"
    s = s.rstrip('0').rstrip('.')
    return s


def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flatten a dict (and lists), producing a single-level mapping.
    Lists become either indexed keys or comma-joined strings.
    """
    items = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_dict(v, new_key, sep=sep))
    elif isinstance(d, list):
        # Option A: index each element
        for idx, v in enumerate(d):
            new_key = f"{parent_key}{sep}{idx}"
            items.update(flatten_dict(v, new_key, sep=sep))
        # Option B (uncomment to use): join list into single comma-separated value
        # items[parent_key] = ','.join(str(x) for x in d)
    else:
        items[parent_key] = d
    return items

def load_and_flatten(config_dir):
    """
    Load each .yaml/.yml in `config_dir`, flatten it, and return
    a dict: { config_name: flat_dict }.
    """
    all_configs = {}
    for fname in sorted(os.listdir(config_dir)):
        if not fname.endswith(('.yaml', '.yml')):
            continue
        name = os.path.splitext(fname)[0]
        path = os.path.join(config_dir, fname)
        try:
            cfg = OmegaConf.load(path)
            plain = OmegaConf.to_container(cfg, resolve=True)
            flat = flatten_dict(plain)
            all_configs[name] = flat
        except Exception as e:
            print(f"⚠️ Warning: failed to load {fname}: {e}")
    return all_configs


def main():
    parser = argparse.ArgumentParser(
        description="Read YAML configs, flatten nested fields into columns, export to Excel, and print LaTeX for A,B,C columns"
    )
    parser.add_argument(
        '-i', '--input_dir', default='config',
        help='Directory containing YAML config files'
    )
    parser.add_argument(
        '-o', '--output_file', default='config/configs.xlsx',
        help='Output Excel filename'
    )
    args = parser.parse_args()

    configs_flat = load_and_flatten(args.input_dir)
    if not configs_flat:
        print(f"No configs loaded from {args.input_dir}")
        return

    # Build DataFrame: rows=config names, cols=all flattened keys
    df = pd.DataFrame.from_dict(configs_flat, orient='index')
    df.index.name = 'config'
    df.reset_index(inplace=True)

    # Save to Excel
    df.to_excel(args.output_file, sheet_name='configs', index=False)
    print(f"Saved {len(df)} configs to {args.output_file}")

    df['defaults.2.model'] =  df['defaults.2.model'].str.replace(r'^cifar_mvgg_A64',
                                          r'\\textbf{VGG-9}', regex=True)
    df['defaults.2.model'] =  df['defaults.2.model'].str.replace(r'^cnn1$',
                                          r'\\textbf{6-layer CNN}', regex=True)
    df['defaults.2.model'] =  df['defaults.2.model'].str.replace(r'^resnet_18_for_tiny$',
                                          r'\\textbf{ResNet-18}', regex=True)
    df['defaults.2.model'] =  df['defaults.2.model'].str.replace(r'^rs50.*$', r'\\textbf{ResNet-50}',
                                          regex=True)
    df['defaults.2.model'] =  df['defaults.2.model'].str.replace(r'^vit_l_32$',
                                          r'\\textbf{ViT-L/32}', regex=True)
    df['defaults.2.model'] =  df['defaults.2.model'].str.replace(r'^clip_vit_l_14',
                                          r'\\textbf{CLIP ViT-L/14}',
                                          regex=True)


    df["Train Split"] = 0.8
    df["Dataset Source"] = "ImageNet-1k"
    subset = df[["defaults.2.model", "Train Split", "defaults.1.data"]]

    subset = subset.apply(lambda col: pd.to_numeric(col, errors='ignore'))
    subset = subset.replace('_', ' ', regex=True)
    subset.set_index(["defaults.2.model"], inplace=True)

    print(subset.to_latex(index=True, float_format="%.2f"))

    df["C"] = 1.0
    subset = df[["defaults.2.model", "alpha", "w", "gamma", "C", "fv_sd", "fv_sd", "grad_based"]]

    subset = subset.apply(lambda col: pd.to_numeric(col, errors='ignore'))
    subset = subset.replace('_', ' ', regex=True)
    subset.set_index(["defaults.2.model"], inplace=True)
    print(subset.to_latex(index=True, float_format=strip_trailing_zeros))

    subset = df[["defaults.2.model", "epochs", "lr", "weight_decay", "adam_eps", "batch_size"]]

    subset = subset.apply(lambda col: pd.to_numeric(col, errors='ignore'))
    subset["lr"] = df["lr"].astype(str).values
    subset["adam_eps"] = df["adam_eps"].astype(str).values
    subset = subset.replace('_', ' ', regex=True)
    subset.set_index(["defaults.2.model"], inplace=True)
    print(subset.to_latex(index=True, float_format=strip_trailing_zeros))


    df["Setting"] = df["defaults.2.model"]
    df["Reference"] = df["defaults.2.model"]

    subset = df[["Reference", "defaults.2.model", "defaults.1.data", "Setting"]]

    subset = subset.apply(lambda col: pd.to_numeric(col, errors='ignore'))
    subset = subset.replace('_', ' ', regex=True)
    print(subset.to_latex(index=False, float_format=strip_trailing_zeros))



    subset = df[["defaults.2.model", "eval_fv_sd", "eval_lr", "eval_nsteps", "strategy"]]

    subset = subset.apply(lambda col: pd.to_numeric(col, errors='ignore'))
    subset = subset.replace('_', ' ', regex=True)
    print(subset.to_latex(index=False, float_format=strip_trailing_zeros))

    subset = df[["defaults.2.model", "img_str", "target_img_path"]]
    subset["Image Source"] = "ImageNet-1k"

    subset = subset.apply(lambda col: pd.to_numeric(col, errors='ignore'))
    subset = subset.replace('_', ' ', regex=True)
    subset.set_index(["defaults.2.model"], inplace=True)

    # 2. Print as LaTeX with 2-decimal formatting
    print(subset.to_latex(index=True, float_format="%.5f"))
if __name__ == '__main__':
    main()

