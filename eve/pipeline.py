from eve.config import load_config

def main():
    cfg = load_config("config.yaml")

    print("stages:", cfg.stages)
    print("output format:", cfg.output_format)
    print("files to process:")
    for f in cfg.inputs.get_files():
        print(" -", f)


if __name__ == "__main__":
    main()