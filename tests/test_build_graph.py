from deep_neurographs.intake import build_graph


if __name__ == "__main__":
    # Parmaters
    anisotropy = [1.0, 1.0, 1.0]
    bucket = "aind-msma-morphology-data"
    dataset = "651324"
    block_id = "block_003"
    pred_id = "pred_3"

    # Initializations
    root_path = f"agrim-postprocessing-exps/data/{dataset}/{block_id}"
    label_path = f"{root_path}/labels/{pred_id}.n5"
    swc_path = f"{root_path}/swcs/{pred_id}"
    mistake_log_path = f"{root_path}/mistake_logs/{pred_id}.json"

    # Main
    build_graph(bucket, label_path, swc_path, mistake_log_path)
    print("Done")
