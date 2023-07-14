import os
import zarr
import numpy as np
from aind_ng_mesh.meshing import labels_to_meshes
from aind_ng_mesh.io_utils import to_s3, write_json, write_to_local
from aind_segmentation_evaluation.evaluate import run_evaluation
from aind_segmentation_evaluation.conversions import swc_to_graph, graphs_to_swc
from neurotorch.core.predictor import Predictor
from neurotorch.nets.RSUNet import RSUNet
from tifffile import imread, imwrite
from time import time
from toolbox.inference import run_inference, postprocess_graph
from toolbox.skeletonization import labels_to_graph
from toolbox.utils import mkdir


def main():
    predictor = Predictor(RSUNet(), path_to_ckpt)
    for dataset in test_data.keys():
        print("   exaSPIM_" + dataset)
        for block_id in test_data[dataset]:
            # Read data
            metadata = dict()
            input = read_block(dataset, block_id, "input.n5")
            target_labels = read_block(dataset, block_id, "Fill_Label_Mask.n5")
            target_graph = read_graph(dataset, block_id, target_labels)
            print("   " + block_id + " - " + str(input.shape))

            # Generate prediction
            metadata["inference_parameters"] = {
                "ckpt": ckpt,
                "dust_threshold": dust_threshold,
                "permute": permute,
                "shift": shift,
                "waterz_thresholds": waterz_thresholds,
            }
            pred_labels = run_inference(
                input,
                predictor,
                batch_size,
                bbox,
                dust_threshold,
                waterz_thresholds,
                shift,
            )

            # Graph postprocessing
            metadata.update(
                {
                    "graph_processing": {
                        "break_crossovers": break_crossovers,
                        "eval": eval,
                    }
                }
            )
            pred_graph = labels_to_graph(pred_labels)
            if break_crossovers:
                pred_graph = postprocess_graph(pred_graph, pred_labels.shape)

            # Run evaluation
            if eval:
                stats = run_evaluation(
                    target_graph,
                    target_labels,
                    pred_graph,
                    pred_labels,
                )
                print(stats)
                metadata.update(
                    {
                        "eval_stats": stats,
                    }
                )

            # Write to local machine
            upload_dir = os.path.join(root_dir, "temp")
            block_root_dir = f"{upload_dir}/{dataset}/{block_id}"
            mkdir(upload_dir)
            mkdir(f"{upload_dir}/{dataset}")
            mkdir(block_root_dir)

            # Write labels
            blocks_dir = os.path.join(block_root_dir, "labels")
            path = os.path.join(blocks_dir, pred_id + ".n5")
            zarr.save(path, to_zarr(pred_labels))
            if write_target:
                path = os.path.join(blocks_dir, "target.n5")
                zarr.save(path, to_zarr(target_labels))

            # Write swcs
            swcs_dir = os.path.join(block_root_dir, "swcs")
            mkdir(swcs_dir)
            graphs_to_swc(
                pred_graph,
                os.path.join(swcs_dir, pred_id),
                permute=permute,
                scale=scale,
            )
            if write_target:
                graphs_to_swc(
                    target_graph,
                    os.path.join(swcs_dir, "target"),
                    permute=permute,
                    scale=scale,
                )

            # Write meshes
            meshes_dir = os.path.join(block_root_dir, "meshes")
            mkdir(meshes_dir)
            pred_meshes = labels_to_meshes(pred_labels)
            write_to_local(pred_labels, pred_meshes, os.path.join(meshes_dir, pred_id))
            if write_target:
                target_meshes = labels_to_meshes(target_labels)
                write_to_local(
                    target_labels, target_meshes, os.path.join(meshes_dir, "target")
                )

            # Write to s3
            s3_path = f"agrim-postprocessing-exps/data"
            ng_path = f"precomputed://s3://{bucket}/{s3_path}/{dataset}/{block_id}/meshes/{pred_id}"
            metadata.update({"ng_path": ng_path})
            write_json(os.path.join(block_root_dir, "metadata.json"), metadata)
            if upload:
                to_s3(upload_dir, bucket, s3_path)
            print("Neuroglancer Path:", ng_path)


def read_block(dataset, block_id, f):
    path = f"{data_dir}/{dataset}/blocks/{block_id}/{f}"
    return zarr.open(zarr.N5FSStore(path), "r").volume[:]


def read_graph(dataset, block_id, block):
    path = f"{data_dir}/{dataset}/swcs/{block_id}/final-trees"
    shape = block.shape
    graph = swc_to_graph(path, shape, permute=permute, scale=scale)
    return graph


def to_zarr(block):
    zarr_block = zarr.array(
        block,
        chunks=(128, 128, 128),
        dtype=block.dtype,
    )
    return zarr_block


if __name__ == "__main__":
    # Upload parameters
    bucket = "aind-msma-morphology-data"
    write_target = True
    pred_id = "pred_1"
    test_data = {
        "651324": ["block_003"],
    }

    # Inference parameters
    batch_size = 8
    bbox = (128, 128, 128)
    ckpt = "stats_best-88.285-1000.ckpt"
    dust_threshold = 1200
    shift = 0
    waterz_thresholds = [-50]

    permute = [2, 1, 0]
    scale = [1.0, 1.0, 1.0]

    break_crossovers = False
    eval = True
    upload = True

    # Initialize paths
    root_dir = "/home/jupyter/workspace"
    ckpt_dir = "/home/jupyter/workspace/outputs/ckpts/postprocessing"
    path_to_ckpt = os.path.join(ckpt_dir, ckpt)
    data_dir = os.path.join(root_dir, "data")

    # Model evaluation
    main()
