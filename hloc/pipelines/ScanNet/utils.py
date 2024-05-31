import logging

import numpy as np

from hloc.utils.read_write_model import read_model, write_model

logger = logging.getLogger(__name__)


def create_reference_sfm(full_model, ref_model, blacklist=None, ext=".bin"):
    """Create a new COLMAP model with only training images."""
    logger.info("Creating the reference model.")
    ref_model.mkdir(exist_ok=True)
    cameras, images, points3D = read_model(full_model, ext)

    if blacklist is not None:
        with open(blacklist, "r") as f:
            blacklist = f.read().rstrip().split("\n")

    images_ref = dict()
    for id_, image in images.items():
        if blacklist and image.name in blacklist:
            continue
        images_ref[id_] = image

    points3D_ref = dict()
    for id_, point3D in points3D.items():
        ref_ids = [i for i in point3D.image_ids if i in images_ref]
        if len(ref_ids) == 0:
            continue
        points3D_ref[id_] = point3D._replace(image_ids=np.array(ref_ids))

    write_model(cameras, images_ref, points3D_ref, ref_model, ".bin")
    logger.info(f"Kept {len(images_ref)} images out of {len(images)}.")


def create_reference_sfm_from_ScanNetDatset(data_path, ref_model, ext=""):
    '''
    Create a new COLMAP model with known camera poses and intrinsics,
    without any points3D as well as 2D points.
    '''
    ref_model.mkdir(exist_ok=True)
    camera = dict()
    images = dict()
    points3D = dict()
    

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default="/media/thuan/8tb/ScanNet/", help="Path to the dataset.")
    parser.add_argument("--ref_model", type=Path, help="Path to the reference model.")
    parser.add_argument("--scene", type=str,default="scene0000_00", help="Scene name.")

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    create_reference_sfm(args.full_model, args.ref_model, args.blacklist, args.ext)
