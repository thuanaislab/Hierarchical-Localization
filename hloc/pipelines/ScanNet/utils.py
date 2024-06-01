import logging

import numpy as np
import os

from hloc.utils.read_write_model import (read_model, write_model, Camera, Image, Point3D,
                                         rotmat2qvec)

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


def create_reference_sfm_from_ScanNetDatset(data_path, ref_model, ext=".bin"):
    '''
    Create a new COLMAP model with known camera poses and intrinsics,
    without any points3D as well as 2D points.
    '''
    ref_model.mkdir(exist_ok=True)
    cameras = dict()
    images = dict()
    points3D = dict()
    img_color_path = data_path / "color"
    intrinsic_path = data_path / "intrinsic"
    pose_path = data_path / "pose"
    file_list = os.listdir(img_color_path)
    
    # read camera intrinsics
    with open(intrinsic_path / "intrinsic_color.txt", "r") as f:
        intrinsic = f.read().rstrip().split("\n")
        intrinsic = [list(map(float, x.split())) for x in intrinsic]
        intrinsic = np.array(intrinsic)
    cameras[0] = Camera(
                    id=0, model="PINHOLE", width=1296, height=968, params=np.array((intrinsic[0,0], 
                                                                                   intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]))
                )
    for file in file_list:
        image_id = int(file.split(".")[0])
        tmp_path = pose_path / f"{image_id}.txt"
        with open(tmp_path, "r") as f:
            pose = f.read().rstrip().split("\n")
            pose = [list(map(float, x.split())) for x in pose]
            pose = np.array(pose)
        pose = np.linalg.inv(pose)
        t = pose[:3, 3]
        R = pose[:3, :3]
        q = rotmat2qvec(R)
        images[image_id] = Image(
            id=image_id, qvec=q, tvec=t, camera_id=0, name=file, xys=[], point3D_ids=[]
        )
    write_model(cameras, images, points3D, ref_model, ".bin")
    

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default="/media/thuan/8tb/ScanNet/scene0120_01/", 
                        help="Path to the dataset.")
    parser.add_argument("--ref_model", type=Path,default="/media/thuan/8tb/ScanNet_HlocOutut/" , 
                        help="Path to the reference model.")
    
    args = parser.parse_args()
    create_reference_sfm_from_ScanNetDatset(args.data_path, args.ref_model, args.scene)
