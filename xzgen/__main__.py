import argparse
import multiprocessing
from xzgen import ImageObject
from xzgen import ImageData, Dimension
from xzgen import Scene
import logging
from pathlib import Path
import time

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

def construct_scene(DEBUG_FLAG, scene_ix):
    scene = Scene(dimension, image_data, scene_ix, 0.8, upper_limit=45)
    scene.write_scene(DEBUG_FLAG)
    annots = scene.csvDataTrain.copy()
    return annots

if __name__ == '__main__':
    description = "Data generation for DL models."

    parser = argparse.ArgumentParser(description = description)

    parser.add_argument(
            "-info",
            "--info",
            type = str, 
            required = True,
            help = "Info file."
            )

    parser.add_argument(
            "-pos",
            "--posdir",
            type = str, 
            required = True,
            help = "Directory containing the positive object images."
            )

    parser.add_argument(
            "-neg",
            "--negdir",
            type = str,
            required = True,
            help = "Directory containing the negative object images."
            )

    parser.add_argument(
            "-asis",
            "--asisdir",
            type = str,
            required = True,
            help = "Directory containing the objects with no augmentation."
            )

    parser.add_argument(
            "-occ",
            "--occdir",
            type = str,
            required = True,
            help = "Directory containing the occlusion images."
            )

    parser.add_argument(
            "-bg",
            "--background",
            type = str,
            required = True,
            help = "Directory containing the background images."
            )

    # parser.add_argument(
    #         "-t",
    #         "--type",
    #         type = int,
    #         help = "Option type for the kind of annotation to be performed."
    #         )

    parser.add_argument(
            "-n",
            "--num_images",
            type = int,
            required = True,
            help = "The number of images to be generated."
            )

    parser.add_argument(
            "--debug",
            dest = 'debug',
            default = False,
            action = 'store_true',
            help = "If true, write images with annotated bounding boxes."
            )

    parser.add_argument(
            "--parallel",
            dest = 'parallel',
            default = False,
            action = 'store_true',
            help = "If true, run scene creation tasks in parallel."
            )

    args = parser.parse_args()

    info_file = args.info
    path_pos = args.posdir
    path_neg = args.negdir
    path_nosku = args.asisdir
    path_occ = args.occdir
    path_bg = args.background
    num_images = args.num_images
    DEBUG_FLAG = args.debug
    RUN_PARALLEL = args.parallel

    dimension = Dimension(info_file)
    image_data = ImageData(
        path_pos = path_pos,
        path_neg = path_neg,
        path_nosku = path_nosku,
        path_occ = path_occ,
        path_bg = path_bg,
        sku_list = dimension.sku_list,
        root_dir = Path.cwd())

    csvData = []

    time0 = time.time()

    if RUN_PARALLEL:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            csvData = pool.map(
                partial(construct_scene, DEBUG_FLAG), 
                range(num_images))
    else:
        for scene_ix in range(num_images):
            logger.info(f'Constructing scene {scene_ix}.')
            scene = Scene(dimension, image_data, scene_ix, 0.8, upper_limit=45)
            annots = scene.csvDataTrain.copy()
            csvData.append(annots)
            scene.write_scene(DEBUG_FLAG)

    time1 = time.time()
    logger.info(f'Application took {time1 - time0} secs to run.')

    csvData = [row for scene in csvData for row in scene]

    annotation_file_path = image_data.folder_name.joinpath(
            'train_annotations.csv')

    logger.info(f'Writing annotations file at {annotation_file_path}.')
    with open(annotation_file_path, 'w') as csvFile:
        for row in csvData:
            csvFile.write(",".join(str(r) for r in row) + "\n")

