import argparse
from xzgen import ImageObject
from xzgen import ImageData, Dimension
from xzgen import Scene
import logging
import csv
from pathlib import Path

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

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

    args = parser.parse_args()

    info_file = args.info
    path_pos = args.posdir
    path_neg = args.negdir
    path_nosku = args.asisdir
    path_occ = args.occdir
    path_bg = args.background
    num_images = args.num_images
    # anno_option = args.type

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
    for scene_ix in range(num_images):
        logger.info(f'Constructing scene {scene_ix}.')
        scene = Scene(dimension, image_data, scene_ix, 0.8, upper_limit=45)
        annots = scene.csvDataTrain.copy()
        csvData.append(annots)
        scene.write_scene()

    annotation_file_path = image_data.folder_name.joinpath(
            'train_annotations.csv')

    logger.info(f'Writing annotations file at {annotation_file_path}.')
    with open( annotation_file_path, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)

