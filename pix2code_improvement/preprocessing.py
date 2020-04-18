import logging
import numpy as np
from pathlib import Path
import math
from constants import (DOMAIN_LANGUAGES, ORIGINAL_DATA_FOLDER_NAME, TRAINING_DATA_FOLDER_NAME,
                       VALIDATION_DATA_FOLDER_NAME, TEST_DATA_FOLDER_NAME, IMAGE_SIZE)
import shutil
from vocabulary import load_voc_from_paths
from utils import preprocess_image, load_code
from typing import List
from tqdm import tqdm

logger = logging.getLogger()


def preprocess_data(datasets_folder_path: str, domain_languages: List[str] = DOMAIN_LANGUAGES,
                    samples_to_ignore: List[str] = [], seed: int = 1234,
                    validation_ratio: float = 0.1, test_ratio: float = 0.1, image_size: int = IMAGE_SIZE):
    """
    Splits the data in the dataset folder into training, evaluation and testing sets and move the data to appropriate
    folders. Moreover, the function also preprocesses the data and creates vocabulary for the files in the appropriate
    file structure.
    :param datasets_folder_path: The path to the main dataset folder (which contain subfolders for each domain)
    :param domain_languages: The domains to preprocess
    :param samples_to_ignore: If certain samples (file-stems) are to be ignored due
    :param seed: The seed for the train-valid-test split randomization
    :param validation_ratio: The ratio of files to be moved to the validation set
    :param test_ratio: The ratio of files to be moved to the test set
    :param image_size: The image size to preprocess for
    :return: None
    """
    datasets_folder_path = Path(datasets_folder_path)
    for domain_language in domain_languages:
        logger.info("Preprocessing domain: {}".format(domain_language))
        logger.debug("Setting up folders and paths".format(domain_language))
        domain_folder = datasets_folder_path / domain_language
        source_data_folder = domain_folder / ORIGINAL_DATA_FOLDER_NAME
        training_data_folder_path = domain_folder / TRAINING_DATA_FOLDER_NAME
        validation_data_folder_path = domain_folder / VALIDATION_DATA_FOLDER_NAME
        test_folder_data_path = domain_folder / TEST_DATA_FOLDER_NAME
        new_data_folders = [training_data_folder_path, validation_data_folder_path, test_folder_data_path]
        vocabulary_path = domain_folder / 'vocabulary.vocab'
        logger.debug("Creating folders")
        for folder in [domain_folder, source_data_folder] + new_data_folders:
            folder.mkdir(exist_ok=True)

        # Find the paths to put into training, validation and testing set.
        # Note that only unique code pieces are used (duplicates skipped)
        logger.debug("Finding unique samples")
        ignored_files_count = 0
        gui_codes = set()
        paths = []
        for gui_code_file_path in source_data_folder.glob("*.gui"):
            if gui_code_file_path.stem in samples_to_ignore:
                logger.info("Ignoring sample {}".format(gui_code_file_path.stem))
                continue
            gui_code = load_code(gui_code_file_path)
            
            image_file_path = gui_code_file_path.with_suffix(".png")
            if not image_file_path.exists():
                logger.warning("Missing image file for sample {}".format(gui_code_file_path.stem))
                continue
            if gui_code in gui_codes:
                ignored_files_count += 1
                continue
            else:
                gui_codes.add(gui_code)
                paths.append(gui_code_file_path)
        logger.info("{} unique samples out of {} were successfully found".format(len(paths),
                                                                                 len(paths)+ignored_files_count))
        logger.debug("Creating vocabulary")
        voc = load_voc_from_paths(paths)
        voc.save(vocabulary_path)

        logger.debug("Copying and preprocessing files")
        # Split the data into train, validation and test set
        np.random.seed(seed)
        np.random.shuffle(paths)
        test_data_len = int(math.ceil(len(paths) * test_ratio))
        test_paths = paths[0:test_data_len]
        validation_paths = paths[test_data_len:(test_data_len + int(math.ceil(len(paths) * validation_ratio)))]
        train_paths = list(set(paths) - set(test_paths) - set(validation_paths))

        # For each folder, copy the relevant files and preprocess
        for path_set, folder_path in [(train_paths, training_data_folder_path),
                                      (validation_paths, validation_data_folder_path),
                                      (test_paths, test_folder_data_path)]:
            logger.debug("Copying and preprocessing {} files for {}".format(len(path_set), folder_path.name))
            nr_of_instances = 0
            folder_path.mkdir(exist_ok=True)
            for path in tqdm(path_set):
                code = load_code(path)
                img_data = preprocess_image(str(path.with_suffix(".png")), image_size)

                np.savez_compressed(folder_path / (path.name[:-4] + "_{}.npz".format(image_size)),
                                    img_data=img_data, code=code)
                # Also copy the original files
                shutil.copy(str(path), str(folder_path / path.name))  # Gui files
                shutil.copy(str(path.with_suffix(".png")), str(folder_path / "{}.png".format(path.stem)))  # Images
                nr_of_instances += len(code.split(" "))
            with open(folder_path / 'nr_of_instances.txt', 'w') as f:
                f.write(str(nr_of_instances))
        logger.debug("Done")
