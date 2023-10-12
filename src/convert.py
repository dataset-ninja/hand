import os
import shutil
from urllib.parse import unquote, urlparse

import scipy.io
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import file_exists, get_file_name
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    train_data_path = os.path.join("archive","hand_dataset","training_dataset","training_data")
    val_data_path = os.path.join("archive","hand_dataset","validation_dataset","validation_data")
    test_data_path = os.path.join("archive","hand_dataset","test_dataset","test_data")
    batch_size = 30
    images_folder = "images"
    bboxes_folder = "annotations"
    bbox_ext = ".mat"

    ds_name_to_data = {"train": train_data_path, "val": val_data_path, "test": test_data_path}


    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        file_name = get_file_name(image_path)

        ann_path = os.path.join(bboxes_path, file_name + bbox_ext)

        if file_exists(ann_path):
            mat = scipy.io.loadmat(ann_path)["boxes"]
            for coords in mat[0]:
                for curr_coords in coords[0]:
                    exterior = []
                    tags = []
                    for idx, curr_coord in enumerate(curr_coords):
                        if len(curr_coord) == 0:
                            continue
                        if idx == 4:
                            tag_meta = tag_to_data.get(curr_coord[0])
                            if tag_meta is not None:
                                tag = sly.Tag(tag_meta)
                                tags.append(tag)
                        elif idx < 4:
                            exterior.append([curr_coord[0][0], curr_coord[0][1]])

                    polygon = sly.Polygon(exterior)
                    label_poly = sly.Label(polygon, obj_class, tags=tags)
                    labels.append(label_poly)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)


    obj_class = sly.ObjClass("hand", sly.Polygon)
    tag_left = sly.TagMeta("left", sly.TagValueType.NONE)
    tag_right = sly.TagMeta("right", sly.TagValueType.NONE)
    tag_to_data = {"L": tag_left, "R": tag_right}
    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[obj_class], tag_metas=[tag_left, tag_right])
    api.project.update_meta(project.id, meta.to_json())

    for ds_name, curr_data_path in ds_name_to_data.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        images_path = os.path.join(curr_data_path, images_folder)
        bboxes_path = os.path.join(curr_data_path, bboxes_folder)

        images_names = os.listdir(images_path)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(images_path, image_name) for image_name in images_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    return project
