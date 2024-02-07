import os , json
import cv2
from tqdm import tqdm
import logging
from PIL import Image
import numpy as np
from .sentinel_api import sentinel_query
from .tools import start_end_time_interpreter, xyz2bbox, xyz2bbox_territory, coords_in_a_xyz, image_dir_in_image_db, coords_2_xyz_newton
from .inference_modular import ship_detection
from fetch_data.models import SatteliteImage, DetectedObject, WaterCraft, QueuedTask

images_db_path = r"E:\SatteliteImages_db"
model_path = r"E:\WebApps\Satellite_monitoring-web\fetch_data\utilities\inference_models\best_model.pth"
concated_images_path = r"E:\SatteliteImages_db\concated_images"

def store_image(x, y, zoom, start_date, end_date, n_days_before_date=None, date=None):
    global image_db_pat
    image, timestamp = sentinel_query(coords=(x, y, zoom), start=start_date, end=end_date, n_days_before_date=n_days_before_date, date=date, output_img=True, output_timestamp=True)

    if os.path.exists(images_db_path) == False:
        os.mkdir(images_db_path)
    path_z = os.path.join(images_db_path, str(zoom))
    if os.path.exists(path_z) == False:
        os.mkdir(path_z)
    path_zx = os.path.join(path_z, str(x))
    if os.path.exists(path_zx) == False:
        os.mkdir(path_zx)
    path_zxy = os.path.join(path_zx, str(y))
    if os.path.exists(path_zxy) == False:
        os.mkdir(path_zxy)

    image_path = os.path.join(path_zxy, f"{timestamp}.png")
    image.save(image_path)


def territory_fetch_inference(x_range, y_range, zoom, start_date, end_date, child_task_id, parent_task_id, parent_queries_done, parent_total_queries,
                              subtasks, overwrite_repetitious=False, images_db_path=images_db_path, inference=True, save_concated=False, confidence_threshold=0.9):
    
    child_task = QueuedTask.objects.get(task_id=child_task_id)
    if subtasks:
        parent_task = QueuedTask.objects.get(task_id=parent_task_id)
    child_total_queries = (x_range[1] - x_range[0] + 1) * (y_range[1] - y_range[0] + 1)
    child_queries_done = 0


    date_data = start_end_time_interpreter(start=start_date, end=end_date)
    start_date, start_formatted = date_data["start_date"], date_data["start_formatted"]
    end_date, end_formatted = date_data["end_date"], date_data["end_formatted"]
    timestamp = date_data["timestamp"]

    if os.path.exists(images_db_path) == False:
        os.mkdir(images_db_path)
    path_z = os.path.join(images_db_path, str(zoom))
    if os.path.exists(path_z) == False:
        os.mkdir(path_z)
    images_meta= []
    for i in tqdm(range(x_range[0], x_range[1]+1)):
        path_zx = os.path.join(path_z, str(i))
        if os.path.exists(path_zx) == False:
            os.mkdir(path_zx)
        for j in tqdm(range(y_range[0], y_range[1]+1)):
            child_queries_done += 1
            if subtasks:
                parent_queries_done += 1
            if (child_queries_done % 10 == 0) or (child_queries_done == child_total_queries):
                child_task.fetch_progress = int(child_queries_done * 100 / child_total_queries)
                child_task.save()
                if subtasks:
                    parent_task.fetch_progress = int(parent_queries_done * 100 / parent_total_queries)
                    parent_task.save()
            path_zxy = os.path.join(path_zx, str(j))
            if os.path.exists(path_zxy) == False:
                os.mkdir(path_zxy)
            
            image_path = image_dir_in_image_db(i, j, zoom, timestamp, base_dir=images_db_path)
            images_meta.append((i, j, image_path))
            if overwrite_repetitious or (os.path.exists(image_path) == False):
                image, url = sentinel_query(coords=(i, j, zoom), start_formatted=start_formatted, end_formatted=end_formatted, output_img=True, output_url=True)
                # lonmin, latmin, lonmax, latmax = map(lambda i: round(i,6), xyz2bbox((i, j, zoom)))
                image.save(image_path)
            lonmin, latmin, lonmax, latmax = map(lambda i: round(i,6), xyz2bbox((i, j, zoom)))
            # SatteliteImage.objects.update_or_create(image_path=image_path, x=i, y=j, zoom=zoom, time_from=start_date,
            #                                         time_to=end_date, bbox_lon1=lonmin, bbox_lat1=latmin,
            #                                         bbox_lon2=lonmax, bbox_lat2=latmax, data_source="Sentinel2")
            img_obj, created = SatteliteImage.objects.update_or_create(image_path=image_path)
            img_obj_attrs = {"x": i, "y": j, "zoom":zoom, "time_from":start_date, "time_to":end_date, "bbox_lon1":lonmin, "bbox_lat1":latmin,
                             "bbox_lon2":lonmax, "bbox_lat2":latmax, "data_source":"Sentinel2"}
            for key, value in img_obj_attrs.items():
                setattr(img_obj, key, value)
            img_obj.save()
                
                
    if inference:
        child_task.task_status = "inferencing"
        child_task.save()
        logging.info("Inferencing began")
        concated_img = concatenate_image(x_range, y_range, zoom, start=start_date, end=end_date, images_db_path=images_db_path,
                                         return_img=True, save_img=False)
        logging.info("Images concatenated for Inferencing")

        global model_path
        coords = xyz2bbox_territory(x_range, y_range, zoom)
        detection_results = ship_detection(concated_img, model_or_model_path=model_path, bbox_coord_wgs84=coords, model_input_dim=768,
                                           confidence_threshold=confidence_threshold, scale_down_factor=1, sahi_overlap_ratio=0.1, nms_iou_threshold=0.15, 
                                           device='adaptive', output_dir=None, output_name="prediction", save_annotated_image=False, output_original_image=False,
                                           output_annotated_image=True, annotations=["score", "length", "coord"], annotation_font=r"calibri.ttf",annotation_font_size=12,
                                           annotation_bbox_width=1, constraints={"length": (10,620)})
        logging.info("Inferencing ended")
        ships_data = detection_results["ships_data"]

        inference_results_coords = detection_results["ships_lon_lat"]
        inference_results_lengths = detection_results["ships_lengths"]
        inference_results_scores = detection_results["scores"].tolist()
        inference_results_bboxes = detection_results["bboxes"].tolist()
        inference_results = {"coords": inference_results_coords, "lengths": inference_results_lengths,
                             "scores": inference_results_scores, "bboxes": inference_results_bboxes}
        inference_results_json = json.dumps(inference_results)
        child_task.inference_result = inference_results_json
        child_task.save()
        print("\n\n\nchild_task.inference_results =", child_task.inference_result, '\n\n\n')
        del inference_results_coords, inference_results_lengths, inference_results_scores, inference_results_bboxes

        annotated_img = detection_results["annotated_image"]
    
        logging.info("Annotated image is being deconcatenated for storing")
        deconcated_annotated_images = deconcat_image(annotated_img, x_range, y_range)
        for x, y, img in deconcated_annotated_images:
            annot_img_path = image_dir_in_image_db(x, y, zoom, timestamp, base_dir=images_db_path, annotation_mode=True)
            img.save(annot_img_path)
            image_obj = SatteliteImage.objects.get(image_path=image_path)
            image_obj.annotated_image_path = annot_img_path
            image_obj.save()
        logging.info("All deconcatenated images paths saved in the SatteliteImage table")

        for obj, data in ships_data.items():
            for x, y, img_path in images_meta:
                if coords_in_a_xyz(data["lon_lat"], (x, y, zoom)):
                    lon, lat = data["lon_lat"]
                    length = data["length"]
                    confidence = data["confidence"]
                    # watercraft_type = data["watercraft_type"]
                    # watercraft_name = data["watercraft_name"]
                    # awake = data["awake"]
                    # object_type = WaterCraft.objects.get(name=watercraft_name)

                    object_type = WaterCraft.objects.get(name="Unknown")   # !!!!!!!!!!!!!!!!!!!  DEBUG MODE  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # awake = True                                     # !!!!!!!!!!!!!!!!!!!  DEBUG MODE  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    obj_id = f"x{x}_y{y}_z{zoom}_({timestamp})_{obj}"
                    image_path = image_dir_in_image_db(x, y, zoom, timestamp, base_dir=images_db_path)

                    # print(f"\n\n\nimg_path\n{img_path}\n\n\n")
                    source_img = SatteliteImage.objects.get(image_path=img_path)
                    obj_attrs = {"lon":lon, "lat":lat, "time_from":start_date, "time_to":end_date, "confidence":confidence, "length":length,
                                 "object_type":object_type, #"awake":awake,
                                       }
                    detected_obj, created = DetectedObject.objects.update_or_create(id=obj_id, image=source_img)
                    if subtasks:
                        detected_obj.task.add(child_task, parent_task)
                    else:
                        detected_obj.task.add(child_task)

                    for key, value in obj_attrs.items():
                        setattr(detected_obj, key, value)
                    detected_obj.save()
        logging.info("All detected objects meta data added to DetectedObject table")
        
        child_task.task_status = "inferenced"
        child_task.save()
    if save_concated:
        logging.info("Concatenated image is being saved")
        if os.path.exists(concated_images_path) is False:
            os.mkdir(concated_images_path)
        concated_img_path = os.path.join(concated_images_path, fr"x({x_range[0]}_{x_range[1]})-y({y_range[0]}_{y_range[1]})-z({zoom})-{timestamp}.png")
        concated_img.save(concated_img_path)
        logging.info("Concatenated image saved")

    if save_concated and inference:
        logging.info("Annotated concatenated image is being saved")
        annotated_concated_img_path = os.path.join(concated_images_path, fr"x({x_range[0]}_{x_range[1]})-y({y_range[0]}_{y_range[1]})-z({zoom})-{timestamp}_annotated.png")
        annotated_img.save(annotated_concated_img_path)
        logging.info("Annotated concatenated image saved")
    return parent_queries_done



def concatenate_image(x_range, y_range, zoom, start=None, end=None, annotated=False ,images_db_path=images_db_path, return_img=True,
                      save_img=False, save_img_path=concated_images_path):
    date_data = start_end_time_interpreter(start=start, end=end)
    timestamp = date_data["timestamp"]
    path_z = os.path.join(images_db_path, str(int(zoom)))
    images_horizontally = []
    for j in range(y_range[0], y_range[1]+1):
        if annotated:
            images_row_path = [os.path.join(path_z, str(i), str(j), f"{timestamp}_annotated.png") for i in range(x_range[0], x_range[1]+1)]
        else:
            images_row_path = [os.path.join(path_z, str(i), str(j), f"{timestamp}.png") for i in range(x_range[0], x_range[1]+1)]
        images_row = [np.array(Image.open(img_path)) for img_path in images_row_path]
        images_horizontally.append(cv2.hconcat(images_row))
    concated_image = cv2.vconcat(images_horizontally)
    concated_image = Image.fromarray(concated_image.astype('uint8')).convert('RGB')
    logging.info("Images concateated")
    if save_img:
        if save_img_path is None:
            raise ValueError("You must specify save_img_path")
        if os.path.exists(save_img_path) is False:
            os.mkdir(save_img_path)
        concated_img_path = os.path.join(save_img_path, fr"x({x_range[0]}_{x_range[1]})-y({y_range[0]}_{y_range[1]})-z({zoom})-{timestamp}.png")
        concated_image.save(concated_img_path)
    if return_img:
        return concated_image
    else:
        return concated_img_path
    

def deconcat_image(concated_image, x_range, y_range):
    concated_image = np.array(concated_image)
    m = y_range[1] - y_range[0] + 1
    n = x_range[1] - x_range[0] + 1
    
    height = concated_image.shape[0] // m
    width = concated_image.shape[1] // n

    small_images = []

    for idy, y in enumerate(range(y_range[0], y_range[1] + 1)):
        for idx, x in enumerate(range(x_range[0], x_range[1] + 1)):
            img = concated_image[idy * height: (idy + 1) * height, idx * width: (idx + 1) * width]
            img = Image.fromarray(img.astype('uint8'))
            small_images.append((x, y, img))

    return small_images