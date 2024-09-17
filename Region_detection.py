import cv2
import numpy as np
from ultralytics import YOLO
import json
from ultralytics.utils.plotting import Annotator


class Manages_Parts:
    """Manages parking occupancy and availability using YOLOv8 for real-time monitoring and visualization."""

    def __init__(
        self,
        model_path,
        txt_color=(0, 0, 0),
        bg_color=(255, 255, 255),
        occupied_region_color=(0, 255, 0),
        available_region_color=(0, 0, 255),
        margin=10,
    ):
        """
        Initializes the parking management system with a YOLOv8 model and visualization settings.

        Args:
            model_path (str): Path to the YOLOv8 model.
            txt_color (tuple): RGB color tuple for text.
            bg_color (tuple): RGB color tuple for background.
            occupied_region_color (tuple): RGB color tuple for occupied regions.
            available_region_color (tuple): RGB color tuple for available regions.
            margin (int): Margin for text display.
        """
        # Model path and initialization
        self.model_path = model_path
        self.model = self.load_model()

        # Labels dictionary
        self.labels_dict = {"Correct_parts": 0, "Empty_parts": 0}

        # Visualization details
        self.margin = margin
        self.bg_color = bg_color
        self.txt_color = txt_color
        self.occupied_region_color = occupied_region_color
        self.available_region_color = available_region_color

        self.window_name = "Ultralytics YOLOv8 Parking Management System"
        # Check if environment supports imshow

    def load_model(self):
        """Load the Ultralytics YOLO model for inference and analytics."""

        return YOLO(self.model_path)

    @staticmethod
    def parking_regions_extraction(json_file):
        """
        Extract parking regions from json file.

        Args:
            json_file (str): file that have all parking slot points
        """
        with open(json_file, "r") as f:
            return json.load(f)

    def process_data(self, json_data, im0, boxes, clss):
        """
        Process the model data for parking lot management.

        Args:
            json_data (str): json data for parking lot management
            im0 (ndarray): inference image
            boxes (list): bounding boxes data
            clss (list): bounding boxes classes list

        Returns:
            filled_slots (int): total slots that are filled in parking lot
            empty_slots (int): total slots that are available in parking lot
        """
        annotator = Annotator(im0)
        empty_slots, filled_slots = len(json_data), 0
        for region in json_data:
            points_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            class_name = region["class"]
            region_occupied = False

            for box, cls in zip(boxes, clss):
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)
                text = f"{self.model.names[int(cls)]}"

                # annotator.display_objects_labels(
                #     im0, text, self.txt_color, self.bg_color, x_center, y_center, self.margin
                # )
                dist = cv2.pointPolygonTest(points_array, (x_center, y_center), False)
                if dist >= 0:
                    if class_name == cls:
                        region_occupied = True
                        break
                    else:
                        region_occupied = False

            color = self.occupied_region_color if region_occupied else self.available_region_color
            cv2.polylines(im0, [points_array], isClosed=True, color=color, thickness=2)
            if region_occupied:
                filled_slots += 1
                empty_slots -= 1

        self.labels_dict["Correct_parts"] = filled_slots
        self.labels_dict["Empty_parts"] = empty_slots

        annotator.display_analytics(im0, self.labels_dict, self.txt_color, self.bg_color, self.margin)

    def display_frames(self, im0):
        """
        Display frame.

        Args:
            im0 (ndarray): inference image
        """
        if self.env_check:
            cv2.namedWindow(self.window_name)
            cv2.imshow(self.window_name, im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
