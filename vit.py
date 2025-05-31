import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor,
    VitPoseForPoseEstimation,
    DetrImageProcessor,
    DetrForObjectDetection,
)


class PoseEstimator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load object detection model
        self.detection_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(self.device)
        
        # Load pose estimation model
        self.pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        self.pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=self.device).to(self.device)
    
    def detect_person(self, image, threshold=0.9):
        inputs = self.detection_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.detection_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.detection_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

        person_box = None
        confidence = None
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            class_name = self.detection_model.config.id2label[label.item()]
            if class_name == 'person':
                if person_box is None or confidence < score.item():
                    person_box = [round(i, 2) for i in box.tolist()]
                    confidence = score.item()

        return image, person_box
    
    def estimate_pose(self, image, person_box):
        x1, y1, x2, y2 = person_box
        width, height = x2 - x1, y2 - y1
        box_to_use = [x1, y1, width, height]
        
        inputs = self.pose_processor(image, boxes=np.array([[box_to_use]]), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.pose_model(**inputs)
        
        image_pose_result = self.pose_processor.post_process_pose_estimation(outputs, boxes=np.array([[box_to_use]]), threshold=0.3)
        print(image_pose_result[0][0])
        pose_results = {}
        for keypoint, label, score in zip(
            image_pose_result[0][0]["keypoints"], image_pose_result[0][0]["labels"], image_pose_result[0][0]["scores"]
        ):
            keypoint_name = self.pose_model.config.id2label[label.item()]
            x, y = keypoint
            print(f" - {keypoint_name}: x={x.item():.2f}, y={y.item():.2f}, score={score.item():.2f}")
            pose_results[keypoint_name] = (int(x.item()), int(y.item()))
        return pose_results
    
    def draw_results(self, image, person_box, pose_results, output_path="pose_estimation_result.png"):
        draw = ImageDraw.Draw(image)
        draw.rectangle(person_box, outline='red', width=3)
        
        # Draw keypoints
        for keypoint_name, coordinates in pose_results.items():
            x, y = coordinates
            draw.ellipse((x-5, y-5, x+5, y+5), fill=(255, 0, 0))
            # Optionally add keypoint name text
            draw.text((x+10, y), keypoint_name, fill=(255, 0, 0))
            
        image.save(output_path)
        image.show()
    
    def process_image(self, image_path):
        image, person_box = self.detect_person(image_path)
        if not person_box:
            print("No person detected.")
            return
        
        pose_results = self.estimate_pose(image, person_box)
        if not pose_results:
            print("No pose detected.")
            return
        
        self.draw_results(image, person_box, pose_results)

        return pose_results
        


import depth_pro
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

class DepthEstimator:
    def __init__(self, image_path):
        # Load model and preprocessing transform
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.eval()

        # Load and preprocess an image
        self.image, _, self.f_px = depth_pro.load_rgb(image_path)
        self.image_processed = self.transform(self.image)

        # Run inference
        with torch.no_grad():
            prediction = self.model.infer(self.image_processed, f_px=self.f_px)

        # Get results and ensure they're proper tensors
        self.depth = self._check_tensor(prediction["depth"], "Depth prediction")
        self.focallength_px = self._check_tensor(prediction["focallength_px"], "Focal length prediction")

        # Convert tensors to numpy arrays with shape verification
        self.depth_np = self._tensor_to_numpy(self.depth)
        self.focallength_px_np = self._tensor_to_numpy(self.focallength_px, fill_like=self.depth_np)

    def _check_tensor(self, tensor, name):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} is not a tensor")
        return tensor

    def _tensor_to_numpy(self, tensor, fill_like=None):
        np_array = tensor.squeeze().detach().cpu().numpy()
        if np_array.ndim < 2:
            if fill_like is not None:
                np_array = np.full_like(fill_like, np_array)
            else:
                np_array = np_array.reshape(self.image.shape[-2:])
        return np_array

    def visualize_results(self):
        plt.figure(figsize=(15, 5))

        # Plot depth map
        plt.subplot(121)
        depth_plot = plt.imshow(self.depth_np, cmap='plasma')
        plt.colorbar(depth_plot, label='Depth (meters)')
        plt.title('Depth Map\n(brighter = further from camera)')

        # Plot focal length
        plt.subplot(122)
        focal_plot = plt.imshow(self.focallength_px_np, cmap='viridis')
        plt.colorbar(focal_plot, label='Focal Length (pixels)')
        plt.title('Focal Length Map\n(camera internal parameter)')

        plt.tight_layout()
        plt.savefig('depth_visualization.png')
        print(f"Depth range: {self.depth_np.min():.2f}m to {self.depth_np.max():.2f}m")
        print(f"Focal length: {self.focallength_px_np.mean():.1f} pixels")

    def calculate_distance(self, point1, point2, focal_length=None):
        """Calculate real-world distance between two points"""
        try:
            if focal_length is None:
                focal_length = float(self.focallength_px_np.mean())
            # Get depth values at points
            z1, z2 = float(self.depth_np[point1[1], point1[0]]), float(self.depth_np[point2[1], point2[0]])

            # Calculate real-world coordinates using focal length
            x1, y1 = point1[0] * z1 / focal_length, point1[1] * z1 / focal_length
            x2, y2 = point2[0] * z2 / focal_length, point2[1] * z2 / focal_length

            # Calculate Euclidean distance
            distance = float(np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2))
            return distance
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return 0.0

    def visualize_distance(self, point1, point2, focal_length=None):
        """Visualize distance between two points"""
        if focal_length is None:
            focal_length = float(self.focallength_px_np.mean())

        # Calculate distance
        distance = self.calculate_distance(point1, point2, focal_length)

        # Visualization
        plt.figure(figsize=(15, 5))

        # Plot depth map with distance line
        plt.subplot(121)
        plt.imshow(self.depth_np, cmap='plasma')
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r-', linewidth=2)
        plt.plot(point1[0], point1[1], 'go', markersize=8)
        plt.plot(point2[0], point2[1], 'go', markersize=8)
        plt.colorbar(label='Depth (meters)')
        plt.title(f'Depth Map\nMeasured Distance: {distance:.2f}m')

        # Plot focal length
        plt.subplot(122)
        plt.imshow(self.focallength_px_np, cmap='viridis')
        plt.colorbar(label='Focal Length (pixels)')
        plt.title('Focal Length Map')

        plt.tight_layout()
        plt.savefig('depth_visualization.png')
        print(f"Depth range: {float(self.depth_np.min()):.2f}m to {float(self.depth_np.max()):.2f}m")
        print(f"Focal length: {focal_length:.1f} pixels")
        print(f"Distance between points: {distance:.2f}m")

        return distance

from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

class BackgroundRemover:
    def __init__(self, model_name='briaai/RMBG-2.0'):
        self.model = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        self.model.eval()
        self.image_size = (1024, 1024)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def remove_background(self, image_path, output_path="no_bg_image.png", mask_output_path="mask.png"):
        image = Image.open(image_path).convert("RGB")
        input_images = self.transform_image(image).unsqueeze(0)
        
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        
        mask.save(mask_output_path)
        image.save(output_path)
        
        print(f"Processed image saved to: {output_path}")
        print(f"Mask saved to: {mask_output_path}")
        
        return image, mask


# url = "IMG_8446.jpg"
# image = Image.open(url)
# pose_estimator = PoseEstimator()
# pose_results = pose_estimator.process_image(image)

# print(pose_results)
# # Example usage
# front_depth_estimator = DepthEstimator(url)
# front_depth_estimator.visualize_results()
# # Usage
# bg_remover = BackgroundRemover()
# processed_image, mask = bg_remover.remove_background(url)
# processed_image.show()

# def mean(p1, p2):
#     x1, y1 = p1
#     x2, y2 = p2
#     return (int((x1 + x2) / 2), int((y1 + y2) / 2))
# def alpha_mean(p1, p2, alplha):
#     x1, y1 = p1
#     x2, y2 = p2
#     x = int(x1 * alplha + x2 * (1 - alplha))
#     y = int(y1 * alplha + y2 * (1 - alplha))
#     return (x, y)

# from math import cos, sin, atan, radians, pi

# def cal_point(point, alpha, mask):
#     pred = point
#     point = (round(pred[0] + 10 * cos(alpha)), round(pred[1] + 10 * sin(alpha)))
#     if mask.getpixel(point) < 100:
#         return round(pred[0]), round(pred[1])
#     else:
#         return cal_point(point, alpha, mask)

# def cal_alpha(point1, point2):
#     x1, y1 = point1
#     x2, y2 = point2
#     return atan((y2 - y1) / (x2 - x1)) + pi /2, atan((y2 - y1) / (x2 - x1)) - pi /2

# def next_point(p1, p2, point, mask):
#     alpha1, alpha2 = cal_alpha(p1, p2)
#     point1 = cal_point(point, alpha1, mask)
#     point2 = cal_point(point, alpha2, mask)
#     return point1, point2

# def next_point_len(p1, p2, points, len):
#     alpha1, alpha2 = cal_alpha(p1, p2)
#     point1 = (round(points[0] + len * cos(alpha1)), round(points[1] + len * sin(alpha1)))
#     point2 = (round(points[0] + len * cos(alpha2)), round(points[1] + len * sin(alpha2)))
#     return point1, point2

# Measure = {}
# unit = front_depth_estimator.visualize_distance(pose_results['L_Shoulder'], pose_results['R_Shoulder'])
# print(unit)
# Measure['Shoulder'] = 2.5 / 2.0 * unit

# print("Distance visualized for shoulders:", Measure['Shoulder'])

# Measure['L_Arm'] = front_depth_estimator.visualize_distance(pose_results['L_Shoulder'], pose_results['L_Elbow']) + front_depth_estimator.visualize_distance(pose_results['L_Elbow'], pose_results['L_Wrist'])
# Measure['R_Arm'] = front_depth_estimator.visualize_distance(pose_results['R_Shoulder'], pose_results['R_Elbow']) + front_depth_estimator.visualize_distance(pose_results['R_Elbow'], pose_results['R_Wrist'])
# Measure['L_Leg'] = front_depth_estimator.visualize_distance(pose_results['L_Hip'], pose_results['L_Knee']) + front_depth_estimator.visualize_distance(pose_results['L_Knee'], pose_results['L_Ankle'])
# Measure['R_Leg'] = front_depth_estimator.visualize_distance(pose_results['R_Hip'], pose_results['R_Knee']) + front_depth_estimator.visualize_distance(pose_results['R_Knee'], pose_results['R_Ankle'])
# Measure['Torso'] = front_depth_estimator.visualize_distance(mean(pose_results['L_Shoulder'], pose_results['R_Shoulder']), mean(pose_results['L_Hip'], pose_results['R_Hip']))

# points_green = []
# points_green.append(mean(pose_results['L_Shoulder'], pose_results['L_Elbow']))
# points_green.append(mean(pose_results['R_Shoulder'], pose_results['R_Elbow']))
# points_green.append(mean(pose_results['L_Hip'], pose_results['L_Knee']))
# points_green.append(mean(pose_results['R_Hip'], pose_results['R_Knee']))
# points_green.append(mean(pose_results['L_Knee'], pose_results['L_Ankle']))
# points_green.append(mean(pose_results['R_Knee'], pose_results['R_Ankle']))
# down_point = mean(pose_results['L_Hip'], pose_results['R_Hip'])
# up_point = mean(pose_results['L_Shoulder'], pose_results['R_Shoulder'])
# points_green.append(alpha_mean(down_point, up_point, 0.0))
# points_green.append(alpha_mean(down_point, up_point, 0.2))
# points_green.append(alpha_mean(down_point, up_point, 0.4))
# points_green.append(alpha_mean(down_point, up_point, 0.6))
# points_green.append(alpha_mean(down_point, up_point, 0.8))
# points_green.append(alpha_mean(down_point, up_point, 1.0))

# for point in points_green:
#     x, y = point
#     draw = ImageDraw.Draw(image)
#     draw.ellipse((x-5, y-5, x+5, y+5), fill=(0,255,0))
# image.show()

# points_blue = {}
# points_blue['L_Upper_Arm']=(next_point(pose_results['L_Shoulder'], pose_results['L_Elbow'], mean(pose_results['L_Shoulder'], pose_results['L_Elbow']), mask))
# points_blue['R_Upper_Arm']=(next_point(pose_results['R_Shoulder'], pose_results['R_Elbow'], mean(pose_results['R_Shoulder'], pose_results['R_Elbow']), mask))
# points_blue['L_Thigh']=(next_point(pose_results['L_Hip'], pose_results['L_Knee'], mean(pose_results['L_Hip'], pose_results['L_Knee']), mask))
# points_blue['L_Calf']=(next_point(pose_results['L_Knee'], pose_results['L_Ankle'], mean(pose_results['L_Knee'], pose_results['L_Ankle']), mask))
# points_blue['L_Ankle']=(next_point(pose_results['L_Ankle'], pose_results['L_Knee'], pose_results['L_Ankle'], mask))
# points_blue['R_Thigh']=(next_point(pose_results['R_Hip'], pose_results['R_Knee'], mean(pose_results['R_Hip'], pose_results['R_Knee']), mask))
# points_blue['R_Calf']=(next_point(pose_results['R_Knee'], pose_results['R_Ankle'], mean(pose_results['R_Knee'], pose_results['R_Ankle']), mask))
# points_blue['R_Ankle']=(next_point(pose_results['R_Ankle'], pose_results['R_Knee'], pose_results['R_Ankle'], mask))
# points_blue['Neck']=(next_point(pose_results['Nose'], mean(pose_results['L_Shoulder'], pose_results['R_Shoulder']), alpha_mean(pose_results['Nose'], mean(pose_results['L_Shoulder'], pose_results['R_Shoulder']), 0.4), mask))

# for key, points in points_blue.items():
#     x1, y1 = points[0]
#     x2, y2 = points[1]
#     print(points)
#     draw.line((x1, y1, x2, y2), fill=(0, 0, 255), width=5)
#     draw.ellipse((x1-5, y1-5, x1+5, y1+5), fill=(0, 0, 255))
#     draw.ellipse((x2-5, y2-5, x2+5, y2+5), fill=(0, 0, 255))
#     Measure[key] = front_depth_estimator.visualize_distance(points[0], points[1]) * pi

# image.show()

# points_yellow = {}
# print(down_point, up_point)
# pixcel_unit = np.sqrt((pose_results['L_Shoulder'][0]-pose_results['R_Shoulder'][0])**2 + (pose_results['L_Shoulder'][1]-pose_results['R_Shoulder'][1])**2) / 2.0
# points_yellow['Cheast']=(next_point(down_point, up_point, alpha_mean(down_point, up_point, 0.3), mask))
# points_yellow['Waist']=(next_point(down_point, up_point, down_point, mask))
# points_yellow['Hip']=(next_point(down_point, up_point, alpha_mean(down_point, up_point, 0.8), mask))


# for key, points in points_yellow.items():
#     x1, y1 = points[0]
#     x2, y2 = points[1]
#     draw.line((x1, y1, x2, y2), fill=(255, 255, 0), width=5)
#     draw.ellipse((x1-5, y1-5, x1+5, y1+5), fill=(255, 255, 0))
#     draw.ellipse((x2-5, y2-5, x2+5, y2+5), fill=(255, 255, 0))
#     Measure[key] = front_depth_estimator.visualize_distance(points[0], points[1])
#     draw.line((points[0][0], points[0][1], points[1][0], points[1][1]), fill=(255, 255, 0), width=5)
# image.show()

# Measure['Arm'] = min(Measure['L_Arm'], Measure['R_Arm'])
# Measure['Leg'] = min(Measure['L_Leg'], Measure['R_Leg'])
# Measure['Upper_Arm'] = min(Measure['L_Upper_Arm'], Measure['R_Upper_Arm'])
# Measure['Ankle'] = min(Measure['L_Ankle'], Measure['R_Ankle'])
# Measure['Calf'] = min(Measure['L_Calf'], Measure['R_Calf'])
# Measure['Thigh'] = min(Measure['L_Thigh'], Measure['R_Thigh'])

# del Measure['L_Arm']
# del Measure['L_Leg']
# del Measure['L_Upper_Arm']
# del Measure['L_Ankle']
# del Measure['L_Calf']
# del Measure['L_Thigh']
# del Measure['R_Arm']
# del Measure['R_Leg']
# del Measure['R_Upper_Arm']
# del Measure['R_Ankle']
# del Measure['R_Calf']
# del Measure['R_Thigh']

# print(Measure)


import math
from math import cos, sin, atan, radians, pi

class HumanMeasure:
    def __init__(self, front_image_path, side_image_path):
        self.front_image = Image.open(front_image_path)
        self.pose_estimator = PoseEstimator()
        self.front_pose_results = self.pose_estimator.process_image(self.front_image)
        print(self.front_pose_results)
        # Example usage
        self.front_depth_estimator = DepthEstimator(front_image_path)
        # Usage
        self.bg_remover = BackgroundRemover()
        self.front_processed_image, self.front_mask = self.bg_remover.remove_background(front_image_path)

        self.side_image = Image.open(side_image_path)
        self.side_pose_results = self.pose_estimator.process_image(self.side_image)
        print(self.side_pose_results)
        # Example usage
        self.side_depth_estimator = DepthEstimator(side_image_path)
        # Usage
        self.side_processed_image, self.side_mask = self.bg_remover.remove_background(side_image_path)
        self.front_mask.show()
        self.side_mask.show()
        self.Measure = {}

    def mean(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    def alpha_mean(self, p1, p2, alplha):
        x1, y1 = p1
        x2, y2 = p2
        x = int(x1 * alplha + x2 * (1 - alplha))
        y = int(y1 * alplha + y2 * (1 - alplha))
        return (x, y)
    def cal_point(self, point, alpha, mask):
        pred = point
        point = (round(pred[0] + 10 * cos(alpha)), round(pred[1] + 10 * sin(alpha)))
        if mask.getpixel(point) < 100:
            return round(pred[0]), round(pred[1])
        else:
            return self.cal_point(point, alpha, mask)

    def cal_alpha(self, point1, point2):
        x1, y1 = point1

        x2, y2 = point2
        return atan((y2 - y1) / (x2 - x1)) + pi /2, atan((y2 - y1) / (x2 - x1)) - pi /2

    def next_point(self, p1, p2, point, mask):
        alpha1, alpha2 = self.cal_alpha(p1, p2)
        point1 = self.cal_point(point, alpha1, mask)
        point2 = self.cal_point(point, alpha2, mask)
        return point1, point2

    def next_point_len(self, p1, p2, points, len):
        alpha1, alpha2 = self.cal_alpha(p1, p2)
        point1 = (round(points[0] + len * cos(alpha1)), round(points[1] + len * sin(alpha1)))
        point2 = (round(points[0] + len * cos(alpha2)), round(points[1] + len * sin(alpha2)))
        return point1, point2   
    
    def ellipse_circumference(self,length, width):
        """Calculate the circumference of an ellipse using Ramanujan's approximation."""
        a = length / 2  # Semi-major axis
        b = width / 2   # Semi-minor axis
        
        # Ramanujan's approximation
        circumference = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
        
        return circumference
    def measure(self):
        unit = self.front_depth_estimator.calculate_distance(self.front_pose_results['L_Shoulder'], self.front_pose_results['R_Shoulder'])
        print(unit)
        self.Measure['Shoulder'] = 2.5 / 2.0 * unit

        print("Distance visualized for shoulders:", self.Measure['Shoulder'])

        self.Measure['L_Arm'] = self.front_depth_estimator.calculate_distance(self.front_pose_results['L_Shoulder'], self.front_pose_results['L_Elbow']) + self.front_depth_estimator.calculate_distance(self.front_pose_results['L_Elbow'], self.front_pose_results['L_Wrist'])
        self.Measure['R_Arm'] = self.front_depth_estimator.calculate_distance(self.front_pose_results['R_Shoulder'], self.front_pose_results['R_Elbow']) + self.front_depth_estimator.calculate_distance(self.front_pose_results['R_Elbow'], self.front_pose_results['R_Wrist'])
        self.Measure['L_Leg'] = self.front_depth_estimator.calculate_distance(self.front_pose_results['L_Hip'], self.front_pose_results['L_Knee']) + self.front_depth_estimator.calculate_distance(self.front_pose_results['L_Knee'], self.front_pose_results['L_Ankle'])
        self.Measure['R_Leg'] = self.front_depth_estimator.calculate_distance(self.front_pose_results['R_Hip'], self.front_pose_results['R_Knee']) + self.front_depth_estimator.calculate_distance(self.front_pose_results['R_Knee'], self.front_pose_results['R_Ankle'])
        self.Measure['Torso'] = self.front_depth_estimator.calculate_distance(self.mean(self.front_pose_results['L_Shoulder'], self.front_pose_results['R_Shoulder']), self.mean(self.front_pose_results['L_Hip'], self.front_pose_results['R_Hip']))

        front_down_point = self.mean(self.front_pose_results['L_Hip'], self.front_pose_results['R_Hip'])
        front_up_point = self.mean(self.front_pose_results['L_Shoulder'], self.front_pose_results['R_Shoulder'])
        side_down_point = self.mean(self.side_pose_results['L_Hip'], self.side_pose_results['R_Hip'])
        side_up_point = self.mean(self.side_pose_results['L_Shoulder'], self.side_pose_results['R_Shoulder'])
        
        points_green = []
        points_green.append(self.mean(self.front_pose_results['L_Shoulder'], self.front_pose_results['L_Elbow']))
        points_green.append(self.mean(self.front_pose_results['R_Shoulder'], self.front_pose_results['R_Elbow']))
        points_green.append(self.mean(self.front_pose_results['L_Hip'], self.front_pose_results['L_Knee']))
        points_green.append(self.mean(self.front_pose_results['R_Hip'], self.front_pose_results['R_Knee']))
        points_green.append(self.mean(self.front_pose_results['L_Knee'], self.front_pose_results['L_Ankle']))
        points_green.append(self.mean(self.front_pose_results['R_Knee'], self.front_pose_results['R_Ankle']))
        points_green.append(self.alpha_mean(front_down_point, front_up_point, 0.0))
        points_green.append(self.alpha_mean(front_down_point, front_up_point, 0.2))
        points_green.append(self.alpha_mean(front_down_point, front_up_point, 0.4))
        points_green.append(self.alpha_mean(front_down_point, front_up_point, 0.6))
        points_green.append(self.alpha_mean(front_down_point, front_up_point, 0.8))
        points_green.append(self.alpha_mean(front_down_point, front_up_point, 1.0))

        for point in points_green:
            x, y = point
            draw = ImageDraw.Draw(self.front_image)
            draw.ellipse((x-5, y-5, x+5, y+5), fill=(0,255,0))
        self.front_image.show()

        points_blue = {}
        points_blue['L_Upper_Arm']=(self.next_point(self.front_pose_results['L_Shoulder'], self.front_pose_results['L_Elbow'], self.mean(self.front_pose_results['L_Shoulder'], self.front_pose_results['L_Elbow']), self.front_mask))
        points_blue['R_Upper_Arm']=(self.next_point(self.front_pose_results['R_Shoulder'], self.front_pose_results['R_Elbow'], self.mean(self.front_pose_results['R_Shoulder'], self.front_pose_results['R_Elbow']), self.front_mask))
        points_blue['L_Thigh']=(self.next_point(self.front_pose_results['L_Hip'], self.front_pose_results['L_Knee'], self.mean(self.front_pose_results['L_Hip'], self.front_pose_results['L_Knee']), self.front_mask))
        points_blue['L_Calf']=(self.next_point(self.front_pose_results['L_Knee'], self.front_pose_results['L_Ankle'], self.mean(self.front_pose_results['L_Knee'], self.front_pose_results['L_Ankle']), self.front_mask))
        points_blue['L_Ankle']=(self.next_point(self.front_pose_results['L_Ankle'], self.front_pose_results['L_Knee'], self.front_pose_results['L_Ankle'], self.front_mask))
        points_blue['R_Thigh']=(self.next_point(self.front_pose_results['R_Hip'], self.front_pose_results['R_Knee'], self.mean(self.front_pose_results['R_Hip'], self.front_pose_results['R_Knee']), self.front_mask))
        points_blue['R_Calf']=(self.next_point(self.front_pose_results['R_Knee'], self.front_pose_results['R_Ankle'], self.mean(self.front_pose_results['R_Knee'], self.front_pose_results['R_Ankle']), self.front_mask))
        points_blue['R_Ankle']=(self.next_point(self.front_pose_results['R_Ankle'], self.front_pose_results['R_Knee'], self.front_pose_results['R_Ankle'], self.front_mask))
        points_blue['Neck']=(self.next_point(self.front_pose_results['Nose'], self.mean(self.front_pose_results['L_Shoulder'], self.front_pose_results['R_Shoulder']), self.alpha_mean(self.front_pose_results['Nose'], self.mean(self.front_pose_results['L_Shoulder'], self.front_pose_results['R_Shoulder']), 0.4), self.front_mask))

        for key, points in points_blue.items():
            x1, y1 = points[0]
            x2, y2 = points[1]
            print(points)
            draw.line((x1, y1, x2, y2), fill=(0, 0, 255), width=5)
            draw.ellipse((x1-5, y1-5, x1+5, y1+5), fill=(0, 0, 255))
            draw.ellipse((x2-5, y2-5, x2+5, y2+5), fill=(0, 0, 255))
            self.Measure[key] = self.front_depth_estimator.calculate_distance(points[0], points[1]) * pi

        self.front_image.show()

        points_yellow = {}
        print(front_down_point, front_up_point)
        pixcel_unit = np.sqrt((self.front_pose_results['L_Shoulder'][0]-self.front_pose_results['R_Shoulder'][0])**2 + (self.front_pose_results['L_Shoulder'][1]-self.front_pose_results['R_Shoulder'][1])**2) / 2.0
        points_yellow['Cheast']={
            'front': (self.next_point(front_down_point, front_up_point, self.alpha_mean(front_down_point, front_up_point, 0.3), self.front_mask)),
            'side': (self.next_point(side_down_point, side_up_point, self.alpha_mean(side_down_point, side_up_point, 0.3), self.side_mask)),
            }
        points_yellow['Waist']={
            'front': (self.next_point(front_down_point, front_up_point, front_down_point, self.front_mask)),
            'side': (self.next_point(side_down_point, side_up_point, side_down_point, self.side_mask))
            }
        points_yellow['Hip']={
            'front': (self.next_point(front_down_point, front_up_point, self.alpha_mean(front_down_point, front_up_point, 0.8), self.front_mask)),
            'side': (self.next_point(side_down_point, side_up_point, self.alpha_mean(side_down_point, side_up_point, 0.8), self.side_mask)),
            }


        for key, points in points_yellow.items():
            front_length = self.front_depth_estimator.calculate_distance(points['front'][0], points['front'][1])
            side_length = self.front_depth_estimator.calculate_distance(points['side'][0], points['side'][1])
            self.Measure[key] = self.ellipse_circumference(front_length, side_length)
        self.front_image.show()

        self.Measure['Arm'] = min(self.Measure['L_Arm'], self.Measure['R_Arm'])
        self.Measure['Leg'] = min(self.Measure['L_Leg'], self.Measure['R_Leg'])
        self.Measure['Upper_Arm'] = min(self.Measure['L_Upper_Arm'], self.Measure['R_Upper_Arm'])
        self.Measure['Ankle'] = min(self.Measure['L_Ankle'], self.Measure['R_Ankle'])
        self.Measure['Calf'] = min(self.Measure['L_Calf'], self.Measure['R_Calf'])
        self.Measure['Thigh'] = min(self.Measure['L_Thigh'], self.Measure['R_Thigh'])

        del self.Measure['L_Arm']
        del self.Measure['L_Leg']
        del self.Measure['L_Upper_Arm']
        del self.Measure['L_Ankle']
        del self.Measure['L_Calf']
        del self.Measure['L_Thigh']
        del self.Measure['R_Arm']
        del self.Measure['R_Leg']
        del self.Measure['R_Upper_Arm']
        del self.Measure['R_Ankle']
        del self.Measure['R_Calf']
        del self.Measure['R_Thigh']

        print(self.Measure)

        return self.Measure

if __name__ == '__main__':
    BodyMeasurer = HumanMeasure('IMG_8446.jpg', 'side.png')
    Measure = BodyMeasurer.measure()
    for key, value in Measure.items():
        print(key, value)