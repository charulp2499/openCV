import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import open3d as o3d
import cv2

# Load MiDaS model for depth estimation
model_type = "DPT_Large"  # MiDaS model type
model = torch.hub.load("intel-isl/MiDaS", model_type)
model.eval()

# Load transforms to resize and normalize the image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def generate_point_cloud(frame, depth_map):
    h, w = depth_map.shape
    focal_length = 1.2 * max(h, w)  # Approximate focal length
    cx, cy = w / 2, h / 2  # Camera center

    points = []
    colors = frame.reshape(-1, 3) / 255.0

    for v in range(h):
        for u in range(w):
            z = depth_map[v, u]
            if z > 0:  # Ignore zero depth values
                x = (u - cx) * z / focal_length
                y = (v - cy) * z / focal_length
                points.append([x, y, z])

    if points:
        points = np.array(points)

        # Create Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        return point_cloud
    return None

# Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Convert image to tensor
    input_tensor = transform(pil_image).unsqueeze(0)

    # Perform depth estimation
    with torch.no_grad():
        prediction = model(input_tensor)

    # Resize depth map to match input image size
    original_size = pil_image.size
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=(original_size[1], original_size[0]),  # Original image size
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Convert depth map to numpy array
    depth_map = prediction.cpu().numpy()

    # Normalize depth map
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)

    # Generate point cloud
    point_cloud = generate_point_cloud(frame, depth_map)

    if point_cloud:
        vis.clear_geometries()
        vis.add_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

    # Display the frame
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
