import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp
from collections import deque
import torch


midas = torch.hub.load("intel-isl/MiDas", "MiDaS_small")
midas.to("cpu")
midas.eval()

transforms = torch.hub.load("intel-isl/MiDas", "transforms")
transform = transforms.small_transform

# Initialize media pipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Detect only one hand (left hand)

# Load the PNG images for each fingertip (excluding thumb)
icons = {
    "index": cv2.imread('vitals.png', cv2.IMREAD_UNCHANGED),
    "middle": cv2.imread('location4.png', cv2.IMREAD_UNCHANGED),
    "ring": cv2.imread('oxygen2.png', cv2.IMREAD_UNCHANGED),
    "pinky": cv2.imread('thermal2.png', cv2.IMREAD_UNCHANGED),
}
# def compute_depth(frame):
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     imgbatch = transform(img).to("cpu")
#     with torch.no_grad():
#         prediction = midas(imgbatch)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=img.shape[:2],
#             mode="bicubic",
#             align_corners=False
#         ).squeeze()
#         depth_map = prediction.cpu().numpy()
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=255 / depth_map.max()), cv2.COLORMAP_HOT)
#     return depth_colormap
def compute_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to("cpu")
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
        depth_map = prediction.cpu().numpy()
        
        # Normalize the depth map
        normalized_depth = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        
        # Convert normalized depth map to uint8 and scale from 0 to 255
        depth_8bit = (normalized_depth * 255).astype(np.uint8)
        
        # Use Otsu's thresholding
        _, thresholded = cv2.threshold(depth_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a black image and draw green contours
        output = np.zeros_like(img)
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
        
    return output


screen_width = 1920
screen_height = 1080
cv2.namedWindow("Virtual Buttons", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Virtual Buttons", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# Resize the images (if needed)
icon_size = (100,100)
for key in icons:
    icons[key] = cv2.resize(icons[key], icon_size,interpolation=cv2.INTER_LINEAR)

cap = cv2.VideoCapture(0)

# Choose a font and size for the custom text
font = ImageFont.truetype("nasa21-font/Nasa21-l23X.ttf", 40)

# Parameters for icon clicking
offset_y = 30
click_threshold = 100
smooth_factor = 2
positions = {finger: deque(maxlen=smooth_factor) for finger in icons.keys()}
positions['thumb'] = deque(maxlen=smooth_factor)
button_states = {finger: 'OFF' for finger in icons.keys()}
pressing = {finger: False for finger in icons.keys()}

def is_clicked(fingertip, icon_center, threshold):
    distance = ((fingertip[0] - icon_center[0]) ** 2 + (fingertip[1] - icon_center[1]) ** 2) ** 0.5
    return distance < threshold

def average_position(deque_points):
    return (int(sum(x for x, _ in deque_points) / len(deque_points)), 
            int(sum(y for _, y in deque_points) / len(deque_points)))

def apply_overlay(frame):
   
    rows, cols, _ = frame.shape
    overlay_bg = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i in range(rows):
        overlay_bg[i, :, :] = (i/rows * 255, (1 - i/rows) * 50, (1 - i/rows) * 200)  # Blue gradient
    start_point = (50, 50)
    end_point = (frame.shape[1] - 50, frame.shape[0] - 50)
    thickness = 2
    color = (0, 0, 0)  # Black border
    cv2.rectangle(overlay_bg, start_point, end_point, color, thickness)
    alpha = 0.3
    cv2.addWeighted(overlay_bg, alpha, frame, 1 - alpha, 0, frame)
    
    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the OpenCV frame to a PIL Image and ensure it's in 'RGBA' mode
    pil_im = Image.fromarray(frame_rgb).convert('RGBA')
    
    # Create a transparent image to draw the custom text
    transparent_img = Image.new('RGBA',pil_im.size, (0, 0, 0, 0))
    
    # Initialize the drawing context on the transparent image
    draw = ImageDraw.Draw(transparent_img)
    
    # Choose a font and size
    font = ImageFont.truetype("nasa21-font/Nasa21-l23X.ttf", 40)
    
    # Draw text multiple times with small offsets for a "bold" effect
    offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, -2), (-2, 2), (2, -2), (2, 2)]
    for offset in offsets:
        draw.text((100 + offset[0], 100 + offset[1]), "O2 (%)", font=font, fill=(173,216,230, 500))
    
    # Draw the main text
    draw.text((100, 100), "O2 (%)", font=font,fill=(173,216,230, 500))
    
    # Composite the transparent image with the custom text onto the original frame
    pil_im = Image.alpha_composite(pil_im, transparent_img)
    
    # Convert the PIL image back to BGR for displaying with OpenCV
    frame[:] = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
screen_resolution = (1920, 1080)
while cap.isOpened():
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_FPS,43)
    #frame = resize_with_aspect_ratio(frame, width=screen_width)
    frame = cv2.flip(frame, 1)
     
    frame = cv2.resize(frame, screen_resolution)
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if results.multi_handedness[0].classification[0].label == 'Left':

                thumb_x, thumb_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]), \
                                   int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])
                positions['thumb'].append((thumb_x, thumb_y))
                thumb_x, thumb_y = average_position(positions['thumb'])

                finger_tips = {
                    "index": mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    "middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    "ring": mp_hands.HandLandmark.RING_FINGER_TIP,
                    "pinky": mp_hands.HandLandmark.PINKY_TIP,
                }

                for finger, landmark in finger_tips.items():
                    x, y = int(hand_landmarks.landmark[landmark].x * frame.shape[1]), \
                           int(hand_landmarks.landmark[landmark].y * frame.shape[0])
                    positions[finger].append((x, y))
                    x, y = average_position(positions[finger])  # Get average position for smoothing

                    icon = icons[finger]
                    
                    top_left_y = y - icon.shape[0]//2 - 0
                    top_left_x = x - icon.shape[1]//2

                    # Check if the thumb is close to the fingertip
                    if is_clicked((thumb_x, thumb_y), (x, y), click_threshold):
                        # If the thumb is near the fingertip and not already pressing it
                        if not pressing[finger]:
                            pressing[finger] = True
                            # Toggle the button state
                            button_states[finger] = 'ON' if button_states[finger] == 'OFF' else 'OFF'
                            print(f"{finger} button toggled to {button_states[finger]}!")
                    else:
                        pressing[finger] = False  # Reset pressing flag when thumb is away

                    # Overlay the icon
                    for i in range(icon.shape[0]):
                        for j in range(icon.shape[1]):
                            if 0 <= top_left_y + i < frame.shape[0] and 0 <= top_left_x + j < frame.shape[1]:
                                if icon[i, j][3] > 0:
                                    frame[top_left_y + i, top_left_x + j] = icon[i, j][:3]

    # If the index finger button is toggled ON, apply the overlay
    if button_states["index"] == "ON":
        apply_overlay(frame)
    if button_states["pinky"] == "ON":
        frame = compute_depth(frame)
        #alpha = 0.6  # Adjust this value to your preference
        #frame = cv2.addWeighted(frame, alpha, depth_overlay, 1 - alpha, 0)
    cv2.imshow('Virtual Buttons', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
