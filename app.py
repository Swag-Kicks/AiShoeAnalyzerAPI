from flask import Flask, request, jsonify
import requests
import torch
from io import BytesIO
import pandas
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cloudinary.uploader
import numpy as np
import pathlib
import threading
import time
import math
import re
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import pickle

import tensorflow as tf

#pip install opencv-python-headless
#pip install ultralytics

app = Flask(__name__)

categories=['1','2','3','4','5']

# Load architecture and weights from the pickle file
with open('ShoeAngle_model.pkl', 'rb') as f:
    model_json, model_weights = pickle.load(f)

# Reconstruct the model from the architecture
shoeAngle = tf.keras.models.model_from_json(model_json)

# Load the weights into the reconstructed model
shoeAngle.set_weights(model_weights)

# Compile the model (optional, depending on your use case)
shoeAngle.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


brand = torch.hub.load('ultralytics/yolov5', 'custom', path='brand.pt')
brand.eval()
component = torch.hub.load('ultralytics/yolov5', 'custom', path='component.pt')
component.eval()
damage = torch.hub.load('ultralytics/yolov5', 'custom', path='damage.pt')
damage.eval()
shoe=torch.hub.load('ultralytics/yolov5', 'custom', path='Shoe.pt')
shoe.eval()
brand_data = []
component_data=[]
damage_data=[]
uploadUrl=[]
output={}

@app.route('/', methods=['GET'])
def keep_alive():
  return 'Server is alive!'


def is_url_image(image_url):
  image_formats = ("image/png", "image/jpeg", "image/jpg")
  r = requests.head(image_url)
  if r.headers["content-type"] in image_formats:
    return True
  return False

def prepare_image(image):
    # Resize image to match model input shape
    img = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define constants for image dimensions
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224  # Assuming the model expects images of size 224x224
def compare_damages(image_data):
    updated_image_data = {}

    # Iterate over each image and its components
    for image_name, components in image_data.items():
        updated_components = {}

        # Iterate over each component in the image
        for component_type, damages in components.items():
            damage_map = {}

            # Iterate over damages in the component
            for damage_type, bbox in damages:
                if damage_type in damage_map:
                    # If damage type already exists, compare bounding boxes and keep the one with greater value
                    prev_bbox = damage_map[damage_type]
                    # Unpack previous and current bounding boxes
                    prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
                    x1, y1, x2, y2 = bbox
                    # Calculate areas of previous and current bounding boxes
                    prev_area = (prev_x2 - prev_x1) * (prev_y2 - prev_y1)
                    curr_area = (x2 - x1) * (y2 - y1)
                    # Keep the bounding box with greater area
                    if curr_area > prev_area:
                        damage_map[damage_type] = bbox
                else:
                    damage_map[damage_type] = bbox

            # Convert damage_map back to list format
            updated_damages = [(damage_type, bbox) for damage_type, bbox in damage_map.items()]

            # Update component with filtered damages
            updated_components[component_type] = updated_damages

        # Update image data with filtered components
        updated_image_data[image_name] = updated_components

    # Compare damages between specific combinations of images
    if 'back' in updated_image_data and 'left' in updated_image_data:
        updated_image_data['left'] = compare_components(updated_image_data['back'], updated_image_data['left'])
    if 'back' in updated_image_data and 'right' in updated_image_data:
        updated_image_data['right'] = compare_components(updated_image_data['back'], updated_image_data['right'])
    if 'front' in updated_image_data:
        if 'left' in updated_image_data and 'midsole' in updated_image_data['left']:
            updated_image_data['left'] = compare_components(updated_image_data['front'], updated_image_data['left'])
        if 'right' in updated_image_data and 'midsole' in updated_image_data['right']:
            updated_image_data['right'] = compare_components(updated_image_data['front'], updated_image_data['right'])

    return updated_image_data

def compare_components(image1, image2):
    # Compare damages between two components of different images
    for component_type, damages in image1.items():
        if component_type in image2:
            for damage_type, bbox in damages:
                for other_damage_type, other_bbox in image2[component_type]:
                    if damage_type == other_damage_type:
                        # If damage type is same, compare bounding boxes and keep the one with greater value
                        x1, y1, x2, y2 = bbox
                        other_x1, other_y1, other_x2, other_y2 = other_bbox
                        area = (x2 - x1) * (y2 - y1)
                        other_area = (other_x2 - other_x1) * (other_y2 - other_y1)
                        if area > other_area:
                            # Keep bbox from image1
                            image2[component_type].remove((other_damage_type, other_bbox))
                        else:
                            # Keep bbox from image2
                            image1[component_type].remove((damage_type, bbox))
    return image2

# Function to calculate overlap area between two bounding boxes
def calculate_overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate overlap area
    overlap_area = max(0, x2 - x1) * max(0, y2 - y1)

    return overlap_area


def separate_int_and_string(value):
    match = re.match(r"([a-zA-Z]+)([0-9]+)", value)
    if match:
        return match.group(1)
    else:
        return value

def DiscardDamage(image_data):
    updated_image_data = {}

    for image_name, components in image_data.items():
        updated_components = {}

        for component_type, damages in components.items():
            damage_map = {}

            for damage_type, bbox in damages:
                damage_type = separate_int_and_string(damage_type)
                if damage_type in damage_map:
                    # If damage type already exists, compare bounding boxes of the same damage type
                    prev_bbox = damage_map[damage_type]
                    # Unpack previous and current bounding boxes
                    prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
                    x1, y1, x2, y2 = bbox
                    # Calculate areas of previous and current bounding boxes
                    prev_area = (prev_x2 - prev_x1) * (prev_y2 - prev_y1)
                    curr_area = (x2 - x1) * (y2 - y1)
                    # Keep the bounding box with greater area
                    if curr_area > prev_area:
                        damage_map[damage_type] = bbox
                else:
                    damage_map[damage_type] = bbox

            # Convert damage_map back to list format
            updated_damages = [(damage_type, bbox) for damage_type, bbox in damage_map.items()]

            # Update component with filtered damages
            updated_components[component_type] = updated_damages

        # Update image data with filtered components
        updated_image_data[image_name] = updated_components

    return updated_image_data    
# Function to find damage associated with components
def find_damage_component(components, damages):
    result = {}
    used_damages = set()
    for component_name, component_bbox in components.items():
        max_overlap = 0
        associated_damage = None
        for damage_name, damage_bbox in damages.items():
            # Calculate overlap area
            overlap = calculate_overlap(component_bbox, damage_bbox)
            # Calculate percentage of overlap relative to the damage box area
            overlap_percent = overlap / ((damage_bbox[2] - damage_bbox[0]) * (damage_bbox[3] - damage_bbox[1]))
            # Check if the overlap percentage is significant
            if overlap_percent > 0.5:
                if overlap_percent > max_overlap:
                    max_overlap = overlap_percent
                    associated_damage = (damage_name, damage_bbox)  # Include bounding box info
        # If no damage is associated with the component, find the nearest damage and associate it
        if associated_damage is None:
            min_distance = float('inf')
            nearest_damage = None
            for damage_name, damage_bbox in damages.items():
                distance = np.linalg.norm(np.array(component_bbox[:2]) - np.array(damage_bbox[:2]))
                if distance < min_distance and damage_name not in used_damages:
                    min_distance = distance
                    nearest_damage = (damage_name, damage_bbox)  # Include bounding box info
            if nearest_damage is not None:
                associated_damage = nearest_damage
        if associated_damage is not None:
            if component_name not in result:
                result[component_name] = []
            result[component_name].append(associated_damage)
            used_damages.add(associated_damage[0])

    # For any unused damage, find the nearest component and associate it
    for damage_name, damage_bbox in damages.items():
        if damage_name not in used_damages:
            min_distance = float('inf')
            nearest_component = None
            for component_name, component_bbox in components.items():
                distance = np.linalg.norm(np.array(component_bbox[:2]) - np.array(damage_bbox[:2]))
                if distance < min_distance:
                    min_distance = distance
                    nearest_component = component_name
            if nearest_component is not None:
                if nearest_component not in result:
                    result[nearest_component] = []
                result[nearest_component].append((damage_name, damage_bbox))  # Include bounding box info
                used_damages.add(damage_name)

    return result

def remove_empty_components(image_data):
    # Remove components with empty damage lists
    for image_name, components in image_data.items():
        components_to_remove = [component_type for component_type, damages in components.items() if not damages]
        for component_type in components_to_remove:
            del image_data[image_name][component_type]

    # Remove images with no components
    images_to_remove = [image_name for image_name, components in image_data.items() if not components]
    for image_name in images_to_remove:
        del image_data[image_name]

    return image_data


def detect_thread(url,PAngle):
  angles=['back','bottom','right','front','left']
  component_data=[]
  damage_data=[]
  cap = cv2.VideoCapture(url)
  ret, frame = cap.read()
  img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  results = brand(img)
  drawn_image = img.copy()
  draw = ImageDraw.Draw(drawn_image)
  boxes = results.xyxy[0].cpu().numpy()
  for box in boxes:
    if len(box) >= 6:  # Ensure the box contains class index information
      x_min, y_min, x_max, y_max, confidence, class_index = box[:6]
      class_name = brand.names[int(class_index)]
      label = f"{class_name}: {confidence:.2f}"
      font_size = max(15, int((x_max - x_min) / 20))
      font = ImageFont.truetype("arial.ttf", font_size)
      
      brand_data.append([class_name, str(confidence)])
      print(label)
      # Draw rectangle and label
      draw.rectangle([x_min, y_min, x_max, y_max], outline='green', width=10)
      #draw.text((x_min, y_min - 10), label, fill='black', font=font)

    else:
      print(f"Ignoring box with unexpected structure: {box}")
  results2 = component(img)
  boxes2 = results2.xyxy[0].cpu().numpy()
  for box in boxes2:
    if len(box) >= 6:  # Ensure the box contains class index information
      x_min, y_min, x_max, y_max, confidence, class_index = box[:6]
      class_name = component.names[int(class_index)]
      label = f"{class_name}: {confidence:.2f}"
      font_size = max(15, int((x_max - x_min) / 20))
      font = ImageFont.truetype("arial.ttf", font_size)
      #print(label)
      component_data.append([class_name, str(confidence)])
      # Draw rectangle and label
      draw.rectangle([x_min, y_min, x_max, y_max],
                    outline='yellow',
                    width=10)
      #draw.text((x_min, y_min - 10), label, fill='black', font=font)

    else:
      print(f"Ignoring box with unexpected structure: {box}")
  results3 = damage(img)
  boxes3 = results3.xyxy[0].cpu().numpy()
  for box in boxes3:
    if len(box) >= 6:  # Ensure the box contains class index information
      x_min, y_min, x_max, y_max, confidence, class_index = box[:6]
      class_name = damage.names[int(class_index)]
      label = f"{class_name}: {confidence:.2f}"
      font_size = max(15, int((x_max - x_min) / 20))
      font = ImageFont.truetype("arial.ttf", font_size)
      #print(label)
      damage_data.append([class_name, str(confidence)])
      # Draw rectangle and label
      draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=10)
      #draw.text((x_min, y_min - 10), label, fill='black', font=font)

    else:
      print(f"Ignoring box with unexpected structure: {box}")
#  print(drawn_image.size)
  categorized_damage = {}
  categorized_components = {}
  damage_types = {0: 'tear', 1: 'scuff', 2: 'yellow', 3: 'spot'}
  component_types = {0: 'outersole', 1: 'midsole', 2: 'uppersole', 3: 'heel'}
  # Process damage data
  for entry in boxes3:
      label = int(entry[-1])
      damage_type = damage_types[label]
      coordinates = entry[:-2]
      if damage_type not in categorized_damage:
          categorized_damage[damage_type] = []
      categorized_damage[damage_type].append(coordinates)

  # Process component data
  for entry in boxes2:
      label = int(entry[-1])
      component_type = component_types[label]
      coordinates = entry[:-2]
      if component_type not in categorized_components:
          categorized_components[component_type] = []
      categorized_components[component_type].append(coordinates)
  
  output[angles[PAngle-1]]=categorized_components,categorized_damage

def predict_from_image(image):
    img_array = prepare_image(image)
    result_array = shoeAngle.predict(img_array, verbose=1)
    return result_array

@app.route('/check_angle',methods=['POST'])
def check_angle():
  uploadUrl=[]
  data = request.get_json()
  url = data.get('image_url')
  angle=data.get('angle')
  if(is_url_image(url)==False):
    print("Invalid Image url")
    return jsonify({'error: ': 'Invalid image URL'}), 300
  cap = cv2.VideoCapture(url)
  ret, frame = cap.read()
  img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  shoeResult=shoe(img)
  if(len(shoeResult)==0):
    return jsonify({'error: ':'Image is not Shoe'}),302
  resultArray = predict_from_image(img)
  predicted_class=categories[np.argmax(resultArray,axis=1)[0]]
  print(predicted_class)
  if(predicted_class == angle):
    return jsonify({'result': 'True'}),200
  else:
    return jsonify({'result': 'False','predicted_class':predicted_class}),201

@app.route('/get_data', methods=['POST'])
def get_data():
  print("request received")
  image_url=""
  shoeangles=[]
  anglecount=0
  try:
    data = request.get_json()
    urls = data.get('image_url')
    for url in urls:#check for each shoe
      if (is_url_image(url) == False):
        print(False)
        return jsonify({'error: ': 'Invalid image URL'}), 400
      cap = cv2.VideoCapture(url)
      ret, frame = cap.read()
      img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      shoeResult=shoe(img)
      if(len(shoeResult)==0):
        return jsonify({'error: ':'All links are not Shoe'}),402

    #check for each angle
    for url in urls:
      if (is_url_image(url) == False):
        print(False)
        return jsonify({'error: ': 'Invalid image URL'}), 400

      cap = cv2.VideoCapture(url)
      ret, frame = cap.read()
      img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      resultArray = predict_from_image(img)
      predicted_class=categories[np.argmax(resultArray,axis=1)[0]]
      print(predicted_class)
      if(int(predicted_class) not in shoeangles):
        anglecount=anglecount+1
      shoeangles.append(int(predicted_class))
    print(anglecount)
    if(not anglecount>=4):
      return jsonify({'error: ':'Not all angles Found'}),401
    threads=[]
    i=0
    target_angles = [1, 2, 3, 4, 5]


    # Check if all target angles are present in the list

    for angle in target_angles:

        if angle not in shoeangles:

            # If a target angle is missing, replace it with a duplicate of an existing angle

            for i, existing_angle in enumerate(shoeangles):

                if existing_angle == angle - 1 or existing_angle == angle + 1:

                    shoeangles[i] = angle

                    break

    
    print(shoeangles)
    for url in urls:
      #multithreading here
      if (is_url_image(url) == False):
        print(False)
        return jsonify({'error: ': 'Invalid image URL'}), 400
      print(int(shoeangles[i]))
      thread=threading.Thread(target=detect_thread,args=(url,int(shoeangles[i]),))
      i=i+1
      threads.append(thread)
      thread.start()
    for thread in threads:
      thread.join()
    # Process the image data
    result = {}
    image_data={}
    data=output
    for key, (part_data, defect_data) in data.items():
      part_dict = {k: [int(x) for x in v[0]] for k, v in part_data.items()}
      defect_dict = {k: [int(x) for x in v[0]] for k, v in defect_data.items()}
      image_data[key] = (part_dict, defect_dict)
    data=image_data

    for image_name, (components, damages) in data.items():
        result[image_name] = find_damage_component(components, damages)

    # Print the result on the console
    print("Result before discard:: ",result)
    resultafterdiscard = DiscardDamage(result)
    
    print("Result after discard (same damage on same component):: ",resultafterdiscard)

    image_data_discard = compare_damages(resultafterdiscard )
    image_data_discard = remove_empty_components(image_data_discard)
    print("Result after discard (same damage on different image of same component): ",image_data_discard)
    image_data = image_data_discard

    # Calculate the condition of the shoes
    updated_image_data = calculate_condition(image_data)

    # Calculate the overall condition score
    overall_condition = calculate_overall_condition(updated_image_data)

    # Round down the overall condition to the nearest multiple of 10
    overall_condition = math.floor(overall_condition / (len(updated_image_data) * 10) * 10)


    # Construct the API response
    api_response = {'brand': brand_data,'overall_condition':overall_condition,'url':uploadUrl,'output':data}
    return jsonify(api_response), 200

  except Exception as e:
    return jsonify({'error': str(e)}), 500

def calculate_condition(image_data):
    # Define area thresholds for small, medium, and large areas
    small_area_threshold = 5000
    medium_area_threshold = 10000
    large_area_threshold = 20000

    # Define point deductions for different damage types and areas
    point_deductions = {
        'default': {
            'small': 0.25,
            'medium': 0.5,
            'large': 1
        },
        'tear': {
            'small': 0.5,
            'medium': 1.5,
            'large': 3
        }
    }

    # Iterate over each image and its components
    for image_name, components in image_data.items():
        total_points = 10  # Initial total points for the image

        # Iterate over each component in the image
        for component_type, damages in components.items():
            # Iterate over damages in the component
            for damage_type, bbox in damages:
                # Calculate the area of the bounding box
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)

                # Determine the damage type and area category
                if damage_type in point_deductions:
                    area_category = 'large' if area > large_area_threshold else ('medium' if area > medium_area_threshold else 'small')
                    point_deduction = point_deductions[damage_type][area_category]
                else:
                    area_category = 'large' if area > large_area_threshold else ('medium' if area > medium_area_threshold else 'small')
                    point_deduction = point_deductions['default'][area_category]

                # Deduct points based on damage type and area category
                total_points -= point_deduction

                # Print the deduction details for debugging
                print(f"Image: {image_name}, Component: {component_type}, Damage: {damage_type}, Area: {area}, Area Category: {area_category}, Deduction: {point_deduction}")

        # Update the image data with the total points (without rounding)
        image_data[image_name]['condition'] = total_points

    return image_data

def calculate_overall_condition(image_data):
    total_points = 0

    # Iterate over each image and its components
    for image_name, components in image_data.items():
        total_points += components.get('condition', 10)  # Get the condition score, default to 10 if not present

    return total_points

# API endpoint to calculate overall condition
@app.route('/calculate_condition', methods=['POST'])
def calculate_overall_condition_api():
    # Get the image data from the request
    image_data = request.json

    # Calculate the condition of the shoes
    updated_image_data = calculate_condition(image_data)

    # Calculate the overall condition score
    overall_condition = calculate_overall_condition(updated_image_data)

    # Round down the overall condition to the nearest multiple of 10
    overall_condition = math.floor(overall_condition / (len(updated_image_data) * 10) * 10)

    # Prepare response data including debugging information
    response_data = {
       
        'overall_condition': overall_condition
    }

    return jsonify(response_data)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
