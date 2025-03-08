# import random
# import time
# from collections import defaultdict
# from datetime import datetime

# import cv2
# import imutils
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# # Import all the necessary modules and functions from the original script
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from flask import Flask, Response, jsonify, render_template
# from flask_cors import CORS
# from PIL import Image
# from pymongo import MongoClient


# # Copy the entire Face_Emotion_CNN class from the original script
# class Face_Emotion_CNN(nn.Module):
#     def __init__(self):
#         super(Face_Emotion_CNN, self).__init__()
#         self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
#         self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
#         self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
#         self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
#         self.cnn5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
#         self.cnn6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
#         self.cnn7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
#         self.relu = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(2, 1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.cnn1_bn = nn.BatchNorm2d(8)
#         self.cnn2_bn = nn.BatchNorm2d(16)
#         self.cnn3_bn = nn.BatchNorm2d(32)
#         self.cnn4_bn = nn.BatchNorm2d(64)
#         self.cnn5_bn = nn.BatchNorm2d(128)
#         self.cnn6_bn = nn.BatchNorm2d(256)
#         self.cnn7_bn = nn.BatchNorm2d(256)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 7)
#         self.dropout = nn.Dropout(0.3)
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x):
#         x = self.relu(self.pool1(self.cnn1_bn(self.cnn1(x))))
#         x = self.relu(self.pool1(self.cnn2_bn(self.dropout(self.cnn2(x)))))
#         x = self.relu(self.pool1(self.cnn3_bn(self.cnn3(x))))
#         x = self.relu(self.pool1(self.cnn4_bn(self.dropout(self.cnn4(x)))))
#         x = self.relu(self.pool2(self.cnn5_bn(self.cnn5(x))))
#         x = self.relu(self.pool2(self.cnn6_bn(self.dropout(self.cnn6(x)))))
#         x = self.relu(self.pool2(self.cnn7_bn(self.dropout(self.cnn7(x)))))

#         x = x.view(x.size(0), -1)

#         x = self.relu(self.dropout(self.fc1(x)))
#         x = self.relu(self.dropout(self.fc2(x)))
#         x = self.log_softmax(self.fc3(x))
#         return x

# # Replicate the helper functions from the original script
# def cos_sim(a, b):
#     dot_product = np.dot(a, b)
#     norm_a = np.linalg.norm(a)
#     norm_b = np.linalg.norm(b)
#     return dot_product / (norm_a * norm_b)

# def cos(a, b):
#     minx = -1 
#     maxx = 1
#     return (cos_sim(a, b) - minx) / (maxx - minx)

# # Load data and setup models
# # Note: You'll need to update these paths to match your system
# embeddings_df = pd.read_excel('app\\data\\face_embeddings.xlsx', index_col=0)
# names = embeddings_df.index.tolist()
# embeddings = [torch.tensor(embedding, dtype=torch.float32) for embedding in embeddings_df.values]

# # Device configuration
# device = torch.device('cpu')
# resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# mtcnn = MTCNN(
#     image_size=160, margin=0, min_face_size=20,
#     thresholds=[0.6, 0.7, 0.7], factor=0.709,
#     device=device, keep_all=True
# )

# # Load emotion detection model
# def load_trained_model(model_path):
#     model = Face_Emotion_CNN()
#     model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
#     return model

# emotion_model = load_trained_model('app\\models\\FER_trained_model.pt')
# emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
#                 4: 'anger', 5: 'disgust', 6: 'fear'}
# val_transform = transforms.Compose([
#     transforms.ToTensor()])

# # MongoDB setup
# mongodb_connection_string = "mongodb+srv://alemosense:alemosense@cluster0.qyn13ye.mongodb.net/?retryWrites=true&w=majority"
# database_name = "AIEMOSENSE"
# collection_name = "Face_Emotion_Sense"
# client = MongoClient(mongodb_connection_string)
# db = client[database_name]
# collection = db[collection_name]

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Global variables for tracking
# recognition_data = defaultdict(lambda: defaultdict(int))
# emotion_counts = defaultdict(lambda: defaultdict(int))
# most_recognized_names = {}
# most_recognized_emotions = {}
# matched_data = {}
# minute_start_time = time.time()

# # Subjects for random selection
# subject_list = ["AAI", "CC", "SNA"]

# def get_current_datetime():
#     now = datetime.now()
#     return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")

# def send_data_to_mongodb(matched_data, unmatched_names, subject):
#     date, time = get_current_datetime()

#     # Insert matched data
#     for name, emotion in matched_data.items():
#         record = {
#             "name": name,
#             "attendance": "yes",
#             "emotion": emotion,
#             "subject": subject,
#             "date": date,
#             "time": time
#         }
#         collection.insert_one(record)

#     # Insert unmatched names
#     for name in unmatched_names:
#         record = {
#             "name": name,
#             "attendance": "no",
#             "emotion": "",
#             "subject": subject,
#             "date": date,
#             "time": time
#         }
#         collection.insert_one(record)

# def generate_frames():
#     global recognition_data, emotion_counts, most_recognized_names, most_recognized_emotions, matched_data, minute_start_time
    
#     vs = cv2.VideoCapture(0)
#     classifier = cv2.CascadeClassifier('app\\models\\haarcascade_frontalface_default.xml')
#     subject = random.choice(subject_list)
    
#     try:
#         while True:
#             ret, im = vs.read()
#             if not ret:
#                 break

#             im = cv2.flip(im, 1)
#             frame = imutils.resize(im, width=400)
#             faces = classifier.detectMultiScale(frame)

#             for (x, y, w, h) in faces:
#                 face_img = frame[y:y + h, x:x + w]
#                 face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
#                 img_cropped = mtcnn(face_pil)

#                 if img_cropped is not None:
#                     if img_cropped.ndimension() == 4:  # Batch of faces
#                         for cropped_face in img_cropped:
#                             img_embedding = resnet(cropped_face.unsqueeze(0).to(device)).detach().cpu().numpy().flatten()
#                             for i, known_embedding in enumerate(embeddings):
#                                 dist = cos(known_embedding.numpy(), img_embedding)
#                                 if dist > 0.85:
#                                     recognized_name = names[i]
#                                     recognition_data[(x, y, w, h)][recognized_name] += 1
#                                     cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#                     else:  # Single face detected
#                         img_embedding = resnet(img_cropped.unsqueeze(0).to(device)).detach().cpu().numpy().flatten()
#                         for i, known_embedding in enumerate(embeddings):
#                             dist = cos(known_embedding.numpy(), img_embedding)
#                             if dist > 0.85:
#                                 recognized_name = names[i]
#                                 recognition_data[(x, y, w, h)][recognized_name] += 1
#                                 cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#                 # Emotion detection
#                 gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
#                 gray_pil = Image.fromarray(gray).resize((48, 48))
#                 X = val_transform(gray_pil).unsqueeze(0)
#                 with torch.no_grad():
#                     emotion_model.eval()
#                     log_ps = emotion_model(X)
#                     ps = torch.exp(log_ps)
#                     top_class = ps.argmax(dim=1).item()
#                     emotion = emotion_dict[top_class]
#                     emotion_counts[(x, y, w, h)][emotion] += 1
#                     cv2.putText(frame, emotion, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             # Check if 1 minute has passed  
#             if time.time() - minute_start_time >= 60:
#                 for box, counts in recognition_data.items():
#                     most_recognized_names[box] = max(counts, key=counts.get)
#                 for box, counts in emotion_counts.items():
#                     most_recognized_emotions[box] = max(counts, key=counts.get)

#                 # Match names and emotions by coordinates
#                 for name_box, name in most_recognized_names.items():
#                     if name_box in most_recognized_emotions:
#                         matched_data[name] = most_recognized_emotions[name_box]
#                 print(matched_data)

#                 # Send data to MongoDB
#                 recognized_names = set(matched_data.keys())
#                 unmatched_names = set(names) - recognized_names
#                 send_data_to_mongodb(matched_data, unmatched_names, subject)

#                 # Reset tracking variables
#                 recognition_data.clear()
#                 emotion_counts.clear()
#                 most_recognized_names.clear()
#                 most_recognized_emotions.clear()
#                 matched_data.clear()
#                 minute_start_time = time.time()

#             # Convert frame to JPEG
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
            
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         vs.release()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), 
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/records')
# def get_records():
#     try:
#         # Retrieve records from MongoDB
#         records = list(collection.find())
#         # Convert ObjectId to string for JSON serialization
#         for record in records:
#             record['_id'] = str(record['_id'])
#         return jsonify(records)
#     except Exception as e:
#         print(f"Error in /records route: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run()


import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from flask import Flask, render_template, request

app = Flask(__name__)

import random
import time
from collections import defaultdict
from datetime import datetime

import cv2
import imutils
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Import all the necessary modules and functions from the original script
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask import Flask, Response, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from pymongo import MongoClient


class Face_Emotion_CNN(nn.Module):
    def __init__(self):
        super(Face_Emotion_CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.cnn5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.cnn6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.cnn7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.cnn1_bn = nn.BatchNorm2d(8)
        self.cnn2_bn = nn.BatchNorm2d(16)
        self.cnn3_bn = nn.BatchNorm2d(32)
        self.cnn4_bn = nn.BatchNorm2d(64)
        self.cnn5_bn = nn.BatchNorm2d(128)
        self.cnn6_bn = nn.BatchNorm2d(256)
        self.cnn7_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)
        self.dropout = nn.Dropout(0.3)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.pool1(self.cnn1_bn(self.cnn1(x))))
        x = self.relu(self.pool1(self.cnn2_bn(self.dropout(self.cnn2(x)))))
        x = self.relu(self.pool1(self.cnn3_bn(self.cnn3(x))))
        x = self.relu(self.pool1(self.cnn4_bn(self.dropout(self.cnn4(x)))))
        x = self.relu(self.pool2(self.cnn5_bn(self.cnn5(x))))
        x = self.relu(self.pool2(self.cnn6_bn(self.dropout(self.cnn6(x)))))
        x = self.relu(self.pool2(self.cnn7_bn(self.dropout(self.cnn7(x)))))

        x = x.view(x.size(0), -1)

        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.log_softmax(self.fc3(x))
        return x


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def cos(a, b):
    minx = -1 
    maxx = 1
    return (cos_sim(a, b) - minx) / (maxx - minx)

# Load data and setup models
embeddings_df = pd.read_excel('F:/Demoo/face_recognition/app/data/face_embeddings.xlsx', index_col=0)
names = embeddings_df.index.tolist()
embeddings = [torch.tensor(embedding, dtype=torch.float32) for embedding in embeddings_df.values]

# Device configuration
device = torch.device('cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    device=device, keep_all=True
)

# Load emotion detection model
def load_trained_model(model_path):
    model = Face_Emotion_CNN()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    return model

emotion_model = load_trained_model('F:/Demoo/face_recognition/app/models/FER_trained_model.pt')
emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
                4: 'anger', 5: 'disgust', 6: 'fear'}
val_transform = transforms.Compose([
    transforms.ToTensor()])

# MongoDB setup
# mongodb_connection_string = "mongodb+srv://alemosense:alemosense@cluster0.qyn13ye.mongodb.net/?retryWrites=true&w=majority"
# database_name = "AIEMOSENSE"
# collection_name = "Face_Emotion_Sense"
# client = MongoClient(mongodb_connection_string)
# db = client[database_name]
# collection = db[collection_name]


mongodb_connection_string = "mongodb+srv://alemosense:alemosense@cluster0.qyn13ye.mongodb.net/?retryWrites=true&w=majority"
database_name = "AIEMOSENSE"
collection_name = "FaceEmotionSense"
client = MongoClient(mongodb_connection_string)
db = client[database_name]
collection = db[collection_name]



# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for tracking
recognition_data = defaultdict(lambda: defaultdict(int))
emotion_counts = defaultdict(lambda: defaultdict(int))
most_recognized_names = {}
most_recognized_emotions = {}
matched_data = {}
minute_start_time = time.time()

# Subjects for random selection
subject_list = ["AAI", "CC", "SNA"]

def get_current_datetime():
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")

def send_data_to_mongodb(matched_data, unmatched_names, subject):
    date, time = get_current_datetime()

    # Insert matched data
    for name, emotion in matched_data.items():
        record = {
            "name": name,
            "attendance": "Present",
            "emotion": emotion,
            "subject": subject,
            "date": date,
            "time": time
        }
        collection.insert_one(record)

    # Insert unmatched names
    for name in unmatched_names:
        record = {
            "name": name,
            "attendance": "Absent",
            "emotion": "",
            "subject": subject,
            "date": date,
            "time": time
        }
        collection.insert_one(record)

def generate_frames():
    global recognition_data, emotion_counts, most_recognized_names, most_recognized_emotions, matched_data, minute_start_time
    
    vs = cv2.VideoCapture(0)
    classifier = cv2.CascadeClassifier('F:/Demoo/face_recognition/app/models/haarcascade_frontalface_default.xml')
    subject = random.choice(subject_list)
    
    try:
        while True:
            ret, im = vs.read()
            if not ret:
                break

            im = cv2.flip(im, 1)
            frame = imutils.resize(im, width=400)
            faces = classifier.detectMultiScale(frame)

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                img_cropped = mtcnn(face_pil)

                if img_cropped is not None:
                    if img_cropped.ndimension() == 4:  # Batch of faces
                        for cropped_face in img_cropped:
                            img_embedding = resnet(cropped_face.unsqueeze(0).to(device)).detach().cpu().numpy().flatten()
                            for i, known_embedding in enumerate(embeddings):
                                dist = cos(known_embedding.numpy(), img_embedding)
                                if dist > 0.85:
                                    recognized_name = names[i]
                                    recognition_data[(x, y, w, h)][recognized_name] += 1
                                    cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    else:  # Single face detected
                        img_embedding = resnet(img_cropped.unsqueeze(0).to(device)).detach().cpu().numpy().flatten()
                        for i, known_embedding in enumerate(embeddings):
                            dist = cos(known_embedding.numpy(), img_embedding)
                            if dist > 0.85:
                                recognized_name = names[i]
                                recognition_data[(x, y, w, h)][recognized_name] += 1
                                cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Emotion detection
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                gray_pil = Image.fromarray(gray).resize((48, 48))
                X = val_transform(gray_pil).unsqueeze(0)
                with torch.no_grad():
                    emotion_model.eval()
                    log_ps = emotion_model(X)
                    ps = torch.exp(log_ps)
                    top_class = ps.argmax(dim=1).item()
                    emotion = emotion_dict[top_class]
                    emotion_counts[(x, y, w, h)][emotion] += 1
                    cv2.putText(frame, emotion, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Check if 1 minute has passed  
            if time.time() - minute_start_time >= 60:
                for box, counts in recognition_data.items():
                    most_recognized_names[box] = max(counts, key=counts.get)
                for box, counts in emotion_counts.items():
                    most_recognized_emotions[box] = max(counts, key=counts.get)

                # Match names and emotions by coordinates
                for name_box, name in most_recognized_names.items():
                    if name_box in most_recognized_emotions:
                        matched_data[name] = most_recognized_emotions[name_box]
                print(matched_data)

                # Send data to MongoDB
                recognized_names = set(matched_data.keys())
                unmatched_names = set(names) - recognized_names
                send_data_to_mongodb(matched_data, unmatched_names, subject)

                # Reset tracking variables
                recognition_data.clear()
                emotion_counts.clear()
                most_recognized_names.clear()
                most_recognized_emotions.clear()
                matched_data.clear()
                minute_start_time = time.time()

            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        vs.release()

# Load the CSV file
def load_data():
    try:
        # Ensure the CSV file exists in the correct location
        csv_path = os.path.join(os.path.dirname(__file__), 'F:/Demoo/face_recognition/app/data/fetched_data.csv')
        df = pd.read_csv(csv_path)
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        
        # Extract month and week information
        df['month'] = df['date'].dt.to_period('M').astype(str)
        df['week'] = df['date'].dt.to_period('W').astype(str)
        
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None



# # Load the Datebase
def load_data():
    try:
        cursor = collection.find({})
        records = list(cursor)
        
        df = pd.DataFrame(records)
        
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        df['date'] = pd.to_datetime(df['date'])
        

        df['month'] = df['date'].dt.to_period('M').astype(str)
        df['week'] = df['date'].dt.to_period('W').astype(str)
        
        return df
    except Exception as e:
        print(f"Error loading data from MongoDB: {e}")
        return None



# Attendance Analysis Routes
@app.route('/attendance/overall')
def overall_attendance():
    df = load_data()
    if df is None:
        return "Error loading data"
    
    # Calculate attendance percentages
    attendance_data = df['attendance'].value_counts(normalize=True) * 100
    
    # Define colors for "Present" and "Absent"
    color_map = {
        'Present': 'blue',
        'Absent': 'red'
    }
    
    # Create interactive pie chart
    fig = go.Figure(data=[go.Pie(
        labels=[f"{label} ({value:.1f}%)" for label, value in zip(attendance_data.index, attendance_data.values)],
        values=attendance_data.values,
        hovertemplate='<b>%{label}</b><br>Percentage: %{value:.1f}%<extra></extra>',
        marker=dict(colors=[color_map.get(label, 'gray') for label in attendance_data.index]) 
    )])
    
    fig.update_layout(
        height=600,
        width=800
    )
    
    # Convert to HTML
    plot_html = pio.to_html(fig, full_html=False)
    return render_template('plot.html', plot_html=plot_html, title='Overall Attendance Analysis')


@app.route('/attendance/monthly')
def monthly_attendance():
    df = load_data()
    if df is None:
        return "Error loading data"
    
    # Group and calculate percentages
    monthly_attendance = df.groupby(['month', 'attendance']).size().unstack(fill_value=0)
    monthly_attendance_percent = monthly_attendance.apply(lambda x: x / x.sum() * 100, axis=1)
    
    # Define colors for each attendance type
    color_map = {'Present': 'blue', 'Absent': 'red'}
    
    # Prepare data for Plotly
    data = []
    for attendance_type in monthly_attendance_percent.columns:
        data.append(go.Bar(
            name=attendance_type,
            x=monthly_attendance_percent.index,
            y=monthly_attendance_percent[attendance_type],
            hovertemplate='<b>%{x} - %{fullData.name}</b><br>Percentage: %{y:.1f}%<extra></extra>',
            text=[f'{val:.1f}%' if val > 0 else '' for val in monthly_attendance_percent[attendance_type]],
            textposition='inside',
            textfont=dict(color='white', size=10),
            marker=dict(color=color_map.get(attendance_type, 'gray'))
        ))
    
    # Create figure
    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='stack',
        height=600,
        width=1000,
        xaxis_title='Month',
        yaxis_title='Percentage',
        bargap=0.1,  # Add some gap between bar groups
        bargroupgap=0.1  # Add some gap between groups
    )
    
    # Customize text and hover information
    fig.update_traces(
        texttemplate='%{text}',
        textposition='inside'
    )
    
    # Convert to HTML
    plot_html = pio.to_html(fig, full_html=False)
    return render_template('plot.html', plot_html=plot_html, title='Monthly Attendance Analysis')


@app.route('/attendance/weekly')
def weekly_attendance():
    df = load_data()
    if df is None:
        return "Error loading data"
    
    # Group and calculate percentages
    weekly_attendance = df.groupby(['week', 'attendance']).size().unstack(fill_value=0)
    weekly_attendance_percent = weekly_attendance.apply(lambda x: x / x.sum() * 100, axis=1)
    
    # Define colors for each attendance type
    color_map = {'Present': 'blue', 'Absent': 'red'}
    
    # Prepare data for Plotly
    data = []
    for attendance_type in weekly_attendance_percent.columns:
        data.append(go.Bar(
            name=attendance_type,
            x=weekly_attendance_percent.index,
            y=weekly_attendance_percent[attendance_type],
            hovertemplate='<b>%{x} - %{fullData.name}</b><br>Percentage: %{y:.1f}%<extra></extra>',
            text=[f'{val:.1f}%' if val > 0 else '' for val in weekly_attendance_percent[attendance_type]],
            textposition='inside',
            textfont=dict(color='white', size=10),
            marker=dict(color=color_map.get(attendance_type, 'gray'))  # Use gray if not mapped
        ))
    
    # Create figure
    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='stack',
        height=600,
        width=1000,
        xaxis_title='Week',
        yaxis_title='Percentage',
        bargap=0.1,  # Add some gap between bar groups
        bargroupgap=0.1  # Add some gap between groups
    )
    
    # Customize text and hover information
    fig.update_traces(
        texttemplate='%{text}',
        textposition='inside'
    )
    
    # Convert to HTML
    plot_html = pio.to_html(fig, full_html=False)
    return render_template('plot.html', plot_html=plot_html, title='Weekly Attendance Analysis')

# Emotion Analysis Routes
@app.route('/emotion/overall')
def overall_emotion():
    df = load_data()
    if df is None:
        return "Error loading data"
    
    # Filter out blank or missing emotion values
    df = df[df['emotion'].notna() & (df['emotion'] != '')]
    
    # Normalize emotion names
    df['emotion'] = df['emotion'].str.strip().str.lower()
    
    # Debug: Print unique emotion values
    print(df['emotion'].unique())
    
    # Calculate emotion percentages
    emotion_data = df['emotion'].value_counts(normalize=True) * 100
    
    # Define color mapping for emotions
    color_map = {
        'neutral': '#FF851B',  # Red
        'happiness': '#2ECC40',    # Green
        'sadness': '#0074D9',      # Blue
        'anger': '#EE1010FF',    # Orange
        'surprise': '#B10DC9', # Purple
    }
    
    # Create interactive pie chart
    fig = go.Figure(data=[go.Pie(
        labels=[f"{label} ({value:.1f}%)" for label, value in zip(emotion_data.index, emotion_data.values)],
        values=emotion_data.values,
        hovertemplate='<b>%{label}</b><br>Percentage: %{value:.1f}%<extra></extra>',
        marker=dict(colors=[color_map.get(emotion, '#CCCCCC') for emotion in emotion_data.index])  # Default to gray if emotion not mapped
    )])
    
    fig.update_layout(
        height=600,
        width=800
    )
    
    # Convert to HTML
    plot_html = pio.to_html(fig, full_html=False)
    return render_template('plot.html', plot_html=plot_html, title='Overall Emotion Analysis')




import plotly.io as pio
from plotly import graph_objects as go


@app.route('/emotion/individual')
def individual_emotion():
    df = load_data()
    if df is None:
        return "Error loading data"
    
    # Filter out blank or missing emotion values
    df = df[df['emotion'].notna() & (df['emotion'] != '')]
    
    # Get unique students and emotions
    students = df['name'].unique()
    all_emotions = df['emotion'].unique()
    
    # Define a fixed color mapping for emotions (lowercase)
    color_map = {
        'neutral': '#FF851B',  # Red
        'happiness': '#2ECC40',    # Green
        'sadness': '#0074D9',      # Blue
        'anger': '#EE1010FF',    # Orange
        'surprise': '#B10DC9', # Purple
    }
    
    # Create subplots for each student
    fig = go.Figure()
    
    for student in students:
        student_df = df[df['name'] == student]
        emotion_data = student_df['emotion'].str.lower().value_counts(normalize=True) * 100
        
        # Ensure colors match emotions consistently
        colors = [color_map.get(emotion, '#808080') for emotion in emotion_data.index]
        
        fig.add_trace(go.Pie(
            labels=[f"{label.capitalize()} ({value:.1f}%)" for label, value in zip(emotion_data.index, emotion_data.values)],
            values=emotion_data.values,
            name=student,
            title=f'Emotion Analysis - {student}',
            domain={'row': students.tolist().index(student) // 2,
                   'column': students.tolist().index(student) % 2},
            hovertemplate='%{label}\nPercentage: %{value:.1f}%',
            marker=dict(colors=colors),
            showlegend=students.tolist().index(student) == 0  # Show legend only for first pie chart
        ))
    
    # Update layout to create a grid of pie charts
    fig.update_layout(
        # title='Individual Student Emotion Analysis',
        # title_x=0.5,
        grid={'rows': (len(students)+1)//2, 'columns': 2},
        height=300 * ((len(students)+1)//2),
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Convert to HTML
    plot_html = pio.to_html(fig, full_html=False)
    return render_template('plot.html', plot_html=plot_html, title='Individual Emotion Analysis')

@app.route('/emotion/monthly')
def monthly_emotion():
    df = load_data()
    if df is None:
        return "Error loading data"
    
    # Filter out blank or missing emotion values
    df = df[df['emotion'].notna() & (df['emotion'] != '')]
    
    # Normalize the emotion column
    df['emotion'] = df['emotion'].str.strip().str.lower()
    
    # Debug: Print unique emotion values
    print(df['emotion'].unique())
    
    # Group and calculate percentages
    monthly_emotion = df.groupby(['month', 'emotion']).size().unstack(fill_value=0)
    monthly_emotion_percent = monthly_emotion.apply(lambda x: x / x.sum() * 100, axis=1)
    
    # Define color map for emotions
    color_map = {
    'neutral': '#FF851B',  # Red
    'happiness': '#2ECC40',    # Green
    'sadness': '#0074D9',      # Blue
    'anger': 'rgba(238,16,16,1)',    # Orange with transparency
    'surprise': '#B10DC9', # Purple
    }
    
    # Prepare data for Plotly
    data = []
    for emotion_type in monthly_emotion_percent.columns:
        emotion_lower = emotion_type.lower()
        data.append(go.Bar(
            name=emotion_type,
            x=monthly_emotion_percent.index,
            y=monthly_emotion_percent[emotion_type],
            hovertemplate='<b>%{x} - %{fullData.name}</b><br>Percentage: %{y:.1f}%<extra></extra>',
            text=[f'{val:.1f}%' if val > 0 else '' for val in monthly_emotion_percent[emotion_type]],
            textposition='inside',
            textfont=dict(color='white', size=10),
            marker_color=color_map.get(emotion_lower, '#808080')  # Default to gray if no color found
        ))
    
    # Create figure
    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='stack',
        height=600,
        width=1000,
        xaxis_title='Month',
        yaxis_title='Percentage',
        bargap=0.1,  # Add some gap between bar groups
        bargroupgap=0.1  # Add some gap between groups
    )
    
    # Convert to HTML
    plot_html = pio.to_html(fig, full_html=False)
    return render_template('plot.html', plot_html=plot_html, title='Monthly Emotion Analysis')


@app.route('/emotion/weekly')
def weekly_emotion():
    df = load_data()
    if df is None:
        return "Error loading data"
    
    # Filter out blank or missing emotion values
    df = df[df['emotion'].notna() & (df['emotion'] != '')]
    
    # Group and calculate percentages
    weekly_emotion = df.groupby(['week', 'emotion']).size().unstack(fill_value=0)
    weekly_emotion_percent = weekly_emotion.apply(lambda x: x / x.sum() * 100, axis=1)
    
    # Define color map for emotions
    color_map = {
    'neutral': '#FF851B',  # Red
    'happiness': '#2ECC40',    # Green
    'sadness': '#0074D9',      # Blue
    'anger': 'rgba(238,16,16,1)',    # Orange with transparency
    'surprise': '#B10DC9', # Purple
    }
    
    # Prepare data for Plotly
    data = []
    for emotion_type in weekly_emotion_percent.columns:
        # Convert emotion to lowercase for color matching
        emotion_lower = emotion_type.lower()
        data.append(go.Bar(
            name=emotion_type,
            x=weekly_emotion_percent.index,
            y=weekly_emotion_percent[emotion_type],
            hovertemplate='<b>%{x} - %{fullData.name}</b><br>Percentage: %{y:.1f}%<extra></extra>',
            text=[f'{val:.1f}%' if val > 0 else '' for val in weekly_emotion_percent[emotion_type]],
            textposition='inside',
            textfont=dict(color='white', size=10),
            marker_color=color_map.get(emotion_lower, '#808080')  # Default to gray if no color found
        ))
    
    # Create figure
    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='stack',
        height=600,
        width=1000,
        xaxis_title='Week',
        yaxis_title='Percentage',
        bargap=0.1,  # Add some gap between bar groups
        bargroupgap=0.1  # Add some gap between groups
    )
    
    # Customize text and hover information
    fig.update_traces(
        texttemplate='%{text}',
        textposition='inside'
    )
    
    # Convert to HTML
    plot_html = pio.to_html(fig, full_html=False)
    return render_template('plot.html', plot_html=plot_html, title='Weekly Emotion Analysis')

# Subject Analysis Routes
@app.route('/subject/emotion')
def subject_emotion():
    df = load_data()
    if df is None:
        return "Error loading data"
    
    # Filter out blank or missing emotion values
    df = df[df['emotion'].notna() & (df['emotion'] != '')]
    
    # Group and calculate percentages
    subject_emotion = df.groupby(['subject', 'emotion']).size().unstack(fill_value=0)
    subject_emotion_percent = subject_emotion.apply(lambda x: x / x.sum() * 100, axis=1)
    
    # Define color map for emotions
    color_map = {
    'neutral': '#FF851B',  # Red
    'happiness': '#2ECC40',    # Green
    'sadness': '#0074D9',      # Blue
    'anger': 'rgba(238,16,16,1)',    # Orange with transparency
    'surprise': '#B10DC9', # Purple
    }
    
    # Prepare data for Plotly
    data = []
    for emotion_type in subject_emotion_percent.columns:
        # Convert emotion to lowercase for color matching
        emotion_lower = emotion_type.lower()
        data.append(go.Bar(
            name=emotion_type,
            x=subject_emotion_percent.index,
            y=subject_emotion_percent[emotion_type],
            hovertemplate='<b>%{x} - %{fullData.name}</b><br>Percentage: %{y:.1f}%<extra></extra>',
            text=[f'{val:.1f}%' if val > 0 else '' for val in subject_emotion_percent[emotion_type]],
            textposition='inside',
            textfont=dict(color='white', size=10),
            marker_color=color_map.get(emotion_lower, '#808080')  # Default to gray if no color found
        ))
    
    # Create figure
    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='stack',
        height=600,
        width=1000,
        xaxis_title='Subject',
        yaxis_title='Percentage',
        bargap=0.1,  # Add some gap between bar groups
        bargroupgap=0.1  # Add some gap between groups
    )
    
    # Customize text and hover information
    fig.update_traces(
        texttemplate='%{text}',
        textposition='inside'
    )
    
    # Convert to HTML
    plot_html = pio.to_html(fig, full_html=False)
    return render_template('plot.html', plot_html=plot_html, title='Subject Emotion Analysis')

@app.route('/subject/attendance')
def subject_attendance():
    df = load_data()
    if df is None:
        return "Error loading data"
    
    # Group and calculate percentages
    subject_attendance = df.groupby(['subject', 'attendance']).size().unstack(fill_value=0)
    subject_attendance_percent = subject_attendance.apply(lambda x: x / x.sum() * 100, axis=1)
    
    # Define color map for attendance types
    color_map = {
        'present': '#0000FF',  # Blue
        'absent': '#FF0000'    # Red
    }
    
    # Prepare data for Plotly
    data = []
    for attendance_type in subject_attendance_percent.columns:
        attendance_lower = attendance_type.lower()  # Normalize attendance type for color matching
        data.append(go.Bar(
            name=attendance_type,
            x=subject_attendance_percent.index,
            y=subject_attendance_percent[attendance_type],
            hovertemplate='<b>%{x} - %{fullData.name}</b><br>Percentage: %{y:.1f}%<extra></extra>',
            text=[f'{val:.1f}%' if val > 0 else '' for val in subject_attendance_percent[attendance_type]],
            textposition='inside',
            textfont=dict(color='white', size=10),
            marker_color=color_map.get(attendance_lower, '#808080')  # Default to gray if no color found
        ))
    
    # Create figure
    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='stack',
        height=600,
        width=1000,
        xaxis_title='Subject',
        yaxis_title='Percentage',
        bargap=0.1,  # Add some gap between bar groups
        bargroupgap=0.1  # Add some gap between groups
    )
    
    # Convert to HTML
    plot_html = pio.to_html(fig, full_html=False)
    return render_template('plot.html', plot_html=plot_html, title='Subject Attendance Analysis')


# Main Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/records')
def get_records():
    try:
        # Retrieve records from MongoDB
        records = list(collection.find())
        # Convert ObjectId to string for JSON serialization
        for record in records:
            record['_id'] = str(record['_id'])
        return jsonify(records)
    except Exception as e:
        print(f"Error in /records route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analysis_window')
def analysis_window():
    return render_template('ana_home.html')

if __name__ == '__main__':
    app.run(debug=True)