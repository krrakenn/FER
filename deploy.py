import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization

def create_model():
    base_model = DenseNet121(include_top=False, weights=None, input_shape=(96, 96, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(8, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_sum(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
    return focal_loss_fixed

model = create_model()
model.load_weights('9833.h5') 

detector = MTCNN()

CLASS_NAMES = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral', 'Contempt']

def detect_emotions(image):
    rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_image)
    
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)
    
    for face in faces:
        x, y, w, h = face['box']
        x2, y2 = x + w, y + h
        
        face_region = rgb_image[y:y2, x:x2]
        if face_region.size == 0:
            continue
        
        face_array = cv2.resize(face_region, (96, 96))
        face_array = face_array.astype('float32') / 255.0
        face_array = np.expand_dims(face_array, axis=0)
        
        predictions = model.predict(face_array)
        emotion = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)
        
        draw.rectangle([(x, y), (x2, y2)], outline='#00ff00', width=2)
        text = f"{emotion} ({confidence:.2f})"
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        text_width, text_height = draw.textsize(text, font=font)
        text_position = (x, y - text_height - 5) if y - text_height - 5 > 0 else (x, y2)
        
        draw.rectangle(
            [(text_position[0] - 2, text_position[1] - 2),
             (text_position[0] + text_width + 2, text_position[1] + text_height + 2)],
            fill='#00ff00'
        )
        draw.text(text_position, text, fill='black', font=font)
    
    return pil_image

css = """
.gradio-container {
    background: linear-gradient(45deg, #1a1a1a, #2a2a2a);
    color: white;
}
h1 {
    color: #EA580C !important;
    text-align: center;
    font-family: 'Arial', sans-serif;
}
footer {
    visibility: hidden;
}"""


with gr.Blocks(css=css) as demo:
    gr.Markdown("# 😠🤢😨😄 Emotion Detector 😟😮😶😐", elem_classes="header")
    gr.Markdown("Upload a photo with faces and see real-time emotion predictions!")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="pil")
            submit_btn = gr.Button("Detect Emotions 🚀", variant="primary")
        with gr.Column():
            image_output = gr.Image(label="Detection Results", elem_id="output-image")
    
    submit_btn.click(
        fn=detect_emotions,
        inputs=image_input,
        outputs=image_output
    )

demo.launch()



