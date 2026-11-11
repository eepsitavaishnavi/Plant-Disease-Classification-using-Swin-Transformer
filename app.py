import io, os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH  = os.path.join(BASE_DIR, 'swin_best.pth')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Load checkpoint ───────────────────────────────────────────────────────────
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(
        "\n❌  swin_best.pth not found!\n"
        f"    Place your model file here: {CKPT_PATH}\n")
print(f"Loading model from {CKPT_PATH} …")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

CLASS_NAMES = ckpt['class_names']      # saved by our training script
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes: {NUM_CLASSES}")

model = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=False,
    num_classes=NUM_CLASSES,
)
model.load_state_dict(ckpt['model_state_dict'])
model.to(DEVICE).eval()
print("✅ Model ready!")

# ── Inference transform (identical to val/test in training) ───────────────────
INFER_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Helpers ───────────────────────────────────────────────────────────────────
DISEASE_INFO = {
    # Apple
    'Apple___Apple_scab':       ('Apple',     'Apple Scab',         'Fungal infection causing dark, scabby lesions on leaves and fruit.'),
    'Apple___Black_rot':        ('Apple',     'Black Rot',          'Fungal disease causing brown spots and fruit rot.'),
    'Apple___Cedar_apple_rust': ('Apple',     'Cedar Apple Rust',   'Fungal disease producing bright orange spots on leaves.'),
    'Apple___healthy':          ('Apple',     'Healthy',            'No disease detected. The plant appears healthy.'),
    # Blueberry
    'Blueberry___healthy':      ('Blueberry', 'Healthy',            'No disease detected. The plant appears healthy.'),
    # Cherry
    'Cherry_(including_sour)___Powdery_mildew': ('Cherry', 'Powdery Mildew', 'White powdery coating on leaves caused by fungal pathogen.'),
    'Cherry_(including_sour)___healthy':        ('Cherry', 'Healthy',        'No disease detected. The plant appears healthy.'),
    # Corn
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': ('Corn', 'Gray Leaf Spot',    'Fungal disease causing rectangular grey lesions on leaves.'),
    'Corn_(maize)___Common_rust_':                        ('Corn', 'Common Rust',       'Fungal infection producing rust-coloured pustules on leaves.'),
    'Corn_(maize)___Northern_Leaf_Blight':                ('Corn', 'Northern Leaf Blight', 'Causes long elliptical grey-green lesions on leaves.'),
    'Corn_(maize)___healthy':                             ('Corn', 'Healthy',           'No disease detected. The plant appears healthy.'),
    # Grape
    'Grape___Black_rot':                       ('Grape', 'Black Rot',          'Fungal disease causing black circular lesions on leaves and fruit.'),
    'Grape___Esca_(Black_Measles)':            ('Grape', 'Esca (Black Measles)','Complex fungal disease causing leaf striping and berry shrivel.'),
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': ('Grape', 'Leaf Blight',    'Fungal infection causing brown necrotic patches on leaves.'),
    'Grape___healthy':                         ('Grape', 'Healthy',            'No disease detected. The plant appears healthy.'),
    # Orange
    'Orange___Haunglongbing_(Citrus_greening)': ('Orange', 'Citrus Greening (HLB)', 'Bacterial disease spread by insects causing yellowing and fruit drop.'),
    # Peach
    'Peach___Bacterial_spot':  ('Peach', 'Bacterial Spot', 'Causes water-soaked spots on leaves, fruit, and twigs.'),
    'Peach___healthy':         ('Peach', 'Healthy',        'No disease detected. The plant appears healthy.'),
    # Pepper
    'Pepper,_bell___Bacterial_spot': ('Bell Pepper', 'Bacterial Spot', 'Causes small, water-soaked lesions on leaves and fruit.'),
    'Pepper,_bell___healthy':        ('Bell Pepper', 'Healthy',        'No disease detected. The plant appears healthy.'),
    # Potato
    'Potato___Early_blight': ('Potato', 'Early Blight', 'Fungal disease causing concentric dark rings on older leaves.'),
    'Potato___Late_blight':  ('Potato', 'Late Blight',  'Destructive disease causing rapid browning and plant collapse.'),
    'Potato___healthy':      ('Potato', 'Healthy',      'No disease detected. The plant appears healthy.'),
    # Raspberry
    'Raspberry___healthy': ('Raspberry', 'Healthy', 'No disease detected. The plant appears healthy.'),
    # Soybean
    'Soybean___healthy': ('Soybean', 'Healthy', 'No disease detected. The plant appears healthy.'),
    # Squash
    'Squash___Powdery_mildew': ('Squash', 'Powdery Mildew', 'White powdery fungal coating that weakens the plant over time.'),
    # Strawberry
    'Strawberry___Leaf_scorch': ('Strawberry', 'Leaf Scorch', 'Fungal disease causing purplish-red spots that dry out the leaf.'),
    'Strawberry___healthy':     ('Strawberry', 'Healthy',     'No disease detected. The plant appears healthy.'),
    # Tomato
    'Tomato___Bacterial_spot':                          ('Tomato', 'Bacterial Spot',          'Water-soaked dark spots on leaves, stems, and fruit.'),
    'Tomato___Early_blight':                            ('Tomato', 'Early Blight',            'Dark concentric ring lesions usually starting from older leaves.'),
    'Tomato___Late_blight':                             ('Tomato', 'Late Blight',             'Rapid browning of leaves and stems; white mold in humid conditions.'),
    'Tomato___Leaf_Mold':                               ('Tomato', 'Leaf Mold',               'Yellow patches on upper leaf, olive-green mold on underside.'),
    'Tomato___Septoria_leaf_spot':                      ('Tomato', 'Septoria Leaf Spot',      'Circular spots with dark borders and light centres on leaves.'),
    'Tomato___Spider_mites Two-spotted_spider_mite':    ('Tomato', 'Spider Mites',            'Tiny pests causing stippling, yellowing and webbing on leaves.'),
    'Tomato___Target_Spot':                             ('Tomato', 'Target Spot',             'Concentric ring lesions on leaves resembling a target pattern.'),
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus':           ('Tomato', 'Yellow Leaf Curl Virus',  'Virus spread by whiteflies causing leaf curl and yellowing.'),
    'Tomato___Tomato_mosaic_virus':                     ('Tomato', 'Mosaic Virus',            'Viral disease causing mosaic-like discolouration and leaf distortion.'),
    'Tomato___healthy':                                 ('Tomato', 'Healthy',                 'No disease detected. The plant appears healthy.'),
}


def get_info(class_name):
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    parts = class_name.split('___')
    crop  = parts[0].replace('_', ' ')
    cond  = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
    return crop, cond, 'No additional information available.'


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img    = Image.open(io.BytesIO(file.read())).convert('RGB')
        tensor = INFER_TF(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1)[0]

        top5_probs, top5_idx = probs.topk(5)

        top5 = []
        for prob, idx in zip(top5_probs.tolist(), top5_idx.tolist()):
            cn = CLASS_NAMES[idx]
            crop, condition, description = get_info(cn)
            top5.append({
                'class':       cn,
                'crop':        crop,
                'condition':   condition,
                'description': description,
                'confidence':  round(prob * 100, 2),
                'healthy':     'healthy' in cn.lower(),
            })

        return jsonify({'predictions': top5})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
