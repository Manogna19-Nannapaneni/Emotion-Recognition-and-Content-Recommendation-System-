import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from flask import session, redirect
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
import re

def is_strong_password(password):
    if len(password) < 8:
        return False
    if not re.search("[A-Z]", password):
        return False
    if not re.search("[a-z]", password):
        return False
    if not re.search("[0-9]", password):
        return False
    if not re.search("[@#$%!]", password):
        return False
    return True


# =============================
# Flask App
# =============================
app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 7)
)

model.load_state_dict(torch.load("best_emotion_model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

app.secret_key = "emotion_secret"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"

db = SQLAlchemy(app)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100), nullable=False)

    email = db.Column(db.String(120), unique=True, nullable=False)

    password = db.Column(db.String(200), nullable=False)

    phone = db.Column(db.String(15))

    age = db.Column(db.Integer)

    gender = db.Column(db.String(10))

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]
#====================================
# time based
#====================================
def get_time_bias(emotion):

    hour = datetime.now().hour

    # Morning
    if 5 <= hour < 12:

        if emotion in ["sad", "neutral","disgust"]:
            return "Morning (Low Energy)"
        else:
            return "Morning (Energetic)"

    # Afternoon
    elif 12 <= hour < 17:

        if emotion in ["sad", "angry","disgust"]:
            return "Afternoon (Stressed)"
        else:
            return "Afternoon (Focused)"

    # Evening
    elif 17 <= hour < 21:

        if emotion in ["sad", "neutral","disgust"]:
            return "Evening (Tired)"
        elif emotion == "angry":
            return "Evening (Stressed)"
        else:
            return "Evening (Relaxed)"

    # Night
    else:

        if emotion in ["happy", "surprise"]:
            return "Night (Active)"
        else:
            return "Night (Tired)"


# =============================
# Feature-based XAI
# =============================
def get_explanation(emotion, confidence):

    confidence = round(confidence*100 , 2)

    if emotion == "happy":
        return f"With {confidence}% confidence, the model focused on eye and mouth regions, detecting raised cheeks and smiling lips."

    elif emotion == "sad":
        return f"With {confidence}% confidence, the model focused on drooping eyes and downward mouth corners indicating sadness."

    elif emotion == "angry":
        return f"With {confidence}% confidence, eyebrow tension and tight lips influenced this prediction."

    elif emotion == "fear":
        return f"With {confidence}% confidence, wide eyes and stretched mouth patterns indicate fear."

    elif emotion == "surprise":
        return f"With {confidence}% confidence, raised eyebrows and open mouth show surprise."

    elif emotion == "neutral":
        return f"With {confidence}% confidence, balanced facial muscles indicate a neutral expression."
    
    elif emotion == "disgust":
        return f"With {confidence}% confidence, the model detected wrinkled nose and tightened lips indicating disgust."
    
    else:
        return "Facial feature patterns influenced this prediction."
    #==================
    # recommendations
    #==================
def get_recommendations(emotion):

    data = {

        "happy": [
            {"title":"Happy Music","url":"https://www.youtube.com/watch?v=X-QQKqJ6LWU&list=RDX-QQKqJ6LWU&start_radio=1","image":"/static/images/music.jpg","button":"Watch"},
            {"title":"Dance Songs","url":"https://www.youtube.com/watch?v=2cF8x6Mhgj4&list=PL-nuykztplfc-uD_5VhZ6vaK1p3muRF1i","image":"/static/images/dance.jpg","button":"Watch"},
            {"title":"Comedy Videos","url":"https://www.youtube.com/watch?v=OHIjMdMOIzU","image":"https://i.ytimg.com/vi/9bZkp7q19f0/hqdefault.jpg","button":"Watch"},
            {"title":"Motivation Talk","url":"https://open.spotify.com/album/02REn6bY4jZJOjKpETPai5","image":"https://i.ytimg.com/vi/mgmVOuLgFB0/hqdefault.jpg","button":"Listen"},
            {"title":"Travel Vlog","url":"https://www.youtube.com/watch?v=BFS9n4B_2xA","image":"/static/images/travel.jpg","button":"Watch"},
            {"title":"Party Mix","url":"https://www.youtube.com/watch?v=7u8MyPmhla4&list=RD7u8MyPmhla4&start_radio=1","image":"/static/images/party.jpg","button":"Watch"},
            {"title":"Music Mix","url":"https://www.youtube.com/watch?v=8ec7XwC4BAU&list=RD8ec7XwC4BAU&start_radio=1","image":"/static/images/musicmix.jpg","button":"Watch"},
            {"title":"Positive Podcast","url":"https://open.spotify.com/show/4dGamU5NVgxAgg85iNHVVN","image":"/static/images/download.png","button":"Listen"},
            {"title":"Quotes Book","url":"https://www.readcentral.com/quotes","image":"/static/default.png","button":"Read"},
            {"title":"Fun Shorts","url":"https://www.youtube.com/shorts/i3_mHGlUgoo","image":"/static/default.png","button":"Open"}
        ],

        "sad": [
            {"title":"Motivation Speech","url":"https://open.spotify.com/album/02REn6bY4jZJOjKpETPai5","image":"https://i.ytimg.com/vi/mgmVOuLgFB0/hqdefault.jpg","button":"Listen"},
            {"title":"Relax Music","url":"https://www.youtube.com/watch?v=lFcSrYw-ARY&list=RDlFcSrYw-ARY&start_radio=1","image":"https://i.ytimg.com/vi/2OEL4P1Rz04/hqdefault.jpg","button":"Listen"},
            {"title":"Healing Song","url":"https://www.youtube.com/watch?v=vPvIxwh9N2w&list=RDvPvIxwh9N2w&start_radio=1","image":"/static/images/healing.jpg","button":"Listen"},
            {"title":"Meditation","url":"https://www.youtube.com/watch?v=d4S4twjeWTs","image":"https://i.ytimg.com/vi/inpok4MKVLM/hqdefault.jpg","button":"Watch"},
            {"title":"Peace Piano","url":"https://www.youtube.com/watch?v=3NycM9lYdRI","image":"https://i.ytimg.com/vi/lFcSrYw-ARY/hqdefault.jpg","button":"Listen"},
            {"title":"Hope Talk","url":"https://www.youtube.com/watch?v=qt0fRSx5Kl4","image":"/static/images/hope.jpg","button":"Watch"},
            {"title":"Confidence Video","url":"https://www.youtube.com/watch?v=eVFzbxmKNUw","image":"/static/images/confidence.jpg","button":"Watch"},
            {"title":"Stress Relief","url":"https://www.youtube.com/watch?v=qUz93CyNIz0","image":"https://i.ytimg.com/vi/1ZYbU82GVz4/hqdefault.jpg","button":"Listen"},
            {"title":"Self Help Book","url":"https://www.infobooks.org/free-pdf-books/self-improvement/","image":"/static/images/selfhelpbooks.jpg","button":"Read"},
            {"title":"Positive Stories","url":"https://medium.com","image":"/static/images/positivestories.jpg","button":"Read"}
        ],

        "angry": [
            {"title":"Calm Meditation","url":"https://www.youtube.com/watch?v=d4S4twjeWTs","image":"https://i.ytimg.com/vi/inpok4MKVLM/hqdefault.jpg","button":"Watch"},
            {"title":"Breathing","url":"https://www.youtube.com/watch?v=tybOi4hjZFQ","image":"https://i.ytimg.com/vi/tybOi4hjZFQ/hqdefault.jpg","button":"Watch"},
            {"title":"Nature Sound","url":"https://www.youtube.com/watch?v=OdIJ2x3nxzQ","image":"https://i.ytimg.com/vi/OdIJ2x3nxzQ/hqdefault.jpg","button":"Listen"},
            {"title":"Yoga","url":"https://www.youtube.com/watch?v=v7AYKMP6rOE","image":"/static/images/yoga.jpg","button":"Watch"},
            {"title":"Soft Music","url":"https://www.youtube.com/watch?v=2OEL4P1Rz04","image":"https://i.ytimg.com/vi/2OEL4P1Rz04/hqdefault.jpg","button":"Listen"},
            {"title":"Mindfulness","url":"https://www.youtube.com/watch?v=NEMUDaLMWJ8","image":"https://i.ytimg.com/vi/ZToicYcHIOU/hqdefault.jpg","button":"Watch"},
            {"title":"Stress Book","url":"https://mindremakeproject.org/2025/07/04/free-stress-management-workbooks/","image":"/static/images/stressbooks.jpg","button":"Read"},
            {"title":"Motivation","url":"https://open.spotify.com/episode/62EQ9oeZ6YTyy5LX7jcG9o","image":"/static/images/motivation.jpg","button":"Watch"},
            {"title":"Peace Music","url":"https://www.youtube.com/watch?v=HpvedStXfTo&list=RDATfpcGVhY2VmdWwgbXVzaWM&start_radio=1","image":"https://i.ytimg.com/vi/lFcSrYw-ARY/hqdefault.jpg","button":"Listen"},
            {"title":"Podcast","url":"https://open.spotify.com/show/1QBP6aNv7BsdQWwhqxLcIC","image":"/static/images/download.png","button":"Listen"}
        ],
        "disgust": [
            {"title": "Calm Relaxing Music","url": "https://www.youtube.com/watch?v=lFcSrYw-ARY","image": "https://i.ytimg.com/vi/2OEL4P1Rz04/hqdefault.jpg","button": "Listen"},
            {"title": "Deep Breathing Exercise","url": "https://www.youtube.com/watch?v=LMS3K8_5KFA&t=30s","image": "/static/images/breathing.webp","button": "Watch"},
            {"title": "Mindfulness Meditation","url": "https://www.youtube.com/watch?v=inpok4MKVLM","image": "https://i.ytimg.com/vi/inpok4MKVLM/hqdefault.jpg","button": "Watch"},
            {"title": "Motivational Speech","url": "https://www.youtube.com/watch?v=mgmVOuLgFB0","image": "https://i.ytimg.com/vi/mgmVOuLgFB0/hqdefault.jpg","button": "Watch"},
            {"title": "Positive Podcast","url": "https://open.spotify.com/show/4dGamU5NVgxAgg85iNHVVN","image": "/static/default.png","button": "Listen"},
            {"title": "Light Comedy Clips","url": "https://www.youtube.com/shorts/GXFKlH3fJBU","image": "/static/default.png","button": "Watch"},
            {"title": "Nature Sounds","url": "https://www.youtube.com/watch?v=OdIJ2x3nxzQ","image": "https://i.ytimg.com/vi/OdIJ2x3nxzQ/hqdefault.jpg","button": "Listen"},
            {"title": "Stress Relief Music","url": "https://www.youtube.com/watch?v=2OEL4P1Rz04","image": "https://i.ytimg.com/vi/2OEL4P1Rz04/hqdefault.jpg","button": "Listen"},
            {"title": "Self Improvement Book","url": "https://www.infobooks.org/free-pdf-books/self-improvement/","image": "/static/images/selfhelpbooks.jpg","button": "Read"},
            {"title": "Positive Articles","url": "https://medium.com/tag/motivation","image": "/static/images/articles.jpg","button": "Read"}
        ],

        "fear": [
            {"title":"Confidence","url":"https://www.youtube.com/watch?v=eVFzbxmKNUw","image":"/static/images/confidence.jpg","button":"Watch"},
            {"title":"Anxiety Relief","url":"https://www.youtube.com/watch?v=WGG7MGgptxE","image":"/static/images/anxiety.jpg","button":"Watch"},
            {"title":"Relax Music","url":"https://www.youtube.com/watch?v=lFcSrYw-ARY&list=RDlFcSrYw-ARY&start_radio=1","image":"https://i.ytimg.com/vi/2OEL4P1Rz04/hqdefault.jpg","button":"Listen"},
            {"title":"Meditation","url":"https://www.youtube.com/watch?v=d4S4twjeWTs","image":"https://i.ytimg.com/vi/inpok4MKVLM/hqdefault.jpg","button":"Watch"},
            {"title":"Motivation","url":"https://www.youtube.com/watch?v=Tuw8hxrFBH8","image":"/static/images/motivation.jpg","button":"Watch"},
            {"title":"Peace Sound","url":"https://www.youtube.com/watch?v=HpvedStXfTo&list=RDATfpcGVhY2VmdWwgbXVzaWM&start_radio=1","image":"https://i.ytimg.com/vi/OdIJ2x3nxzQ/hqdefault.jpg","button":"Listen"},
            {"title":"Self Help","url":"https://www.infobooks.org/free-pdf-books/self-improvement/","image":"/static/images/selfhelp.jpg","button":"Read"},
            {"title":"Stories","url":"https://www.youtube.com/watch?v=zyr6gwcUa-A","image":"/static/default.png","button":"Read"},
            {"title":"Calm Talk","url":"https://www.youtube.com/watch?v=8jPQjjsBbIc","image":"https://i.ytimg.com/vi/ZToicYcHIOU/hqdefault.jpg","button":"Watch"},
            {"title":"Podcast","url":"https://open.spotify.com/show/1QBP6aNv7BsdQWwhqxLcIC","image":"/static/images/OIP.webp","button":"Listen"}
        ],
        "surprise": [
           {"title":"Trending Videos","url":"https://www.youtube.com/shorts/hfRt5oA1-lA","image":"/static/images/trending.jpg","button":"Watch"},
           {"title":"New Songs","url":"https://open.spotify.com/playlist/6w4NVkOj7vHg8FUXyOF1f9","image":"/static/images/newsongs.jpg","button":"Listen"},
           {"title":"Funny Shorts","url":"https://www.youtube.com/shorts/GXFKlH3fJBU","image":"/static/images/funnyshorts.jpg","button":"Watch"},
           {"title":"Viral Reels","url":"https://www.instagram.com/reels/","image":"/static/images/viralreels.jpg","button":"Open"},
           {"title":"Amazing Facts","url":"https://www.youtube.com/watch?v=FkQWpQd9Zdo","image":"/static/images/amazingfacts.jpg","button":"Watch"},
           {"title":"Magic Tricks","url":"https://www.youtube.com/shorts/65PmIz6NyT0","image":"/static/images/magictricks.jpg","button":"Watch"},
           {"title":"Science Experiments","url":"https://www.youtube.com/shorts/VxPFxDiAUko","image":"/static/images/science.jpg","button":"Watch"},
           {"title":"Travel Shorts","url":"https://www.youtube.com/shorts/mkfSZDwAaf8","image":"/static/default.png","button":"Watch"},
           {"title":"Tech Innovations","url":"https://www.youtube.com/shorts/_KIzFIgzuYQ","image":"/static/images/techinnov.jpg","button":"Watch"},
           {"title":"Interesting Articles","url":"https://tetw.org/Greats","image":"/static/images/articles.jpg","button":"Read"}
        ],


        "neutral": [
            {"title":"Study Music","url":"https://www.youtube.com/watch?v=WPni755-Krg","image":"https://i.ytimg.com/vi/WPni755-Krg/hqdefault.jpg","button":"Listen"},
            {"title":"Coding Tutorial","url":"https://www.youtube.com/watch?v=2NWeucMKrLI&list=PL6gx4Cwl9DGAKIXv8Yr6nhGJ9Vlcjyymq","image":"/static/images/coding.jpg","button":"Watch"},
            {"title":"Focus Music","url":"https://www.youtube.com/watch?v=_4kHxtiuML0","image":"https://i.ytimg.com/vi/5qap5aO4i9A/hqdefault.jpg","button":"Listen"},
            {"title":"Productivity Podcast","url":"https://open.spotify.com/show/6idQBTQNbAQEKSDJHV5OjX","image":"/static/images/OIP.webp","button":"Listen"},
            {"title":"Online Courses","url":"https://www.coursera.org","image":"/static/images/courses.png","button":"Learn"},
            {"title":"Tech News","url":"https://medium.com/tag/technology","image":"/static/images/technews.jpg","button":"Read"},
            {"title":"Learning Videos","url":"https://www.youtube.com/watch?v=cDf98hISlDA","image":"/static/images/learning.jpg","button":"Watch"},
            {"title":"Time Management","url":"https://cstn.wordpress.com/wp-content/uploads/2009/10/successful-time-management.pdf","image":"/static/images/timemanagement.jpg","button":"Read"},
            {"title":"Goal Setting","url":"https://www.youtube.com/watch?v=XpKvs-apvOs","image":"/static/images/goal.jpg","button":"Watch"},
            {"title":"Skill Building","url":"https://www.udemy.com","image":"/static/images/skill.jpg","button":"Learn"}
        ]
    }

    return data.get(emotion, [])

# =============================
# Prediction Function
# =============================
def predict_emotion(face):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = transform(face).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    emotion = emotion_labels[predicted.item()]
    confidence = confidence.item()

    return emotion, confidence

# =============================
# Routes
# =============================

@app.route("/")
def index():

    if "user" not in session:
        return redirect("/login")

    return render_template("/home.html")



@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        phone = request.form["phone"]
        age = request.form["age"]
        gender = request.form["gender"]

        # ✅ Check password strength
        if not is_strong_password(password):
            return render_template(
                "register.html",
                error="Password must be 8+ chars with uppercase, lowercase, number, special symbol."
            )

        # ✅ Check if email already exists
        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            return render_template(
                "register.html",
                error="This email is already registered. Please login."
            )

        # ✅ Create new user
        new_user = User(
            name=name,
            email=email,
            password=password,
            phone=phone,
            age=age,
            gender=gender
        )

        db.session.add(new_user)
        db.session.commit()

        return redirect("/login")

    # GET request
    return render_template("register.html")


@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        # Check email
        if not user:
            return render_template(
                "login.html",
                error="Email not registered"
            )

        # Check password
        if user.password != password:
            return render_template(
                "login.html",
                error="Invalid password"
            )

        # ✅ Save user in session
        session["user"] = user.name

        return redirect("/home")

    return render_template("login.html")

@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect("/login")
@app.route("/emotion")
def emotion():

    if "user" not in session:
        return redirect("/login")

    return render_template("dashboard.html")


@app.route("/detect", methods=["POST"])
def detect():

    data = request.json["image"]

    # Decode image
    img = cv2.imdecode(
        np.frombuffer(
            base64.b64decode(data.split(",")[1]),
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    if img is None:
        return jsonify({"emotion": "Camera Error"})


    # Resize
    img = cv2.resize(img, (640, 480))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(80, 80)
    )

    # No face
    if len(faces) == 0:
       return jsonify({"emotion": "No Face Detected"})


    # Take first face
    x, y, w, h = faces[0]

    face_img = img[y:y+h, x:x+w]


    # Predict
    emotion, confidence = predict_emotion(face_img)
    time_bias = get_time_bias(emotion)

    explanation = get_explanation(emotion, confidence)
    recommendations = get_recommendations(emotion)



    recs = recommendations

    return jsonify({
    "emotion": emotion,
    "confidence": round(confidence * 100 , 2),
    "explanation": explanation,
    "time_bias": time_bias,
    "recommendations": recommendations
    })
@app.route("/home")
def home():

    if "user" not in session:
        return redirect("/login")

    return render_template("home.html")


@app.route("/about")
def about():

    if "user" not in session:
        return redirect("/login")

    return render_template("about.html")


@app.route("/instructions")
def instructions():

    if "user" not in session:
        return redirect("/login")

    return render_template("instructions.html")


@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/login")

    return render_template("dashboard.html")


@app.route("/profile")
def profile():

    if "user" not in session:
        return redirect("/login")

    user = User.query.filter_by(name=session["user"]).first()

    return render_template("profile.html", user=user)
@app.route("/edit_profile", methods=["GET", "POST"])
def edit_profile():

    if "user" not in session:
        return redirect("/login")

    user = User.query.filter_by(
        name=session["user"]
    ).first()

    if request.method == "POST":

        user.name = request.form["name"]
        user.phone = request.form["phone"]
        user.age = request.form["age"]
        user.gender = request.form["gender"]

        db.session.commit()

        session["user"] = user.name   # update session name

        return redirect("/profile")

    return render_template(
        "edit_profile.html",
        user=user
    )

# =============================
# Run App
# =============================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()   # ✅ Creates all tables
    app.run(debug=True)

