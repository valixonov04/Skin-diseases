import streamlit as st
from fastai.vision.all import *
import pathlib
from fasttransform.transform import Transform, Pipeline
import pathlib
import os
print(os.path.exists('food_mix.pkl'))
plt = platform.system()
if plt =="Linux": pathlib.WindowsPath = pathlib.PosixPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Model faylining mavjudligini tekshirish
model_path = 'teri_kasaligi.pkl'
if not os.path.exists(model_path):
    st.error("Model fayli topilmadi! Iltimos, 'food_mix.pkl' faylini tekshiring.")
    st.stop()

# Streamlit sahifasini sozlash
st.set_page_config(page_title="Teri kasaliglari klassifikatsiyasi 10 ta kasalik uchun.  ", page_icon="💊", layout="centered")

# CSS stilini qo‘shish
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
        transition: all 0.2s ease-in-out;
    }
    .stFileUploader {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 10px;
        background-color: #ffffff;
    }
    .stSuccess {
        background-color: #e6f3e6;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Sarlavha va ta'rif
st.title("Teri kasaliglari 💊")
st.markdown("Bu AI model deep learning orqali fine-tuning qilingan bo'lib sizning 10 turdagi teri kasaliglarini klassifikatsiya qilib beradi !")
st.markdown("**Rasm yuklang va natijani ko‘ring!**")

# Rasm yuklash
file = st.file_uploader("Rasmni tanlang", type=["jpg", "jpeg", "png", "webp"], help="Faqat JPG, JPEG, PNG yoki WEBP formatidagi rasmlar qabul qilinadi.")

# Sessiya holatini boshqarish
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Rasmni tekshirish tugmasi
if st.button("Rasmni Tekshirish", key="check_button"):
    if file is not None:
        st.session_state.uploaded_file = file
    else:
        st.warning("Iltimos, avval rasm yuklang!")

# Natija va grafikni ko‘rsatish
if st.session_state.uploaded_file is not None:
    try:
        # Rasmni yuklash va ko‘rsatish
        img = PILImage.create(st.session_state.uploaded_file)
        st.image(img, caption="Yuklangan rasm", use_column_width=True)

        # Modelni yuklash va bashorat qilish
        model = load_learner(model_path)
        prediction, _, probs = model.predict(img)

        # Natijani ko‘rsatish
        st.markdown(f"<div class='stSuccess'>Tasnif: <strong>{prediction}</strong> (Ehtimollik: {probs.max().item():.2%})</div>", unsafe_allow_html=True)

        # Ehtimolliklar uchun grafik
        st.subheader("Klassifikatsiya Ehtimolliklari")
        class_names = model.dls.vocab
        prob_values = probs.numpy() * 100  # Foizga aylantirish

        # Bar chart yaratish
        chart_data = {
            "type": "bar",
            "data": {
                "labels": list(class_names),
                "datasets": [{
                    "label": "Ehtimollik (%)",
                    "data": prob_values.tolist(),
                    "backgroundColor": ["#4CAF50", "#FF9800", "#2196F3", "#F44336", "#9C27B0"],
                    "borderColor": ["#388E3C", "#F57C00", "#1976D2", "#D32F2F", "#7B1FA2"],
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Ehtimollik (%)"
                        }
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Klasslar"
                        }
                    }
                },
                "plugins": {
                    "legend": {
                        "display": False
                    }
                }
            }
        }
        st.components.v1.html(f"""
            <div style='background-color: white; padding: 20px; border-radius: 10px;'>
                <canvas id='myChart'></canvas>
                <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
                <script>
                    const ctx = document.getElementById('myChart').getContext('2d');
                    new Chart(ctx, {JSON.stringify(chart_data)});
                </script>
            </div>
        """, height=400)

    except Exception as e:
        st.error(f"Xatolik yuz berdi: {str(e)}")

# Faylni tozalash tugmasi
if st.button("Yuklangan Rasmni Tozalash", key="clear_button"):
    st.session_state.uploaded_file = None
    st.rerun()

# Qo‘shimcha ma'lumot
with st.expander("Qo‘llanma"):
    st.markdown("""
    - **Rasm yuklash**: JPG, JPEG, PNG yoki WEBP formatidagi rasmni yuklang.
    - **Tekshirish**: "Rasmni Tekshirish" tugmasini bosing.
    - **Natija**: Model rasmni tahlil qilib, tasnif va ehtimollikni ko‘rsatadi.
    - **Grafik**: Har bir klass uchun ehtimollik foizlarini ko‘rsatuvchi grafik.
    - **Tozalash**: Yuklangan rasmni olib tashlash uchun "Yuklangan Rasmni Tozalash" tugmasini bosing.
    """)
