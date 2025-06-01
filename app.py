import streamlit as st
from fastai.vision.all import *
import os

# Model faylining mavjudligini tekshirish
model_path = 'teri_kasaligi.pkl'
if not os.path.exists(model_path):
    st.error(f"Model fayli topilmadi: {model_path}")
    st.write(f"Current directory: {os.getcwd()}")
    st.stop()

# Modelni yuklash
model = load_learner(model_path)

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
        background-color: #11a811;
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
# Ehtimolliklar uchun grafik
st.subheader("Klassifikatsiya Ehtimolliklari")
class_names = model.dls.vocab
prob_values = probs.numpy() * 100  # Foizga aylantirish
chart_data = {name: prob for name, prob in zip(class_names, prob_values)}
st.bar_chart(chart_data)

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
