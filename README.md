# Face Recognition Project 🎯

Sistema completo de **detecção e reconhecimento facial em tempo real**, utilizando:

- 🔍 Detecção de faces com **MTCNN**
- 🧠 Embeddings faciais com **FaceNet (TensorFlow)**
- 📹 Captura em tempo real com **OpenCV**
- 📊 Logging com **Pandas**
- 📈 Similaridade com **scikit-learn**

---

## 🖼️ Demonstração

Reconhece múltiplas faces simultaneamente via webcam ou vídeo.

![demo](https://media.giphy.com/media/YOUR_GIF_LINK/giphy.gif) <!-- opcional -->

---

## 📦 Requisitos

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Pandas
- NumPy
- MTCNN
- scikit-learn

Instale com:

```bash
pip install -r requirements.txt

---

🔌 Download do Modelo FaceNet

Baixe o modelo pré-treinado FaceNet (facenet_keras.h5) por aqui:

📥 Download do modelo (Google Drive)

Coloque o arquivo na raiz do projeto.

---

🧠 Adicionando Pessoas ao Reconhecimento

Adicione imagens de referência em:

known_faces/

├── nome_pessoa/

│   ├── foto1.jpg

│   ├── foto2.jpg

---

▶️ Como Executar

Conecte sua webcam.

Rode o script principal:

python main.py
