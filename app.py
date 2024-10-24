from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io

app = Flask(__name__)

def process_image(image):
    try:
        # Modeli yükle
        model = YOLO('yeni.pt')

        # Görüntüyü NumPy array'e dönüştür
        image_np = np.array(image)

        # Tahmin yap
        sonuc = model.predict(source=image_np)

        # İlk sonuç üzerinde işlem yap
        detections = sonuc[0].boxes.data.cpu().numpy()

        # Görüntü üzerinde çizim yapmak için ImageDraw modülünü kullan
        draw = ImageDraw.Draw(image)

        # Font ayarları
        font_path = "C:/Windows/Fonts/arial.ttf"  # Windows için örnek font yolu, kendi sisteminize göre değiştirin
        font_size = 20
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
            print("Font yüklenemedi, varsayılan font kullanılacak.")

        # Tespit edilen nesneler üzerinde döngü
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if conf > 0.5:
                # Dikdörtgen çiz
                draw.rectangle([(x1, y1), (x2, y2)], outline="black", width=15)

                # Nesne ismi ve conf yazdır
                text = f'{model.names[int(cls)]} {conf:.2f}'

                # Textbox boyutunu hesapla
                text_bbox = draw.textbbox((x1, y1), text, font=font)
                text_background = [(text_bbox[0], text_bbox[1] - 10), (text_bbox[2], text_bbox[3] + 10)]

                # Textbox arka planını ve metni çiz
                draw.rectangle(text_background, fill="red")
                draw.text((text_bbox[0] + 5, text_bbox[1] - 5), text, fill="white", font=font)

                print(f'Nesne: {model.names[int(cls)]}, Güven Skoru: {conf}')

        return image
    except Exception as e:
        print(f"Error during image processing: {e}")
        raise

@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        image = Image.open(file.stream)

        processed_image = process_image(image)

        # İşlenmiş görüntüyü bellekte sakla ve yanıt olarak döndür
        img_byte_arr = io.BytesIO()
        processed_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return send_file(io.BytesIO(img_byte_arr), mimetype='image/png')
    except Exception as e:
        print(f"Error during file upload or processing: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
