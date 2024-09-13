import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import util
import re

# Model dosya yolları tanımlanır.
model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')  # YOLOv3 konfigürasyon dosyasının yolu.
model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')  # YOLOv3 ağırlık dosyasının yolu.
class_names_path = os.path.join('.', 'model', 'class.names')  # Sınıf isimlerinin bulunduğu dosyanın yolu.

# Giriş dizini tanımlanır.
input_dir = r"C:\\Users\\Projects\\LPDS\\data"  # Giriş görüntülerinin bulunduğu dizin.

# Görüntü işleme fonksiyonu tanımlanır.
def image_processing(img_path):

    # Sınıf isimlerini yükler.
    with open(class_names_path, 'r') as f:
        class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
        f.close()

    # Modeli yükler.
    net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

    # Görüntüyü yükler.
    img = cv2.imread(img_path)
    H, W, _ = img.shape

    # Görüntüyü blob formatına dönüştürür.
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    # Tespitleri alır.
    net.setInput(blob)
    detections = util.get_outputs(net)

    bboxes = []  # Bounding box listesi.
    class_ids = []  # Sınıf ID listesi.
    scores = []  # Güven skoru listesi.

    # Tespit edilen her nesne için tekrarla.
    for detection in detections:
        bbox = detection[:4]  # Bounding box koordinatları.
        xc, yc, w, h = bbox  # Merkez koordinatları ve boyutlar.
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]  # Orijinal görüntü boyutlarına göre ölçekle.
        bbox_confidence = detection[4]  # Bounding box güven skoru.
        class_id = np.argmax(detection[5:])  # En yüksek skorlu sınıf ID'si.
        score = np.amax(detection[5:])  # En yüksek skor.
        bboxes.append(bbox)  # Bounding box'ı listeye ekle.
        class_ids.append(class_id)  # Sınıf ID'sini listeye ekle.
        scores.append(score)  # Skoru listeye ekle.

    # NMS (Non-Maximum Suppression) uygulayarak gereksiz kutuları filtrele.
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

    # OCR (Optik Karakter Tanıma) işlemi için EasyOCR okuyucusunu başlat.
    reader = easyocr.Reader(['en'])
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox

        # Bounding box'ı sıkı bir şekilde ayarla.
        x1 = max(0, int(xc - w / 2))
        y1 = max(0, int(yc - h / 2))
        x2 = min(W, int(xc + w / 2))
        y2 = min(H, int(yc + h / 2))

        # Plakanın bölgesini kırp ve kopyala.
        license_plate = img[y1:y2, x1:x2, :].copy()

        # Bounding box'ı görüntü üzerine çiz.
        license_plate = img[y1:y2, x1:x2, :].copy()
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        license_plate_gray = cv2.equalizeHist(license_plate_gray)
        license_plate_gray = cv2.fastNlMeansDenoising(license_plate_gray, h=30, templateWindowSize=7,
                                                      searchWindowSize=21)
        license_plate_thresh = cv2.adaptiveThreshold(license_plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY_INV, 11, 2)

        # OCR işlemi
        output = reader.readtext(license_plate_thresh)

        license_plate_text = ""  # Plaka metni
        for out in output:
            text_bbox, text, text_score = out  # OCR çıktısı ve skoru.
            print(f"OCR Detected: {text} with score {text_score}")  # OCR çıktısını ve skorunu yazdır (debug için).
            if text_score > 0.4:
                # Karakterleri filtrele ve birleştir.
                filtered_text = re.sub(r'[^A-Z0-9]', '', text.upper())  # Sadece büyük harfler ve rakamlar.
                license_plate_text += filtered_text + " "

        # İstenmeyen bölgeleri (mavi alan ve alt metin) filtrele.
        license_plate_text = license_plate_text.replace(" ", "")
        if len(license_plate_text) >= 7:  # Minimum uzunluğu kontrol et.
            # Beklenen formatta olup olmadığını kontrol et (3 harf ve 4 rakam).
            possible_plate = re.findall(r'[A-Z]{3}[0-9]{4}', license_plate_text)
            if possible_plate:
                print(f"Plaka: {possible_plate[0]}")
            else:
                print("Plaka bulunamadı veya OCR başarısız oldu.")
        else:
            print("Plaka bulunamadı veya OCR başarısız oldu.")

    # Görüntüleri ve sonuçları görselleştir.
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(cv2.cvtColor(license_plate_gray, cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))
    plt.show()

# Ana fonksiyon
def main():
    while True:
        img_name = input("İşlemek istediğiniz resmin adını (örneğin, 1.jpg) girin veya çıkmak için 'q' tuşlayın: ")  # Kullanıcıdan resim adı girmesini iste.
        if img_name.lower() == 'q':  # Eğer kullanıcı 'q' girerse programdan çık.
            print("Programdan çıkılıyor.")
            break
        img_path = os.path.join(input_dir, img_name)  # Resmin tam yolunu oluştur.
        if os.path.exists(img_path):  # Eğer resim varsa işleme başla.
            image_processing(img_path)
        else:
            print("Geçersiz dosya adı. Lütfen tekrar deneyin.")  # Geçersiz dosya adı durumunda uyarı ver.

# Bu dosya doğrudan çalıştırıldığında ana fonksiyonu çalıştır.
if __name__ == "__main__":
    main()
