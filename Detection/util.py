import numpy as np
import cv2

# Non-Maximum Suppression (NMS) fonksiyonu
def NMS(boxes, class_ids, confidences, overlapThresh=0.5):

    # Gelen kutuları, sınıf ID'lerini ve güven skorlarını numpy dizilerine dönüştür.
    boxes = np.asarray(boxes)
    class_ids = np.asarray(class_ids)
    confidences = np.asarray(confidences)

    # Eğer hiç kutu yoksa boş listeler döndür.
    if len(boxes) == 0:
        return [], [], []

    # Kutuların sol üst köşelerinin koordinatlarını hesapla.
    x1 = boxes[:, 0] - (boxes[:, 2] / 2)  # Sol üst köşe x koordinatı
    y1 = boxes[:, 1] - (boxes[:, 3] / 2)  # Sol üst köşe y koordinatı

    # Kutuların sağ alt köşelerinin koordinatlarını hesapla.
    x2 = boxes[:, 0] + (boxes[:, 2] / 2)  # Sağ alt köşe x koordinatı
    y2 = boxes[:, 1] + (boxes[:, 3] / 2)  # Sağ alt köşe y koordinatı

    # Kutuların alanlarını hesapla.
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Kutuların indekslerini al.
    indices = np.arange(len(x1))

    # Her bir kutu için tekrarla.
    for i, box in enumerate(boxes):

        # Geçici indeksler oluştur (şu anki kutu hariç).
        temp_indices = indices[indices != i]

        # Kesişme kutusunun koordinatlarını bul.
        xx1 = np.maximum(box[0] - (box[2] / 2), boxes[temp_indices, 0] - (boxes[temp_indices, 2] / 2))
        yy1 = np.maximum(box[1] - (box[3] / 2), boxes[temp_indices, 1] - (boxes[temp_indices, 3] / 2))
        xx2 = np.minimum(box[0] + (box[2] / 2), boxes[temp_indices, 0] + (boxes[temp_indices, 2] / 2))
        yy2 = np.minimum(box[1] + (box[3] / 2), boxes[temp_indices, 1] + (boxes[temp_indices, 3] / 2))

        # Kesişme kutusunun genişlik ve yüksekliğini hesapla.
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Kesişme oranını hesapla.
        overlap = (w * h) / areas[temp_indices]

        # Eğer kesişme oranı belirlenen eşiği aşarsa, kutuyu kaldır.
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]

    # Geriye kalan kutuları, sınıf ID'lerini ve güven skorlarını döndür.
    return boxes[indices], class_ids[indices], confidences[indices]

# Model çıktılarının alındığı fonksiyon
def get_outputs(net):

    # Ağdaki katman isimlerini al.
    layer_names = net.getLayerNames()

    # Bağlanmamış çıkış katmanlarını al.
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Modeli ileri doğru çalıştırarak çıktıları al.
    outs = net.forward(output_layers)

    # Çıktıları filtrele (güven skoru 0.1'den büyük olanları seç).
    outs = [c for out in outs for c in out if c[4] > 0.1]

    # Filtrelenmiş çıktıları döndür.
    return outs

# Görüntü üzerine dikdörtgen çizme fonksiyonu
def draw(bbox, img):

    # Kutunun merkez koordinatları ve boyutları.
    xc, yc, w, h = bbox

    # Görüntü üzerine dikdörtgen çiz.
    img = cv2.rectangle(img,
                        (xc - int(w / 2), yc - int(h / 2)),  # Sol üst köşe
                        (xc + int(w / 2), yc + int(h / 2)),  # Sağ alt köşe
                        (0, 255, 0), 20)  # Renk ve kalınlık
    # Dikdörtgen çizilmiş görüntüyü döndür.
    return img
