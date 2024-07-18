import cv2
import numpy as np
import time

# Video dosyasının yolu
video_path = "stabilized_video_block_matching.mp4"

# Videoyu yakalama
cap = cv2.VideoCapture(video_path)

# Park alanlarının koordinatları
parking_slots = [(55, 100), (56, 146), (51, 192), (51, 241), (53, 290), (55, 337),
                 (46, 385), (52, 431), (53, 479), (52, 527), (51, 573), (56, 623),

                 (163, 99), (164, 147), (162, 194), (159, 243), (161, 290), (162, 339),
                 (160, 388), (162, 429), (163, 479), (168, 525), (167, 576), (165, 620),

                 (405, 90), (402, 138), (395, 193), (402, 245), (402, 289), (402, 338),
                 (404, 382), (405, 427), (405, 526), (403, 569), (406, 619),

                 (514, 92), (511, 139), (514, 187), (512, 236), (511, 284), (513, 329),
                 (511, 380), (511, 426), (512, 524), (512, 568), (513, 620),

                 (751, 88), (751, 136), (750, 188), (753, 232), (753, 276), (751, 330),
                 (753, 377), (755, 417), (753, 472), (757, 520), (749, 567), (760, 616),

                 (903, 145),  (892, 190), (893, 235),  (894, 284), (897, 325), (898, 375),
                 (901, 424), (907, 474), (910, 522), (901, 576),  (901, 620)]

# Dikdörtgenin genişliği ve yüksekliği
rect_width, rect_height = 106, 37
# Boş park yeri tespiti için eşik değeri
threshold = 30
# Son çağrı zamanı ve önceki boş park yeri sayısı
last_call_time = time.time()
prevFreeslots = 0

# İkili görüntüye dönüştürme fonksiyonu
def convert_to_binary(frame):
    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(grayScale, 150, 255, cv2.THRESH_BINARY)
    return binary

# Park alanlarını işaretleme fonksiyonu
def mark_slots(frame, binary_frame):
    global last_call_time
    global prevFreeslots

    current_time = time.time()

    # Boş park yeri sayısını başlat
    freeslots = 0

    for x, y in parking_slots:
        #park alanları için kutuların kordinatlarını belirle
        x1 = x + 8
        x2 = x + rect_width - 13
        y1 = y + 3
        y2 = y + rect_height

        start_point, stop_point = (x1, y1), (x2, y2)

        #kutuların içinde kalan yeri kırp
        crop = binary_frame[y1:y2, x1:x2]

        # siyah olmayan pikselleri say
        count = cv2.countNonZero(crop)

        # park alanlarının boş veya dolu olarak işaretle
        if count < threshold:
            color = (0, 255, 0)  # Yeşil
            thick = 5
            freeslots += 1
        else:
            color = (0, 0, 255)  # Kırmızı
            thick = 2

        # Dikdörtgeni çiz
        cv2.rectangle(frame, start_point, stop_point, color, thick)

    # boş alarlarının sayısını yazdır ve sürekli güncelle
    if current_time - last_call_time >= 0.1:
        cv2.putText(frame, "Free Slots: " + str(freeslots), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        last_call_time = current_time
        prevFreeslots = freeslots
    else:
        cv2.putText(frame, "Free Slots: " + str(prevFreeslots), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    return frame

# Sonsuz döngüde video karelerini işleyin
while True:
    # Videoyu oku
    ret, frame = cap.read()

    #ret işlemin başarılı mı başarısı< mı olduğunu gösterir
    if not ret:
        break

    # binary görüntüye dönüştür
    binary_frame = convert_to_binary(frame)

    # Park alanlarını işaretle
    out_image = mark_slots(frame, binary_frame)

    # Orijinal video ve ikili görüntüyü göster
    cv2.imshow("Parking Spot Detector", out_image)
    #cv2.imshow("Binary Image", binary_frame)

    # Çıkış koşulu
    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        break
# Videoyu kapat ve tüm pencereleri yok et
time.sleep(25)
cap.release()
cv2.destroyAllWindows()
