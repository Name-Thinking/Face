import cv2
import numpy as np
import os

def knn(X, y, z, k=5):
    # คำนวณระยะห่างแบบ Euclidean distance ระหว่างภาพใหม่ (z) กับภาพใน Dataset (X)
    d = np.sum((X - z) ** 2, axis=1)
    # ดึง index ของภาพที่มีระยะห่างใกล้ที่สุด k อันดับแรก
    idx = np.argsort(d)[:k]
    # นับจำนวนการโหวตของแต่ละคลาส (ชื่อคน)
    cls, vote = np.unique(y[idx], return_counts=True)
    # คืนค่าคลาสที่ได้รับการโหวตมากที่สุด
    return cls[np.argmax(vote)]

# 1. โหลดข้อมูลภาพและ Label จากโฟลเดอร์ data
X = []
y = []

data_path = 'data'

if os.path.exists(data_path):
    for name in os.listdir(data_path):
        person_dir = os.path.join(data_path, name)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(person_dir, img_name)
                    # อ่านภาพแบบ Grayscale เพื่อให้ตรงกับโครงสร้างที่ FaceTrain บันทึกไว้
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        X.append(img.flatten())
                        y.append(name)

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("ไม่พบข้อมูลภาพในโฟลเดอร์ 'data' กรุณารัน FaceTrain.py เพื่อเก็บข้อมูลใบหน้าก่อน")
    exit()

print(f"โหลดข้อมูลสำเร็จ: พบทั้งหมด {len(X)} ภาพ จากคลาส: {np.unique(y)}")

# 2. เปิดกล้องเพื่อทำนายผลแบบ Real-time
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # วาดกรอบโฟกัสใบหน้าตำแหน่งเดียวกับ FaceTrain
    cv2.rectangle(frame, (250, 120), (390, 300), (0, 255, 0), 2)
    
    # ตัดภาพส่วนใบหน้าและแปลงเป็น Grayscale
    face = cv2.cvtColor(frame[120:300, 250:390, :], cv2.COLOR_BGR2GRAY)
    
    # แปลงภาพเป็น 1D Vector เพื่อส่งให้ KNN
    z = face.flatten()
    
    # ทำนายผลด้วย KNN
    predicted_name = knn(X, y, z, k=5)

    # แสดงชื่อที่ทำนายได้บนหน้าจอ
    cv2.putText(frame, predicted_name, (250, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Face Recognition', frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()