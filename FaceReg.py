import cv2, os, numpy as np

# 1. ฟังก์ชัน KNN สั้นกระชับ
def knn(X, y, z, k=3):
    d = np.sum((X - z) ** 2, axis=1) # Euclidean Distance
    idx = np.argsort(d)[:k]
    return np.unique(y[idx])[0]       # คืนค่าชื่อที่โหวตได้มากที่สุด

# 2. โหลด Dataset ทั้งหมด
X, y = [], []
for name in os.listdir('data'):
    p_path = f'data/{name}'
    if os.path.isdir(p_path):
        for img in os.listdir(p_path):
            X.append(cv2.imread(f'{p_path}/{img}', 0).flatten())
            y.append(name)
X, y = np.array(X), np.array(y)

# 3. อ่านกล้องและ Predict Real-time
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    face = cv2.cvtColor(frame[120:300, 250:390], cv2.COLOR_BGR2GRAY)
    
    label = knn(X, y, face.flatten()) # ทำนายชื่อ
    
    cv2.rectangle(frame, (250, 120), (390, 300), (0, 255, 0), 2)
    cv2.putText(frame, label, (250, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
