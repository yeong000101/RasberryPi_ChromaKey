import cv2
import cvzone
from custom_selfi_segmentation2 import CustomSelfiSegmentation
from picamera2 import Picamera2
import psutil
import time

# Picamera2 객체 생성
picam2 = Picamera2()

# 미리보기 모드 구성 설정
preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(preview_config)

# 카메라 시작
picam2.start()

segmentor = CustomSelfiSegmentation()

# 배경 이미지 로드
img_bg = cv2.imread('/home/embe/Final/final-project/1.jpg')
if img_bg.shape[2] == 4:  # Convert to 3-channel if the image has an alpha channel
    img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGRA2BGR)
img_bg = cv2.resize(img_bg, (640, 480))  # Resize to match the expected input size

# 임계값 초기값 설정
threshold = 0.7  # 임계값 조정 가능

# 사용자로부터 threshold 값 입력 받기
while True:
    try:
        threshold = float(input("임계값을 설정하세요 (0 ~ 1): "))
        if 0 <= threshold <= 1:
            break
        else:
            print("0과 1 사이의 값을 입력하세요.")
    except ValueError:
        print("숫자를 입력하세요.")

# FPS 계산 초기화
start_fps_time = time.time()
frame_count = 0
fps = 0.0

# 프로그램 시작
print("프로그램 시작")

while True:
    # CPU 및 메모리 사용량 업데이트
    cpu_usage = psutil.cpu_percent()
    mem_usage = psutil.virtual_memory().percent
    
    # 카메라로부터 이미지 캡처
    img = picam2.capture_array()
    img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환
    
    # 배경 제거 수행
    imgOut = segmentor.removeBG(img, img_bg, cutThreshold=threshold)

    # 이미지 스택
    imgStack = cvzone.stackImages([img, imgOut], 2, 1)

    # FPS 계산
    frame_count += 1
    elapsed_time = time.time() - start_fps_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        start_fps_time = time.time()
        frame_count = 0

    # CPU 및 메모리 사용량 표시
    cv2.putText(imgStack, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(imgStack, f"CPU Usage: {cpu_usage:.2f}%", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(imgStack, f"Memory Usage: {mem_usage:.2f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 이미지 출력
    cv2.imshow("image", imgStack)
    
    # 키 입력 처리
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# 종료 시 카메라 정리
picam2.stop()
cv2.destroyAllWindows()
