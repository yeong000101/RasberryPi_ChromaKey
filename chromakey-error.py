#!/usr/bin/python3

import time
import cv2
import numpy as np
import psutil
from picamera2 import Picamera2
import os

# 현재 실행 중인 파일 이름을 기반으로 녹화 파일명 생성
current_filename = os.path.splitext(os.path.basename(__file__))[0]
video_filename = current_filename + ".mp4"

# Picamera2 객체 생성
picam2 = Picamera2()

# 미리보기 모드 구성 설정
preview_config = picam2.create_preview_configuration(main={"size": (800, 600)})
picam2.configure(preview_config)

# 카메라 시작
picam2.start()

# 비디오 녹화 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_filename, fourcc, 30.0, (800, 600))

# 배경 이미지 로드
back = cv2.imread('/home/embe/Final/final-project/1.jpg')
if back is None:
    raise ValueError("Background image not found at specified path.")

# 모폴로지 연산을 위한 커널 생성
kernel = np.ones((5, 5), np.uint8)

# 측정 상태 변수 초기화
start_fps_time = time.time()
frame_count = 0
fps = 0.0
recording_start_time = time.time()

while True:
    # 카메라에서 이미지 캡처
    im = picam2.capture_array()
    if im is None:
        print("Failed to capture image from camera.")
        continue

    # 색상 채널 변환 (RGB -> BGR)
    sq_im = cv2.cvtColor(im[:, :, :3], cv2.COLOR_RGB2BGR)

    # HSV 색 공간으로 변환
    hsv = cv2.cvtColor(sq_im, cv2.COLOR_BGR2HSV)

    # 초록색 영역 마스킹
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 모폴로지 연산 적용 (팽창 및 침식)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 가우시안 블러 적용
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask_inv = cv2.GaussianBlur(cv2.bitwise_not(mask), (15, 15), 0)

    # 배경 이미지 크기 조정
    back_resized = cv2.resize(back, (im.shape[1], im.shape[0]))

    # 알파 블렌딩 적용
    alpha = 0.5
    blended = cv2.addWeighted(sq_im, alpha, back_resized, 1 - alpha, 0)

    # 초록색 영역을 배경으로 대체
    fg = cv2.bitwise_and(sq_im, sq_im, mask=mask_inv)
    bg = cv2.bitwise_and(back_resized, back_resized, mask=mask)
    final_frame = cv2.add(fg, bg)

    # FPS 계산
    frame_count += 1
    elapsed_time = time.time() - start_fps_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        start_fps_time = time.time()
        frame_count = 0

    # 메모리 및 CPU 사용량 표시
    cpu_usage = psutil.cpu_percent()
    mem_usage = psutil.virtual_memory().percent
    text1 = f"FPS: {fps:.2f} | CPU Usage: {cpu_usage:.2f}%"
    text2 = f"Memory Usage: {mem_usage:.2f}%"

    # 텍스트를 이미지에 표시
    cv2.putText(final_frame, text1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(final_frame, text2, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 합성된 이미지를 화면에 표시
    cv2.imshow("Camera", final_frame)

    # 녹화
    out.write(final_frame)

    # 키 입력 감지
    key = cv2.waitKey(1) & 0xFF

    # ESC 키 입력 또는 20초 후 루프 탈출
    if key == 27 or (time.time() - recording_start_time) > 20:
        break

# 창 닫기 및 카메라 정지
cv2.destroyAllWindows()
picam2.stop()
out.release()
