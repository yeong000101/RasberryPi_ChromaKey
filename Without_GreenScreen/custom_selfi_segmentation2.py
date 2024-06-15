import cv2
import mediapipe as mp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class CustomSelfiSegmentation():

        def __init__(self, model=1):
                """
                :param model: model type 0 or 1. 0 is general 1 is landscape(faster)
                """
                self.model = model
                self.mpDraw = mp.solutions.drawing_utils
                self.mpSelfieSegmentation = mp.solutions.selfie_segmentation
                self.selfieSegmentation = self.mpSelfieSegmentation.SelfieSegmentation(model_selection=self.model)

        def _process_row(self, mask_row, img_row, imgBg, cutThreshold):
            """
            Process a single row of the image in parallel.
            """
            condition = mask_row > cutThreshold
            if isinstance(imgBg, tuple):
                _imgBg_row = np.zeros(img_row.shape, dtype=np.uint8)
                _imgBg_row[:] = imgBg
                return np.where(condition[..., None], img_row, _imgBg_row)
            else:
                return np.where(condition[..., None], img_row, imgBg)

        def removeBG(self, img, imgBg=(255, 255, 255), cutThreshold=0.1):
                """
                :param img: image to remove background from
                :param imgBg: Background Image. can be a color (255,0,255) or an image . must be same size
                :param cutThreshold: higher = more cut, lower = less cut
                :return:
                """
                 # 이미지 크기를 줄여서 처리 속도를 높임 (원본의 70%)
                scale_factor = 0.7
                img_small = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        
                # 이미지를 RGB 형식으로 변환
                imgRGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

                # SelfiSegmentation을 사용하여 이미지 처리
                results = self.selfieSegmentation.process(imgRGB)

                # 세그멘테이션 마스크에 가우시안 블러 적용
                blurred_mask = cv2.GaussianBlur(results.segmentation_mask, (5, 5), 0)

                # 마스크를 원래 크기로 다시 확장 (고품질 업샘플링 사용)
                blurred_mask = cv2.resize(blurred_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                
                 # 1.Create condition for background removal based on the blurred mask and cut threshold
                condition = blurred_mask > cutThreshold

                # Replace the background with imgBg
                imgOut = np.where(condition[..., None], img, imgBg)
                
                return imgOut
                
                '''
                # 2.Create condition for background removal based on the blurred mask and cut threshold
                condition = np.stack(
                    (blurred_mask,) * 3, axis=-1) > cutThreshold

                # Replace the background with imgBg
                if isinstance(imgBg, tuple):
                    _imgBg = np.zeros(img.shape, dtype=np.uint8)
                    _imgBg[:] = imgBg
                    imgOut = np.where(condition, img, _imgBg)
                else:
                    imgOut = np.where(condition, img, imgBg)
                return imgOut 
                
                 # 3.병렬 처리를 위해 ThreadPoolExecutor를 사용하여 처리
                with ThreadPoolExecutor() as executor:
                    # 이미지의 각 픽셀에 대해 배경 제거 조건을 계산합니다.
                    futures = []
                    for i in range(img.shape[0]):
                        future = executor.submit(self._process_row, blurred_mask[i], img[i], imgBg, cutThreshold)
                        futures.append(future)

                    imgOut_rows = [future.result() for future in as_completed(futures)]

                imgOut = np.stack(imgOut_rows)

                return imgOut
                '''

        
        def removeBG2(self, img, imgBg=(255, 255, 255), cutThreshold=0.1):
                """
                :param img: 배경을 제거할 이미지
                :param imgBg: 배경 이미지. (255,0,255)와 같은 색상 또는 이미지일 수 있습니다. 크기는 동일해야 합니다.
                :param cutThreshold: 높을수록 더 많이 자르고, 낮을수록 적게 자릅니다.
                :return: 배경이 제거된 이미지
                """
                # 이미지를 RGB 형식으로 변환
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # SelfiSegmentation을 사용하여 이미지 처리
                results = self.selfieSegmentation.process(imgRGB)

                # 세그멘테이션 마스크에 모폴로지 연산을 적용하여 노이즈 제거
                kernel = np.ones((5, 5), np.uint8)
                opening = cv2.morphologyEx(results.segmentation_mask, cv2.MORPH_OPEN, kernel)

                # 가우시안 블러 적용
                blurred_mask = cv2.GaussianBlur(opening, (7, 7), 0)
                
                cv2.imshow("blurred_mask", blurred_mask)

                # 모폴로지가 적용된 마스크와 잘라내기 임계값을 이용하여 배경 제거 조건 생성
                condition = blurred_mask > cutThreshold

                # 조건을 원본 이미지에 적용
                img_result = np.where(condition[..., None], img, imgBg)

                return img_result

def main():
        # Initialize the webcam. '2' indicates the third camera connected to the computer.
        # '0' usually refers to the built-in camera.
        cap = cv2.VideoCapture(0)

        # Set the frame width to 640 pixels
        cap.set(3, 640)
        # Set the frame height to 480 pixels
        cap.set(4, 480)

        # Initialize the CustomSelfiSegmentation class.
        segmentor = CustomSelfiSegmentation(model=0)

        # Infinite loop to keep capturing frames from the webcam
        while True:
        # Capture a single frame
                success, img = cap.read()

                # Use the CustomSelfiSegmentation class to remove the background
                # Replace it with a magenta background (255, 0, 255)
                # imgBG can be a color or an image as well. must be same size as the original if image
                # 'cutThreshold' is the sensitivity of the segmentation.
                imgOut = segmentor.removeBG(img, imgBg=(255, 0, 255), cutThreshold=0.1)

                # Stack the original image and the image with background removed side by side
                imgStacked = cv2.hconcat([img, imgOut])

                # Display the stacked images
                # cv2.imshow("Image", imgStacked)

                # Check for 'q' key press to break the loop and close the window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == "__main__":
    main()
