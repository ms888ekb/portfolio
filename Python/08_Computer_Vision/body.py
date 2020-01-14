import recognize_liveness

if __name__ == "__main__":

    camera, predictor, known_imgs = recognize_liveness.init()

    if not camera.isOpened():
       print("Camera access error")

    data = recognize_liveness.process_and_encode(known_imgs)

    while camera.isOpened():

        slide = recognize_liveness.predict_and_show(camera, data, predictor)

        recognize_liveness.cv2.imshow('Liveness detection', slide)

        if recognize_liveness.cv2.waitKey(1) & 0xFF == ord('z'):
                break

    camera.release()

    recognize_liveness.cv2.destroyAllWindows()
