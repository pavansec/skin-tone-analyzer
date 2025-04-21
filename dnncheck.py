import cv2
img = cv2.imread("fairface/validation/image_0.jpg")
img = cv2.resize(img, (300, 300))
dnn_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
dnn_net.setInput(blob)
detections = dnn_net.forward()
print("Detections shape:", detections.shape)