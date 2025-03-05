from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from object_detector import *
import cv2
import numpy as np
import base64
import re
import os

app = Flask(__name__, static_folder='dist')
CORS(app)

@app.route('/api/upload', methods=['POST'])
def upload():
    data = request.json.get('image')
    if not data:
        return jsonify({'error': 'No image data found'}), 400
    
    match = re.search(r"data:image/\w+;base64,(.*)", data)
    if not match:
        return jsonify({'error': 'Invalid image data'}), 400

    image_data = base64.b64decode(match.group(1))
    npimg = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if corners:
        response = {}

        w_marker_real = 50.0
        h_marker_real = 50.0

        marker_corners = corners[0][0]
        aruco_rect = cv2.minAreaRect(corners[0])
        (aruco_x, aruco_y), (aruco_w, aruco_h), _ = aruco_rect
        aruco_position = (aruco_x, aruco_y, aruco_w, aruco_h)

        w_marker_image = np.linalg.norm(marker_corners[0] - marker_corners[1])
        h_marker_image = np.linalg.norm(marker_corners[0] - marker_corners[3])

        scale_width = w_marker_real / w_marker_image
        scale_height = h_marker_real / h_marker_image

        detector = HomogeneousBgDetector()
        object_contour = detector.detect_objects(image)

        for cnt in object_contour:
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            box = cv2.boxPoints(rect)
            box = np.intp(box)

            w_object_real = float(w * scale_width)
            h_object_real = float(h * scale_height)

            if aruco_position:
                aruco_x, aruco_y, aruco_w, aruco_h = aruco_position
                if (abs(x - aruco_x) < aruco_w / 2 and abs(y - aruco_y) < aruco_h / 2):
                    continue
                else:
                    w_object_real = round(w_object_real, 2)
                    h_object_real = round(h_object_real, 2)
                    temp = 0

                    if w_object_real > h_object_real:
                        temp = w_object_real
                        w_object_real = h_object_real
                        h_object_real = temp

                    response['width'] = w_object_real
                    response['height'] = h_object_real
                    response['classification'] = detector.classify(h_object_real)
                    cv2.polylines(image, [box], True, (0, 255, 0), 3)

        _, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        image = f"data:image/jpeg;base64,{img_str}"
        response['image'] = image
        return jsonify(response)
    else:
        return jsonify({'error': 'No markers found'}), 400

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and (
        path.endswith(".js") or 
        path.endswith(".css") or 
        path.endswith(".html") or 
        path.endswith(".png") or
        path.endswith(".ico") or
        path.endswith(".json") or
        path.endswith(".ttf") or 
        path.endswith(".eot") or 
        path.endswith(".woff2") or
        path.endswith(".woff") or
        path.endswith(".webmanifest")):
        return send_from_directory('dist', path)
    else:
        return send_from_directory('dist', 'index.html')
if __name__ == '__main__':
    app.run(debug=True)