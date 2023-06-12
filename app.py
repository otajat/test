from flask import Flask, request, render_template , redirect
from PIL import Image
import cv2
import numpy as np


app = Flask(__name__)

def lap(gray):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    my_image = np.zeros_like(gray, dtype=np.float32)
    for i in range(1,my_image.shape[0]-1):
        for j in range(1,my_image.shape[1]-1):
            my_image[i,j] = gray[i+1,j] - 2*gray[i,j] + gray[i-1,j]  
    d2x = my_image
    my_image = np.zeros_like(gray, dtype=np.float32)
    for i in range(0,my_image.shape[0]-1):
        for j in range(0,my_image.shape[1]-1):
            my_image[i,j] = gray[i,j+1] - 2*gray[i,j]  + gray[i,j-1]
    d2y = my_image
    lap = np.abs(d2x) + np.abs(d2y)
    lap = cv2.normalize( lap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return lap



def shadow_corroo( dx, D, cfl=0.25, max_iter=100):
    # Apply the heat equation using the explicit Euler method
    img=cv2.imread('uploaded_image.png', -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    delta_t = cfl * (dx ** 2) / (4 * D)
    v0 = np.log(gray+0.6)
    
    for i in range(max_iter):
        v = v0 + delta_t * np.maximum(np.zeros_like(gray),lap(v0))
        diff = np.abs(v - v0)
        v0 = v
        print(f"Iteration {i}, max difference = {np.max(diff)}")
        
        # check if the solution has converged
        if np.max(diff) < 1e-2:
            break
            
        # adjust delta_t if necessary
        if np.max(diff) > 1e-2:
            delta_t /= 2
    return np.exp(np.log(gray+0.25) - v)
        

# def shadow_removel():
#     img = cv2.imread('uploaded_image.png', -1)
#     rgb_planes = cv2.split(img)

#     result_planes = []
#     result_norm_planes = []
#     for plane in rgb_planes:
#         dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
#         bg_img = cv2.medianBlur(dilated_img, 21)
#         diff_img = 255 - cv2.absdiff(plane, bg_img)
#         norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#         result_planes.append(diff_img)
#         result_norm_planes.append(norm_img)

#     result = cv2.merge(result_planes)
#     result_norm = cv2.merge(result_norm_planes)
#     return result_norm





@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file.stream) 
        img.save("static/images/" + file.filename)
        img.save('uploaded_image.png')
        # return 'Image successfully uploaded!'
        return redirect('/images/'+file.filename)
    return render_template('upload.html')

@app.route('/images/<filename>')
def display_image(filename):
    rr=shadow_corroo(1,1)
    img_uint8 = cv2.normalize(rr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite('static/images/test.png',img_uint8)



    # img = Image.fromarray(rr)
    # img = img.convert("RGB")
    # img.save('static/images/edited.png')
    return render_template('image.html', filename='test.png')

    # img = cv2.imread('image3.jpg')
    # image_color = shadow_corroo(1,1)
    # My_image = cv2.imread(filename+".png")
    rr=shadow_corroo(1,1)
    img = Image.fromarray(rr)
    img = img.convert("RGB")

    img.save('static/images/edited.png')
    return render_template('image.html', filename='edited.png')



if __name__ == '__main__':
    app.run(debug=True)


