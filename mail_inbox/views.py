# from msilib.schema import File
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from django.http import JsonResponse
from django.shortcuts import render
from django.views.generic import TemplateView
from django.core.files import File
from django.core.files.base import ContentFile
from django.core.files.temp import NamedTemporaryFile
from urllib.request import urlopen
from django.core.files.storage import default_storage
import cv2
import numpy as np
import json
import base64
import os
from datetime import datetime


import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from django.core.files.storage import default_storage

def mapp(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


# Create your views here.
class MailInbox(TemplateView):
    template_name = "test.html"

    # def __init__(self, **kwargs):
    #     super().__init__(kwargs)
    #     self.FILES = None

#     def camera(self,request, args, *kwargs):
#         if request.method == 'POST':
#             image_path = request.POST["src"]
#             image = NamedTemporaryFile()
#             urlopen(image_path).read()
#             image.write(urlopen(image_path).read())
#             image.flush()
#             image = File(image)
#             name = str(image.name).split('\\')[-1]
#             name += 'input_image.jpg'  # store image in jpeg format
#             image.name = name
#             with open('image.txt', 'w+') as file:
#                 file.write(str(name))
#             default_storage.save('media/a.jpg',
#                                  ContentFile(urlopen(image_path).read()))
#             # return HttpResponse('Done!')
#         return render(request,  self.template_name)
#
#     import numpy as np
#
    def get(self, request, *args, **kwargs):
        return render(request, self.template_name)
#
#     def scan_image_api(request):
#         image_path = r"C:\Users\ankit\PycharmProjects\MailBox\CamScanner-In-Python\input.jpg"
#         image = cv2.imread(image_path)
#         # Rest of your code...
#
#         # Modify the code as needed to return the desired response
#         return JsonResponse({'message': 'API response'})


def process_image(request):
    if request.method == 'POST':
        print(request.GET)
        photo = request.POST['photo']
        print("11111111",photo.split('base64,')[1])
        file_name = str(datetime.now()).replace(' ','-').replace(':','-')
        with open("static/input_image/input_"+file_name+".png", "wb") as f:
            f.write(base64.b64decode(photo.split('base64,')[1]))#photo.split('base64,')[1]
        image_path = "static/input_image/input_"+file_name+".png"
        # image_path = "static/input_image/1689262050027.JPEG"
        image = cv2.imread(image_path)  # Read in the image
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        # image = cv2.imread(r'C:\Users\ankit\PycharmProjects\MailBox\mail\input_image\input.jpg')  # read in the image
        image = cv2.resize(image, (1300, 800))  # resizing because opencv does not work well with bigger images
        orig = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB To Gray Scale
        blurred = cv2.GaussianBlur(gray, (5, 5),0)  # (5,5) is the kernel size and 0 is sigma that determines the amount of blur
        # cv2.imshow("Blur",blurred)

        edged = cv2.Canny(blurred, 30, 50)  # 30 MinThreshold and 50 is the MaxThreshold
        # cv2.imshow("Canny",edged)

        contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  # retrieve the contours as a list, with simple apprximation model
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # the loop extracts the boundary contours of the page
        for c in contours:
            p = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * p, True)

            if len(approx) == 4:
                target = approx
                break
        approx = mapp(target)  # find endpoints of the sheet

        pts = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])  # map to 800*800 target window

        op = cv2.getPerspectiveTransform(approx, pts)  # get the top or bird eye view effect
        dst = cv2.warpPerspective(orig, op, (800, 800))
        output = "static/output_images/output_"+file_name+".png"
        cv2.imwrite(output, dst)
        return JsonResponse({'url': output})



def order_points(pts):
    '''Rearrange coordinates to order:
       top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect.astype('int').tolist()
from django.utils.crypto import get_random_string

def implemnetation2(request):
    if request.method == 'POST':
        # print(request.GET)
        # photo = request.POST['photo']
        # print("11111111",photo.split('base64,')[1])
        # file_name = str(datetime.now()).replace(' ','-').replace(':','-')
        # with open("static/input_image/input_"+file_name+".JPEG", "wb") as f:
        #     f.write(base64.b64decode(photo.split('base64,')[1]))#photo.split('base64,')[1]
        image=request.FILES['photo']
        print(image)
        random_string = get_random_string(length=10)
        file_path = f"static/input_image/input_{random_string}.JPEG"
        image_path = default_storage.save(file_path, image)

        # image_path = default_storage.save(f"static/input_image/input_{random_string}.JPEG", image)


        # image_path = "static/input_image/input_"+image+".JPEG"
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to load the image")
        dim_limit = 1080
        max_dim = max(img.shape)
        if max_dim > dim_limit:
            resize_scale = dim_limit / max_dim
            img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

        # Making Copy of original image.
        orig_img = img.copy()
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        # Edge Detection.
        canny = cv2.Canny(gray, 100, 200)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        con = np.zeros_like(img)
        # Finding contours for the detected edges.
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Keeping only the largest detected contour.
        page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)
        # Blank canvas.
        con = np.zeros_like(img)
        # Loop over the contours.
        for c in page:
            # Approximate the contour.
            epsilon = 0.02 * cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, epsilon, True)
            # If our approximated contour has four points
            if len(corners) == 4:
                break
        cv2.drawContours(con, c, -1, (0, 255, 255), 3)
        cv2.drawContours(con, corners, -1, (0, 255, 0), 10)
        # Sorting the corners and converting them to desired shape.
        corners = sorted(np.concatenate(corners).tolist())

        # Displaying the corners.
        for index, c in enumerate(corners):
            character = chr(65 + index)
            cv2.putText(con, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 5, cv2.LINE_AA)

        # Rearranging the order of the corner points.
        corners = order_points(corners)

        print(corners)

        # plt.figure(figsize=(10, 7))
        # plt.imshow(con)
        # plt.title('Corner Points')
        # plt.show()
        (tl, tr, br, bl) = corners
        # Finding the maximum width.
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Finding the maximum height.
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # Final destination co-ordinates.
        destination_corners = [
            [0, 0],
            [maxWidth, 0],
            [maxWidth, maxHeight],
            [0, maxHeight]]
        print(destination_corners)
        homography = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
        # Perspective transform using homography.
        final = cv2.warpPerspective(orig_img, np.float32(homography), (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

        output = "static/output_images/output_"+random_string+".JPEG"
        cv2.imwrite(output, final)
        return JsonResponse({'url': output})