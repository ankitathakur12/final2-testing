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
    template_name = "index.html"

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
        photo = json.loads(request.POST['photo'])
        print("11111111",photo.split('base64,')[1])
        with open("static/input_image/input1.png", "wb") as f:
            f.write(base64.b64decode(photo.split('base64,')[1]))#photo.split('base64,')[1]
        image_path = "input_image/input.jpg"
        image = cv2.imread(image_path)  # Read in the image
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        # image = cv2.imread(r'C:\Users\ankit\PycharmProjects\MailBox\mail\input_image\input.jpg')  # read in the image
        image = cv2.resize(image, (1300, 800))  # resizing because opencv does not work well with bigger images
        orig = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB To Gray Scale
        blurred = cv2.GaussianBlur(gray, (5, 5),
                                   0)  # (5,5) is the kernel size and 0 is sigma that determines the amount of blur
        # cv2.imshow("Blur",blurred)

        edged = cv2.Canny(blurred, 30, 50)  # 30 MinThreshold and 50 is the MaxThreshold
        # cv2.imshow("Canny",edged)

        contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)  # retrieve the contours as a list, with simple apprximation model
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

        cv2.imwrite("static/output_images/output_image1.jpg", dst)
        return JsonResponse({'url': 'static/output_images/output_image1.jpg'})