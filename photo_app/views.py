import base64

from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.http import JsonResponse
from django.http import StreamingHttpResponse
from .meiyan import *
from django.views.decorators.csrf import csrf_exempt
from djangoProject import settings
from . import models
from . import forms # 引入表单
from django.shortcuts import render, redirect
from .models import Photo
from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import os
import cv2
from django.views.decorators import gzip
import dlib
from django.shortcuts import render
from django.http import JsonResponse
from .models import Photo


# Create your views here.

def index(request):
    if not request.session.get('is_login', None):
        return redirect('/login/')
    photos = Photo.objects.all()
    photos_reversed = list(reversed(list(photos)))
    print(photos.last().image.url)
    # 修正了字典键，并直接传递字典而不是嵌套在另一个字典中
    context = {
        'photos': photos,  # 这里添加了冒号，并使用了正确的键名 'photos'
        'photos_reversed': photos_reversed,
        'A': "/media/imged.jpg"
    }
    return render(request, 'photo_app/index.html', context)


def login(request):
    if request.session.get('is_login', None):  # 不允许重复登录
        return redirect('/index/')
    if request.method == 'POST':
        login_form = forms.UserForm(request.POST)
        message = '请检查填写的内容！'
        if login_form.is_valid():
            username = login_form.cleaned_data.get('username')
            password = login_form.cleaned_data.get('password')

            try:
                user = models.User.objects.get(name=username)
            except :
                message = '用户不存在！'
                return render(request, 'photo_app/login.html', locals())

            if user.password == password:
                request.session['is_login'] = True
                request.session['user_id'] = user.id
                request.session['user_name'] = user.name
                request.session['user_email'] = user.email
                request.session['user_sex'] = user.sex
                return redirect('/index/')
            else:
                message = '密码不正确！'
                return render(request, 'photo_app/login.html', locals())
        else:
            return render(request, 'photo_app/login.html', locals())

    login_form = forms.UserForm()
    return render(request, 'photo_app/login.html', locals())
def register(request):
    if request.session.get('is_login', None):
        return redirect('/index/')

    if request.method == 'POST':
        register_form = forms.RegisterForm(request.POST)
        message = "请检查填写的内容！"
        if register_form.is_valid():
            username = register_form.cleaned_data.get('username')
            password1 = register_form.cleaned_data.get('password1')
            password2 = register_form.cleaned_data.get('password2')
            email = register_form.cleaned_data.get('email')
            sex = register_form.cleaned_data.get('sex')

            if password1 != password2:
                message = '两次输入的密码不同！'
                return render(request, 'photo_app/register.html', locals())
            else:
                same_name_user = models.User.objects.filter(name=username)
                if same_name_user:
                    message = '用户名已经存在'
                    return render(request, 'photo_app/register.html', locals())
                same_email_user = models.User.objects.filter(email=email)
                if same_email_user:
                    message = '该邮箱已经被注册了！'
                    return render(request, 'photo_app/register.html', locals())

                new_user = models.User()
                new_user.name = username
                new_user.password = password1
                new_user.email = email
                new_user.sex = sex
                new_user.save()

                return redirect('/login/')
        else:
            return render(request, 'photo_app/register.html', locals())
    register_form = forms.RegisterForm()
    return render(request, 'photo_app/register.html', locals())

def logout(request):
    if not request.session.get('is_login', None):
        # 如果本来就未登录，也就没有登出一说
        return redirect("/login/")
    request.session.flush()
    # 或者使用下面的方法
    # del request.session['is_login']
    # del request.session['user_id']
    # del request.session['user_name']
    return redirect("/login/")

def image_view(request):
    all_photos = Photo.objects.all()
    return render(request, 'photo_app/image.html',{'all_photos':all_photos})


def profile_settings(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        sex = request.POST.get('sex')

        user_id = request.session.get('user_id')
        if not user_id:
            return redirect('login')  # 如果没有用户ID，重定向到登录页面

        user = models.User.objects.get(id=user_id)
        user.name = username
        user.email = email
        user.sex = sex
        user.save()

        # 更新session中的信息
        request.session['user_name'] = user.name
        request.session['user_email'] = user.email
        request.session['user_sex'] = user.sex

        return render(request,'photo_app/profile_setting.html')  # 重定向到用户资料页面或其他页面

    return render(request, 'photo_app/profile_setting.html')


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def face_detection(photo):
    # 读取图像
    processed_image = np.array(Image.open(photo.image))

    # 确保图像是RGB格式
    if len(processed_image.shape) == 2 or processed_image.shape[2] == 1:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    elif processed_image.shape[2] == 4:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGBA2RGB)

    # 将图像转换为灰度图像进行人脸检测
    gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)

    # 使用预训练的人脸检测模型
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 绘制矩形到检测到的人脸上
    for (x, y, w, h) in faces:
        cv2.rectangle(processed_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 将numpy数组转换为Pillow的Image对象
    processed_image_pil = Image.fromarray(processed_image)

    return processed_image_pil


def image_recognition(request):
    photos_image_directory = os.path.join(settings.MEDIA_ROOT, 'photos')
    processed_image_directory = os.path.join(settings.MEDIA_ROOT, 'processed')

    ensure_directory(photos_image_directory)
    ensure_directory(processed_image_directory)

    photos_image_storage = FileSystemStorage(location=photos_image_directory, base_url=settings.MEDIA_URL + 'photos/')
    processed_image_storage = FileSystemStorage(location=processed_image_directory,
                                                base_url=settings.MEDIA_URL + 'processed/')

    if request.method == 'POST' and request.FILES.get('image'):
        title = request.POST.get('username')
        image = request.FILES['image']

        filename = photos_image_storage.save(image.name, image)
        uploaded_file_url = photos_image_storage.url(filename)

        # 保存到数据库
        photo = Photo(title=title, image=os.path.join('photos', filename))
        photo.save()

        return render(request, 'photo_app/image_recognition.html', {
            'uploaded_file_url': uploaded_file_url
        })

    if request.method == 'POST' and request.POST.get('uploaded_file_url'):
        uploaded_file_url = request.POST.get('uploaded_file_url')
        filename = os.path.basename(uploaded_file_url)

        original_image_path = os.path.join(photos_image_storage.location, filename)

        # 人脸检测处理
        photo = Photo.objects.get(image=os.path.join('photos', filename))
        processed_image = face_detection(photo)

        processed_filename = 'processed_' + filename
        processed_image_path = os.path.join(processed_image_storage.location, processed_filename)

        # 保存处理后的图像，确保使用相同的格式
        processed_image.save(processed_image_path)

        processed_file_url = processed_image_storage.url(processed_filename)

        return JsonResponse({'processed_file_url': processed_file_url})

    return render(request, 'photo_app/image_recognition.html')

def video_face_detection(request):
    return render(request, 'photo_app/video_face_detection.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 绘制矩形到检测到的人脸上
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 将帧转换为字节流
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# 使用gzip压缩视频流，提高传输速度
@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type="multipart/x-mixed-replace;boundary=frame")




def digital_makeup(request):
    photos_image_directory = os.path.join(settings.MEDIA_ROOT, 'photos')
    processed_image_directory = os.path.join(settings.MEDIA_ROOT, 'processed')

    ensure_directory(photos_image_directory)
    ensure_directory(processed_image_directory)

    photos_image_storage = FileSystemStorage(location=photos_image_directory, base_url=settings.MEDIA_URL + 'photos/')
    processed_image_storage = FileSystemStorage(location=processed_image_directory, base_url=settings.MEDIA_URL + 'processed/')

    if request.method == 'POST' and request.FILES.get('image'):
        title = request.POST.get('username')
        image = request.FILES['image']

        filename = photos_image_storage.save(image.name, image)
        uploaded_file_url = photos_image_storage.url(filename)

        # 保存到数据库
        photo = Photo(title=title, image=os.path.join('photos', filename))
        photo.save()

        return render(request, 'photo_app/digital_makeup.html', {
            'uploaded_file_url': uploaded_file_url
        })

    if request.method == 'POST' and request.POST.get('uploaded_file_url'):
        uploaded_file_url = request.POST.get('uploaded_file_url')
        filename = os.path.basename(uploaded_file_url)

        original_image_path = os.path.join(photos_image_storage.location, filename)

        # 数字化妆处理
        photo = Photo.objects.get(image=os.path.join('photos', filename))
        processed_image = makeup(photo)

        processed_filename = 'processed_' + filename
        processed_image_path = os.path.join(processed_image_storage.location, processed_filename)
        processed_image.save(processed_image_path)
        processed_file_url = processed_image_storage.url(processed_filename)

        return JsonResponse({'processed_file_url': processed_file_url})

    return render(request, 'photo_app/digital_makeup.html')
def makeup(photo):
    # 加载图片到numpy array
    image = face_recognition.load_image_file(photo.image.path)

    # 标识脸部特征
    face_landmarks_list = face_recognition.face_landmarks(image)

    for face_landmarks in face_landmarks_list:
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image, 'RGBA')

        # 绘制眉毛
        d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

        # 绘制嘴唇
        d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

        # 绘制眼睛
        d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

        # 绘制眼线
        d.line(
            face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]],
            fill=(0, 0, 0, 110),
            width=6)
        d.line(
            face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]],
            fill=(0, 0, 0, 110),
            width=6)
    return pil_image

def funtion1(request):  #一键美颜
    photos_image_directory = os.path.join(settings.MEDIA_ROOT, 'photos')
    processed_image_directory = os.path.join(settings.MEDIA_ROOT, 'processed')

    ensure_directory(photos_image_directory)
    ensure_directory(processed_image_directory)

    photos_image_storage = FileSystemStorage(location=photos_image_directory, base_url=settings.MEDIA_URL + 'photos/')
    processed_image_storage = FileSystemStorage(location=processed_image_directory,
                                                base_url=settings.MEDIA_URL + 'processed/')

    if request.method == 'POST' and request.FILES.get('image'):
        title = request.POST.get('username')
        image = request.FILES['image']
        filename = photos_image_storage.save(image.name, image)
        uploaded_file_url = photos_image_storage.url(filename)
        photo = Photo(title=title, image=os.path.join('photos', filename))
        photo.save()

        return render(request, 'photo_app/funtion1.html', {
            'uploaded_file_url': uploaded_file_url
        })

    if request.method == 'POST' and request.POST.get('uploaded_file_url'):
        uploaded_file_url = request.POST.get('uploaded_file_url')
        filename = os.path.basename(uploaded_file_url)
        original_image_path = os.path.join(photos_image_storage.location, filename)
        photo = Photo.objects.get(image=os.path.join('photos', filename))
        processed_image = meiyan(photo)
        processed_filename = 'processed_' + filename
        processed_image_path = os.path.join(processed_image_storage.location, processed_filename)
        processed_image.save(processed_image_path)
        processed_file_url = processed_image_storage.url(processed_filename)
        return JsonResponse({'processed_file_url': processed_file_url})
    return render(request, 'photo_app/funtion1.html')

def funtion2(request):  #裁剪
    return render(request, 'photo_app/funtion2.html')


def funtion3(request):
    photos_image_directory = os.path.join(settings.MEDIA_ROOT, 'photos')
    processed_image_directory = os.path.join(settings.MEDIA_ROOT, 'processed')

    ensure_directory(photos_image_directory)
    ensure_directory(processed_image_directory)

    photos_image_storage = FileSystemStorage(location=photos_image_directory, base_url=settings.MEDIA_URL + 'photos/')
    processed_image_storage = FileSystemStorage(location=processed_image_directory, base_url=settings.MEDIA_URL + 'processed/')

    if request.method == 'POST' and request.FILES.get('image'):
        title = request.POST.get('username')
        image = request.FILES['image']
        filename = photos_image_storage.save(image.name, image)
        uploaded_file_url = photos_image_storage.url(filename)
        photo = Photo(title=title, image=os.path.join('photos', filename))
        photo.save()

        return render(request, 'photo_app/funtion3.html', {
            'uploaded_file_url': uploaded_file_url
        })

    if request.method == 'POST' and request.POST.get('uploaded_file_url'):
        uploaded_file_url = request.POST.get('uploaded_file_url')
        filename = os.path.basename(uploaded_file_url)
        original_image_path = os.path.join(photos_image_storage.location, filename)
        photo = Photo.objects.get(image=os.path.join('photos', filename))
        processed_image = face_landmark(photo)
        processed_filename = 'processed_' + filename
        processed_image_path = os.path.join(processed_image_storage.location, processed_filename)
        processed_image.save(processed_image_path)
        processed_file_url = processed_image_storage.url(processed_filename)
        return JsonResponse({'processed_file_url': processed_file_url})
    return render(request, 'photo_app/funtion3.html')

def face_landmark(photo):
    # 加载图像
    img = face_recognition.load_image_file(photo.image.path)

    # 确保图像是RGB格式
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # 将图像转换为灰度图像进行人脸检测
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 使用dlib的人脸检测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("photo_app/static/face_landmark_dilb/shape_predictor_68_face_landmarks.dat")

    # 检测人脸
    dets = detector(gray, 1)
    for face in dets:
        shape = predictor(img, face)  # 找到人脸的68个特征点
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 1, (0, 255, 0), 2)  # 在特征点上绘制圆圈

    # 将numpy数组转换为Pillow的Image对象
    pil_image = Image.fromarray(img)

    return pil_image

def contour_identification(request):
    photos_image_directory = os.path.join(settings.MEDIA_ROOT, 'photos')
    processed_image_directory = os.path.join(settings.MEDIA_ROOT, 'processed')

    ensure_directory(photos_image_directory)
    ensure_directory(processed_image_directory)

    photos_image_storage = FileSystemStorage(location=photos_image_directory, base_url=settings.MEDIA_URL + 'photos/')
    processed_image_storage = FileSystemStorage(location=processed_image_directory, base_url=settings.MEDIA_URL + 'processed/')

    if request.method == 'POST' and request.FILES.get('image'):
        title = request.POST.get('username')
        image = request.FILES['image']

        filename = photos_image_storage.save(image.name, image)
        uploaded_file_url = photos_image_storage.url(filename)

        # Save to database
        photo = Photo(title=title, image=os.path.join('photos', filename))
        photo.save()

        return render(request, 'photo_app/contour_identification.html', {
            'uploaded_file_url': uploaded_file_url
        })

    if request.method == 'POST' and request.POST.get('uploaded_file_url'):
        uploaded_file_url = request.POST.get('uploaded_file_url')
        filename = os.path.basename(uploaded_file_url)

        original_image_path = os.path.join(photos_image_storage.location, filename)

        # Face outline recognition processing
        photo = Photo.objects.get(image=os.path.join('photos', filename))
        processed_image = outline(photo)

        processed_filename = 'processed_' + filename
        processed_image_path = os.path.join(processed_image_storage.location, processed_filename)
        processed_image.save(processed_image_path)
        processed_file_url = processed_image_storage.url(processed_filename)

        return JsonResponse({'processed_file_url': processed_file_url})

    return render(request, 'photo_app/contour_identification.html')

def outline(photo):
    image = face_recognition.load_image_file(photo.image.path)

    # Find all facial features in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    for face_landmarks in face_landmarks_list:
        facial_features = [
            'chin',  # Jawline
            'left_eyebrow',  # Left eyebrow
            'right_eyebrow',  # Right eyebrow
            'nose_bridge',  # Nose bridge
            'nose_tip',  # Nose tip
            'left_eye',  # Left eye
            'right_eye',  # Right eye
            'top_lip',  # Upper lip
            'bottom_lip'  # Lower lip
        ]
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image)
        for facial_feature in facial_features:
            d.line(face_landmarks[facial_feature], fill=(0, 255, 255), width=2)
        return pil_image


def search_photos(request):
    query = request.GET.get('query', '')
    if query:
        results = Photo.objects.filter(title__icontains=query)
    else:
        results = Photo.objects.all()

    context = {
        'photos': results,
        'query': query,
    }

    return render(request, 'photo_app/search_results.html', context)