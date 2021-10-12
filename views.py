from django.shortcuts import render
from django.shortcuts import render
import os
import requests

def button(request):
    os.system('python django_ML_API/facenet/align/align_dataset_yolo_gpu.py')
    return render(request,"home.html")

