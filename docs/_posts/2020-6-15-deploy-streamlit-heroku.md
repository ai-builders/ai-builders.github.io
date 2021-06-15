---
layout: post
title: Deploy โมเดลบน Heroku ด้วย streamlit และ fastai
---

เมื่อสร้างโมเดลด้วย fastai (หรือ framework อื่นๆ) เสร็จเรียบร้อยแล้ว เราสามารถ deploy โมเดลของเราเป็น webapp เพื่อง่ายต่อการใช้งานด้วย `streamlit` บน `heroku` ตามขั้นตอนดังต่อไปนี้

## สิ่งที่ต้องเตรียม

- ลง `streamlit`

```
pip install streamlit
```

- ไฟล์ `.pkl` ที่ถูก export ด้วย `learner`; ถ้าขนาดไม่เกิน 100MB สามารถเก็บไว้ใน git repository ได้ หากใหญ่กว่านั้นสามารถใส่ไว้ใน storage อื่น เช่น Google Drive

```
learn.export('yourmodelname.pkl')
```

- สมัครบัญชี [heroku](https://signup.heroku.com/)

## ขั้นตอนการสร้าง   webapp ด้วย streamlit แบบ local

- สร้าง branch สำหรับ webapp; ในที่นี้คือชื่อ `your_webapp_branch`

```
git checkout -b your_webapp_branch
```

- สร้างไฟล์ `app.py` 

```
#import library ที่ต้องใช้ทั้งหมด
from fastai.vision.all import (
    load_learner,
    PILImage,
)
import glob
from random import shuffle
import urllib.request

#import streamlit มาในชื่อ st เพื่อใช้ในการสร้าง user interface
import streamlit as st

# โหลดโมเดลจากแหล่งข้อมูลในอินเตอร์เน็ตเพื่อประหยัดพื้นที่เวลา deploy บน heroku
MODEL_URL = "https://github.com/cstorm125/choco-raisin/raw/main/notebooks/models/resnet34_finetune1e3_5p.pkl"
urllib.request.urlretrieve(MODEL_URL, "model.pkl")
learn_inf = load_learner('model.pkl', cpu=True)

# เราจะแบ่งหน้าจอเป็น 
# 1. sidebar ประกอบด้วยตัวเลือกรูปภาพ
# 2. main page ประกอบด้วยรูปและคำทำนาย

##################################
# sidebar
##################################

# ใส่ title ของ sidebar
st.sidebar.write('### Enter cookie to classify')

# radio button สำหรับเลือกว่าจะทำนายรูปจาก validation set หรือ upload รูปเอง
option = st.sidebar.radio('', ['Use a validation image', 'Use your own image'])
# โหลดรูปจาก validation set แล้ว shuffle
valid_images = glob.glob('images/valid/*/*')
shuffle(valid_images)

if option == 'Use a validation image':
    st.sidebar.write('### Select a validation image')
    fname = st.sidebar.selectbox('',
                                 valid_images)

else:
    st.sidebar.write('### Select an image to upload')
    fname = st.sidebar.file_uploader('',
                                     type=['png', 'jpg', 'jpeg'],
                                     accept_multiple_files=False)
    if fname is None:
        fname = valid_images[0]

##################################
# main page
##################################

# ใส่ title ของ main page
st.title("Chocolate Chip vs Raisin Cookies")

#function การทำนาย
def predict(img, learn):

    # ทำนายจากโมเดลที่ให้
    pred, pred_idx, pred_prob = learn.predict(img)

    # โชว์ผลการทำนาย
    st.success(f"This is {pred} cookie with the probability of {pred_prob[pred_idx]*100:.02f}%")
    
    # โชว์รูปที่ถูกทำนาย
    st.image(img, use_column_width=True)

# เปิดรูป
img = PILImage.create(fname)

# เรียก function ทำนาย
predict(img, learn_inf)
```

- ลอง run `app.py` ที่สร้างขึ้นบนเครื่องตัวเอง

```
streamlit run app.py
```

- รอ webapp ดาวน์โหลดโมเดลเรียบร้อยแล้ว เราก็จะได้ webapp ที่รันบนเครื่องของเราเองแบบนี้

<figure align="center">
  <img src="{{ site.baseurl }}/images/streamlit_demo.JPG" style="width: 400px;"/>
  <figcaption>ตัวอย่าง webapp ด้วย streamlit</figcaption>
</figure>

## ขั้นตอนเพิ่มเติมเพื่อ   deploy webapp บน heroku

- เนื่องจาก heroku มีข้อจำกัดว่าเราสามารถ deploy github repository ที่ใหญ่ไม่เกิน 500MB เท่านั้น เราจึงควรลบไฟล์ที่ไม่จำเป็นต้องใช้ออกจาก repository ของเราใน branch `your_webapp_branch`ก่อน เช่น ไฟล์โมเดล `.pkl` ให้ไปเก็บไว้ใน branch อื่น (`main`) หรือใน Google Drive

- ใน `branch` your_webapp_branch ควรมีเพียง
	- `app.py` ที่เราสร้างในขั้นตอนที่แล้ว
	- `requirements.txt`
	- `Procfile`
	- (optional) `runtime.txt`

- `requirements.txt` ประกอบด้วย

```
https://download.pytorch.org/whl/cpu/torch-1.8.1%2Bcpu-cp39-cp39-linux_x86_64.whl
https://download.pytorch.org/whl/cpu/torchvision-0.9.1%2Bcpu-cp39-cp39-linux_x86_64.whl
fastai==2.3.0
streamlit
```

- `Procfile` ประกอบด้วย

```
web: streamlit run --server.enableCORS false --server.port $PORT app.py
```

- `runtime.txt` ประกอบด้วย

```
python-3.9.5
```

- add และ commit ไฟล์ที่จำเป็นทั้งหมดขึ้นไปที่ branch `your_webapp_branch`

```
git add .
git commit -m 'first webapp commit'
git push origin your_webapp_branch
```

- login เข้าไปใน [heroku](https://dashboard.heroku.com/apps) แล้วกด `New > Create new app`

<figure align="center">
  <img src="{{ site.baseurl }}/images/heroku_create_new.JPG" style="width: 400px;"/>
</figure>

- ตั้งชื่อ webapp แล้วกด `Create app`

<figure align="center">
  <img src="{{ site.baseurl }}/images/heroku_app_name.JPG" style="width: 400px;"/>
</figure>

- ตรง `Deployment method` เลือก `Github; Connect to Github`; ใส่ชื่อ repository ที่ต้องการแล้วกด `Search`; เมื่อเจอ repository ที่ต้องการแล้วกด `Connect`

<figure align="center">
  <img src="{{ site.baseurl }}/images/heroku_deployment_method.JPG" style="width: 400px;"/>
</figure>

- เลื่อนลงมาที่ `Manual Deploy`; เลือก branch ที่ต้องการ (หากทำตามขั้นตอนมาคือ `your_webapp_branch`) แล้วกด `Deploy Branch`

<figure align="center">
  <img src="{{ site.baseurl }}/images/heroku_manual_deploy.JPG" style="width: 400px;"/>
</figure>

- หาก deploy สำเร็จสามารถเข้าไปเช็ค webapp ของเราได้โดยการกด `Open app` จากหน้าแรกของ webapp นั้นๆ

<figure align="center">
  <img src="{{ site.baseurl }}/images/heroku_open_app.JPG" style="width: 400px;"/>
</figure>

- แชร์ webapp ของคุณให้ทุกคนได้ใช้ผ่านทาง `https://your-app-name.herokuapp.com/`


## ตัวอย่างการ  deploy โมเดล chocolate chip vs raisin cookie classifier

- webapp บน heroku: https://choco-raisin.herokuapp.com/
- code สำหรับ deploy: https://github.com/cstorm125/choco-raisin/tree/streamlit

## อ้างอิง

- [fastai Deploying to Heroku with Voila](https://course.fast.ai/deployment_heroku)