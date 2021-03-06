---
layout: post
title: สรุป course.fast.ai (part1 v4) คาบที่ 3
---

คาบที่ 3 เริ่มต้นด้วยการนำความรู้จากคาบที่แล้วมาสร้างแอพพลิเคชั่นเพื่อการแบ่งแยกประเภทต่างๆของหมี แล้วต่อด้วยการย้อนกลับ
ไปทำความเข้าใจพื้นฐานของการหาพารามิเตอร์ที่ทำให้ objective function มีค่าต่ำที่สุด (หรือสูงที่สุด) โดยใช้เทคนิค gradient descent

* เริ่มต้นมา Jeremy อธิบายการใช้เทคนิคต่างๆสำหรับการ Image Transformation ภาพหลังจากเราได้ Dataloader มาเช่น `Resize(128)`,
`RandomResizedCrop(128, min_sclae=0.3)` โดยหลังจากที่เราใช้แล้วก็จะสามารถลองดูรูปได้ดังนี้

``` python
bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))  # apply การ crop จากรูปเดิมแบบสุ่มและ scale รูปด้วย
dls = bears.dataloaders(path) # ได้ Dataloader มา
dls.valid.show_batch(max_n=4, n_rows=1) # โชว์รูปภาพหลังจากใช้ Image transfrom: RandomResizedCrop
```

นอกจากนั้นเรายังใส่ method เข้าไปได้เช่น `ResizeMethod.Pad` เป็นต้น หลังจากนั้นเราก็เทรนโมเดลโดยใช้ `cnn_learner` เช่นเดิมโดยใส่เข้าไปคือ Dataloader, Model Architecture และ Metrics

``` py
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4) # เทรนจำนวน 4 epochs
```

พอได้ข้อมูลออกมาเราใช้ ความผิดพลาดที่เกิดขึ้นโดยใช้ `ClassificationInterpretation` กับ `ImageClassifierCleaner`

``` py
interp = ClassificationInterpretation.from_learner(learn) # class สำหรับดูรายละเอียดของการทำนายผลข้อมูล
cleaner = ImageClasfierCleaner(learn) # ดึงภาพที่ loss สูงที่สุดแล้วแก้ได้ เช่นบางภาพที่โหลดมาเป็นภาพเปล่า
```

หลังจากได้ `cleaner` เราสามารถนำข้อมูลที่ผิดอยู่ออกไปได้

``` py
for idx in cleaner.delete():
    cleaner.fns[idx].unlink() # ลบ index นั้นๆจาก Path
```

* จากนั้น Jeremy สอนใช้ `widgets` สำหรับการทำ Web application ง่ายๆพอเราเทรนโมเดลเสร็จเรียบร้อย โดยเอาปุ่มต่างๆมาต่อกันได้ เช่น upload, classify เพื่อทำนายภาพ เมื่อเราทำ widgets ขึ้นมาเรียบร้อยแล้ว เราสามารถแยก widgets ให้อยู่ใน Jupyter notebook เดียว
แล้ววางไปบน Github และใช้ [`Viola`](https://voila.readthedocs.io/en/stable/) หรือ [`Binder`](https://mybinder.org/) เพื่อเปลี่ยน Jupyter notebook เป็น Web application สำหรับการ classify รูปได้

* ระวังการสร้างโมเดลจากข้อมูลที่อาจจะไม่ครอบคลุมกับประชากรหลายๆแบบ โดย Jeremy ยกตัวอย่างการเสิร์ชคำว่า Healthy skin เข้าไปใน Bing หรือ Duck Duck Go จะเห็นว่าภาพที่ได้ออกมาเป็นภาพผู้หญิง(ส่วนมากผิวขาว)  ให้มือป้องหน้า ซึ่งถ้าเราเอาข้อมูลเหล่านี้มาเทรนโมเดลจริงๆ อาจจะทำให้โมเดลของเราบอกว่าการมีผิวขาว หรือเอามือป้องหน้าคือการมีผิวที่ดี Healhy Skin ก็ได้

* จากนั้น Jeremy ย้อนกลับมาที่การแบ่งภาพข้อมูลการเขียนตัวเลข MNIST เพื่อทำให้เราเรียนรู้การทำงานของการ optimize parameters
โดยการใช้ gradient descent เพื่อหาจุดต่ำสุดของ objective function ที่เรากำหนด

* ยกตัวอย่างการใช้ gradient descent เพื่อหาจุดต่ำสุดของกราฟพาราโบลาทำได้ดังนี้

``` py
import torch
import matplotlib.pyplot as plt

def f(x):
    return x ** 2 - 4 * x + 5
x = torch.arange(-4, 10, 0.1)
y = f(x)
plt.plot(x, y)

# gradient descent algorithm เพื่อหาค่า x ที่ทำให้ f(x) มีค่าต่ำที่สุด
x = torch.tensor(5.).requires_grad_()
x_opt = torch.tensor(-10.).requires_grad_() # ใส่ requires_grad_() เพื่อทำให้หา gradient ได้
for i in range(1000):
    loss = f(x_opt) # แทนค่าเพื่อหา loss
    loss.backward() # คำนวณ​ gradient
    x_opt.data -= x_opt.grad.data * 0.01 # gradient descent
    x_opt.grad = None # หรือใช้ x_opt.zero_() เพื่อเคลียร์​ gradient ที่หามาจากจุดเดิม
```

* ยกตัวอย่างการใช้ gradient descent เพื่อแยกภาพเลข 3 ออกจากเลข 7 โดยใช้ MNIST Dataset

``` py
import torch
from fastai.vision.all import *

path = untar_data(URLs.MNIST) # โหลดข้อมูลและ untar ในคอมพิวเตอร์ของเรา
threes = (path/"training"/"3").ls().sorted()
sevens = (path/"training"/"7").ls().sorted()

valid_threes = (path/"testing"/"3").ls().sorted()
valid_sevens = (path/"testing"/"7").ls().sorted()

Image.open(threes[19]) # โชว์ภาพเลข 3

# อ่านภาพและนำภาพมาต่อกันเป็นชั้นๆ
three_tensors = [tensor(Image.open(o)) for o in threes] # อ่านภาพเข้ามาใน list
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = torch.stack(three_tensors) # ต่อภาพใช้ torch.stack
seven_tensors = torch.stack(seven_tensors)

# เตรียมข้อมูลและ class โดยใช้ 3 เป็น class 0 และ 7 เป็น class 1
X = torch.cat([three_tensors, seven_tensors]).view(-1, 28 * 28) / 255
y = tensor([1] * len(threes) + [0] * len(sevens)).unsqueeze(1)

def init_params(size, std=1.0):
    return (torch.randn(size)*std).requires_grad_()

def sigmoid(x):
    return 1/(1 + torch.exp(-x))

def linear1(x):
    return x @ weights + bias

# สร้างฟังก์ชั่นเพื่อแปลภาพให้กลายเป็น class 0, 1
weights = init_params((28 * 28, 1))
bias = init_params(1)
y_pred = (linear1(X).sigmoid() > 0.5).float() # ลองทำนายจาก random weight

# loss ในที่นี้คือเราพยายยามจะทำให้ทำนายใน class 1 ตรงกับ 1 มากที่สุด และ class 0 ตรงกับ 0 มากที่สุด จึงใช้ torch.where(cond, x1 (ถ้าตรง condition), x2 (ถ้าไม่ตรง condition))
def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1 - predictions, predictions).mean()


# จากนั้นสร้าง DataLoader เพื่อมาใส่โมเดล
dl = DataLoader(list(zip(X, y)), batch_size=256)
def train_epoch(model, lr, params):
    for x_batch, y_batch in dl:
        y_pred_batch = linear1(x_batch)
        loss = mnist_loss(y_pred_batch, y_batch)
        loss.backward()
        for p in params:
            p.data -= p.grad * lr
            p.grad.zero_()

# เทรนจำนวน 20 ephochs
for _ in range(20):
    train_epoch(linear1, 1, params)

# หา accuracy หลังจากการเทรนจำนวน 20 epochs
y_pred = (linear1(X).sigmoid() >= 0.5).float()
accuracy = (y_pred == y).float().mean()
```

* กลับมาที่การใช้ Fast AI บ้าง เราสามารถใช้ Fast AI เทรนโมเดลได้เช่นกัน ดังต่อไปนี้

``` py
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

# เตรียม validation data คล้ายๆกับการเตรียม training data ข้างต้น
valid_three_tensors = torch.stack([tensor(Image.open(o)) for o in valid_threes]) / 255
valid_seven_tensors = torch.stack([tensor(Image.open(o)) for o in valid_sevens]) / 255

valid_x = torch.cat([valid_three_tensors, valid_seven_tensors]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_three_tensors) + [0]*len(valid_seven_tensors)).unsqueeze(1)
valid_dset = list(zip(valid_x, valid_y))

# ใส่ training data, validation data เข้าไปใน dataloader
valid_dl = DataLoader(valid_dset, batch_size=256)
dls = DataLoaders(dl, valid_dl)

# สร้าง Fast AI Learner ใส่ dataloader, model architecture (Linear), optimization function, loss, metrics
learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)

# เทรนทั้งหมด 40 epochs ด้วย learning rate = 0.1
learn.fit(40, 0.1)
```
