---
layout: post
title: สรุป course.fast.ai (part1 v4) คาบที่ 1
---

ในคาบแรกของ fastai เราได้เรียนรู้ประวัติของ Deep Learning ซึ่งมีที่มาจากการพัฒนาของโมเดล Neural Networks เราได้เรียนรู้แอพพลิเคชั่นต่างๆของ Deep Learning ตั้งแต่ Computer Vision, Natural language processing และอื่นๆ โดยสรุปเนื้อหาในคาบแรกมีดังนี้

01. เราจะใช้หนังสือ [fastai/fastbook](https://github.com/fastai/fastbook) เป็นหนังสือเรียนหลัง โค้ดใน notebook นั้นแจกจ่ายโดยลิขสิทธิ์ open source ส่วนหนังสือ [Deep Learning for Coders with fastai and PyTorch](https://www.oreilly.com/library/view/deep-learning-for/9781492045519/) พิมพ์โดยสำนักพิมพ์ O’Reilly เราสามารถหาซื้อเพื่อสนับสนุน fastai และสำนักพิมพ์ได้

02. การทำความเข้าใจและใช้งาน deep learning นั้น
  - ใช้ความรู้คณิตศาสตร์เพียงระดับมัธยมปลาย; พีชคณิตเชิงเส้น, partial derivatives, chain rules, ความน่าจะเป็นและสถิติพื้นฐาน
  - ไม่จำเป็นต้องใช้ข้อมูลขนาดใหญ่มากเสมอไป; เมื่อเรามี pretrained models จากชุดข้อมูลขนาดใหญ่ เช่น [ImageNet](http://www.image-net.org/) หรือ [Assorted Thai Texts](https://arxiv.org/abs/2101.09635) แม้แต่โจทย์ง่ายๆที่ไม่มี pretrained models เราก็สามารถแสดงให้เห็นได้ว่า [deep learning ทำได้ดีกว่า linear model ทั่วไป](https://github.com/cstorm125/sophia)
  - และไม่จำเป็นต้องใช้คอมพิวเตอร์สุดแพงเพื่อให้ได้ผลที่ดีตามที่ต้องการ; ยกตัวอย่าง [thai2fit](https://github.com/cstorm125/thai2fit) และ [WangchanBERTa](https://medium.com/airesearch-in-th/wangchanberta-%E0%B9%82%E0%B8%A1%E0%B9%80%E0%B8%94%E0%B8%A5%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B8%A1%E0%B8%A7%E0%B8%A5%E0%B8%9C%E0%B8%A5%E0%B8%A0%E0%B8%B2%E0%B8%A9%E0%B8%B2%E0%B9%84%E0%B8%97%E0%B8%A2%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B9%83%E0%B8%AB%E0%B8%8D%E0%B9%88%E0%B9%81%E0%B8%A5%E0%B8%B0%E0%B8%81%E0%B9%89%E0%B8%B2%E0%B8%A7%E0%B8%AB%E0%B8%99%E0%B9%89%E0%B8%B2%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B8%AA%E0%B8%B8%E0%B8%94%E0%B9%83%E0%B8%99%E0%B8%82%E0%B8%93%E0%B8%B0%E0%B8%99%E0%B8%B5%E0%B9%89-d920c27cd433)

03. เราใช้ deep learning ในการทำงานเฉพาะทางต่อไปนี้ได้ดีเทียบเท่าหรือดีกว่ามนุษย์

04. AI Winter ครั้งที่ 1 เกิดขึ้นส่วนหนึ่งจากการที่ Marvin Minsky (รุ่นน้องโรงเรียนมัธยมของ Frank Rosenblatt ผู้สร้างเครื่องจำลอง neural network ชื่อ Perceptron เป็นคนแรก) แสดงให้เห็นว่า [neural network ขนาดเล็กไม่สามารถแก้ปัญหา XOR ได้](https://medium.com/@jayeshbahire/the-xor-problem-in-neural-networks-50006411840b)

05. ช่วงปี 1980s เริ่มมีแนวคิดว่าหากเราสร้าง neural network ที่มากชั้น-ขนาดใหญ่ขึ้นเรื่อยๆ เราจะ[สามารถประมาณ function อะไรก็ได้](https://en.wikipedia.org/wiki/Universal_approximation_theorem) แต่ปัญหาคือ neural network นั้นจะใหญ่และช้าเกินไปที่จะมีประโยชน์ด้วยทรัพยากรการคำนวณที่เรามีอยู่

06. ปรัชญาการเรียนของ fastai คือ
  - สอนให้เล่นเกมทั้งเกมก่อน ยกตัวอย่างการสอนเล่นเบสบอลโดยการให้ไปเล่นเบสบอล ไม่ใช่อธิบายว่าสนามขนาดเท่าไหร่ หรือดูวิดีโอการแข่งไปเรื่อยๆ
  - ทำให้เกมมีค่าที่จะเล่น; สอนว่าเบสบอลแพ้ชนะกันอย่างไร ชนะแล้วดีอย่างไร มีการนับแต้มอย่างจริงจัง
  - ทำในสิ่งที่ยาก; เล่นไปสักพักแล้วต้องมาปรับปรุงวิธีการเล่นให้ดีขึ้น ปรับท่าตี-วิ่ง-ขว้าง

07. Tech Stack ที่เราจะใช้งาน
  - Python
  - Pytorch
  - fastai; high-level API for Pytorch ใช้ได้ตั้งแต่ระดับ[ผู้เริ่มต้นไปจนถึงใช้งานจริงในอุตสาหกรรม](https://arxiv.org/abs/2002.04688)

08. [Jupyter notebook 101](https://github.com/fastai/fastbook/blob/master/clean/app_jupyter.ipynb)

09. วิธีการเรียนที่แนะนำ
  - ฟัง lecture จาก course.fast.ai
  - ลองทำตามด้วย notebook จาก notebook ใน [fastai/fastbook/clean](https://github.com/fastai/fastbook/clean)
  - มีข้อสงสัยกลับไปดู notebook ใน [fastai/fastbook](https://github.com/fastai/fastbook)
  - ตอบคำถามท้ายบท; ไม่ต้องถูกทุกข้อ ทำเพื่อเป็นการตรวจสอบตัวเองว่าเข้าใจไหม
  - ลองทำใหม่ขึ้นเอง-นำไปใช้กับโครงงาน

10. Traditional programming vs machine learning
  - Traditional programming: input → model → output
  - Training machine learning: [input, weights] → model → output→ performance → update weights
  - Using machine learning (inference): input → model → output

11. Machine learning ทำงานอย่างไร
  - [inputs, parameters] → architecture → [predictions, labels] → loss → update parameters
  - Architecture; รูปร่างของโมเดล เช่น y = mx+c
  - Parameters, weights; ค่าตัวแปรต่างๆของ architecture เช่น y = 2x+5; 2 และ 5 คือ parameter ของ architecture แบบ y=mx+c
  - Inputs, independent variables, features; ตัวแปรต้น ค่าที่ใส่เข้าไปเพื่อให้ได้คำทำนายเกี่ยวกับตัวแปรตาม
  - Predictions, outputs; คำทำนายของโมเดลเกี่ยวกับตัวแปรตาม
  - Labels, dependent variables, targets; ค่าจริงของตัวแปรตาม
  - Loss; ความแตกต่างระหว่างคำทำนายกับค่าจริงของตัวแปรตาม
  - Update; การปรับเปลี่ยน parameters เพื่อให้ได้ค่า loss ที่น้อยลง หรือคำทำนายที่เหมือนกับค่าจริงของตัวแปรตามมากขึ้น
  - Train; ทำซ้ำการ update ไปเรื่อยๆจนได้ parameters ที่ดีพอ

12. ข้อจำกัดของ machine learning
  - ต้องอาศัยการเรียนรู้จากข้อมูล
  - เรียนรู้รูปแบบของข้อมูลได้จากข้อมูลที่ใช้เทรนเท่านั้น
  - สร้างได้เฉพาะคำทำนาย ไม่สามารถตัดสินใจทำอะไรเองได้
  - ต้องอาศัย label ที่โดยทั่วไปต้องสร้างโดย annotators ที่เป็นมนุษย์

13. ข้อคำนึงถึงการใช้โมเดลกับ positive feedback loop; เราเชื่อโมเดลที่บอกว่าย่านใดย่านหนึ่งมีโอกาสเกิดเหตุอาชญากรรมเยอะ → เราส่งตำรวจไปประจำการเยอะ-บ่อยขึ้น → ตำรวจจับอาชญากรได้เยอะขึ้น → ข้อมูลถูกส่งกลับไปให้โมเดลว่าย่านนี้มีโอกาสเกิดเหตุอาชญากรรมเยอะ → โมเดลทำนายว่าย่านนี้มีโอกาสเกิดเหตุอาชญากรรมเยอะ → … →

14. การ `import *`
  - อาจจะทำให้เราสับสนได้ว่า module ที่เรากำลังจะใช้นั้นมาจากไหน หากอยากรู้ว่า module ต่างๆมาจากไหนให้พิมพ์ `?module_name`
  - ในการนำไปใช้จริงไม่ควรทำ `import *` แต่ควรเลือก import เฉพาะ module ที่เราต้องการใช้

015. ตัวอย่างแรกของการรันโค้ดบน Jupyter Notebook

``` py
# import the vision modules
from fastai.vision.all import *

# path to cats and dogs images
path = untar_data(URLs.PETS)/'images'

# path is not a string but Path object
path #  Path('/root/.fastai/data/oxford-iiit-pet/images')

# it contains a generator of Path objects
list(path.glob('*'))[:10] 
# [Path('/root/.fastai/data/oxford-iiit-pet/images/Ragdoll_118.jpg'),
# Path('/root/.fastai/data/oxford-iiit-pet/images/Persian_194.jpg'),
# Path('/root/.fastai/data/oxford-iiit-pet/images/american_pit_bull_terrier_69.jpg'),
# Path('/root/.fastai/data/oxford-iiit-pet/images/german_shorthaired_60.jpg'),
# ...]

# cat images start with uppercase; return true if cat iamge else false
def is_cat(x): return x[0].isupper()

# create dataloaders to load image into model during training
dls = ImageDataLoaders.from_name_func(
    path,  # path to images 
    get_image_files(path), 
    valid_pct=0.2, # randomly split 20% for validation set
    seed=42,  # set seed for image shuffling
    label_func=is_cat,  # get label from this function
    item_tfms=Resize(224)  # apply transformation; resize to 224x224 pixels
)

# create learner
learn = cnn_learner(
    dls, #with the dataloaders we created
    resnet34, #using resnet34 architecture pretrained on imagenet
    metrics=error_rate #metric to show is error rate (1-accuracy)
)

# finetune the architecture once frozen and once unfrozen
learn.fine_tune(1)
```

16. เฟรมเวิร์ค `fastai` สามารถใช้กับข้อมูลหลากหลายไม่เฉพาะภาพ แต่รวมถึงข้อความและตาราง

17. ตอบคำถามท้ายบทได้ที่ [aiquizzes](https://aiquizzes.com/howto)
