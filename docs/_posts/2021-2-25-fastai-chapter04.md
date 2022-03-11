---
layout: post
title: สรุป course.fast.ai (part1 v4) คาบที่ 4
---

เราจะมาลงรายละเอียดว่า stochastic gradient descent (SGD) ทำงานอย่างไร ต่อจากคาบที่แล้วด้วย `04_mnist_basics.ipynb` และ `05_pet_breeds.ipynb`

01. การใช้งาน loss function ด้วย Pytorch ทำได้ 2 วิธี (อ้างอิง `05_pet_breeds.ipynb`)

```py
# สร้าง class ขึ้นมาแล้วเรียกใช้
from torch import nn
loss_func = nn.CrossEntropyLoss()
loss_func(acts, targ)

# เรียกใช้ functional API
import torch.nn.functional as F
F.cross_entropy(acts, targ)

#ถ้าอยากได้ตัวเลขก่อน aggregation (ส่วนใหญ๋ aggregate ด้วย mean)
nn.CrossEntropyLoss(reduction='none')(acts, targ)
```

02. โมเดลแยกรูปเลข 3 และเลข 7 (อ้างอิง `04_mnist_basics.ipynb`) เริ่มจากสร้างคู่ตัวอย่าง (X, y) จากรูปภาพ 28x28

```py
#สร้างตัวแปร X, y สำหรับเทรนโมเดล 
#ต่อ tensor สองอันเข้าด้วยกัน (default ด้วยมิติ 0)
train_x = torch.cat([stacked_threes, stacked_sevens])\
	.view(-1, 28*28) #เปลี่ยนมิติเป็น (-1, 28*28); -1 คือให้มีมิติเท่าเดิม (จำนวนรูป)

#ให้เลข 3 เป็น 1 ส่วนเลข 7 เป็น 0
train_y = tensor([1]*len(threes) + [0]*len(sevens))\
	.unsqueeze(1) #เพิ่ม dimension ให้กับ tensor; ในกรณีนี้คือเจาะจงว่าให้เพิ่มใน dimension ที่ 1 คือ (# examples,) -> (# examples, 1)
```

03. สร้าง dataset สำหรับ training และ validation sets

```py
#สร้าง dataset object ไว้เพื่อเก็บคู่ (X,y) ของ training set
dset = list(zip(train_x,train_y)) #zip (X,y) เข้าด้วยกันแล้วเปลี่ยนเป็น list
x,y = dset[0] #เลือก (X,y) ที่ index 0

#ทำแบบเดียวกันกับ validation set
valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))
```

04. ตั้งค่า weights และ bias เริ่มต้นด้วยการสุ่ม

```py
#ตั้งค่า weights และ bias เริ่มต้นด้วยการสุ่มจาก standard normal distribution
def init_params(size, std=1.0): 
	return (torch.randn(size)*std).requires_grad_() #requires_grad_() เพื่อให้ Pytorch เก็บ gradient สำหรับ backpropagation

#สมการของเราคือ y = weights * X + bias
weights = init_params((28*28,1))
bias = init_params(1)
```

05. ทำนายจากตัวอย่าง

```py
#prediction สำหรับ example #0
#ผลรวมของ (# examples, 784) * (1, 784) บวกด้วย (1,)
#weights.T มีมิติ (1, 784) จะถูก broadcast ไปคูณสำหรับทุก example 
#ดูวิธีทำได้จาก http://matrixmultiplication.xyz/
(train_x[0] * weights.T).sum() + bias 

#สำหรับ Pytorch @ คือการทำ matrix multiplication ตามวิธีข้างต้น
def linear1(xb): return xb@weights + bias
preds = linear1(train_x) #dimension คือ (# examples, 1)

corrects = (preds>0.0).float() == train_y #ถ้า prediction > 0 ให้ทายว่า 1 (class ของ 7) แล้วดูว่าตรงกับ ground truth (y) แค่ไหน (True/False)

#จากโมเดลที่ไม่ได้เทรนเลยเราจะเห็นได้ว่าตรงประมาณครึ่งๆ (เพราะเรามีเลข 3 และ 7 พอๆกันถ้าเดามั่วๆมีโอกาสถูกครึ่งๆ)
corrects.float().mean().item() #item() เพื่อเอาค่าที่ถูกเก็บไว้ใน tensor ไม่ใช่ตัว tensor ทั้งหมด
```

06. สร้าง loss function

```py
#หนึ่งในเหตุผลที่เราไม่ใช้ accuracy เป็น loss function คือ accuracy ไม่ไวต่อการเปลี่ยนแปลงของ weights/bias พอ เช่น ในกรณีนี้หากเราคูณ weights ด้วย 1.0001 เราก็จะยังได้ accuracy เท่าเดิม (เพราะ threshold ที่จะเปลี่ยนคำทำนายของเราคือ prediction > 0.0) ทำให้ยากที่โมเดลจะเรียนรู้
weights[0] *= 1.0001
preds = linear1(train_x)
((preds>0.0).float() == train_y).float().mean().item()

#ถ้า ground truth คือ 1 ให้ค่า 1-predictions ถ้าเป็น 0 ให้ค่า predictions
#prediction ยิ่งใกล้ ground truth ที่ถูกเท่าไหร่ ค่า loss ยิ่งน้อยลงเท่านั้น
#ค่า loss เปลี่ยนทุกครั้งที่ prediction เปลี่ยน ไม่เหมือนกับ accuracy ด้านบน
def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()

#ปัญหาคือถ้า predictions ไม่อยู่ระหว่าง 0 กับ 1 ค่า loss อาจจะไม่สมเหตุสมผล
#เช่น predictions = 100, ground truth = 1, loss = 1-100 = -99
#เราจึงใช้ activation function เช่น sigmoid
def sigmoid(x): return 1/(1+torch.exp(-x))

#เราใส่ sigmoid function ไปใน loss เลยก็ได้

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()
```

07. Metric คือค่าที่เราใช้วัดผลของโมเดล; loss function คือค่าที่ใกล้เคียงกับ metric และมี gradient ที่ดีเหมาะกับการเทรนด้วย SGD

08. ห้องส่งมีคำถามว่า "ทำไมเราถึงหา mean แทนที่จะใช้ median เพราะน่าจะเป็นค่าที่ "นิ่ง" กว่าถ้าเจอ outlier" Jeremy ตอบว่าเขายังไม่เคยใช้ แต่การใช้ median 1.ทำให้โมเดลสนใจแค่ค่าตรงกลาง 2. อาจจะทำให้มี gradient เป็น 0 เยอะ เพราะค่าไม่ค่อยเปลี่ยนเหมือนกับที่เราใช้ accuracy ทีแรก แต่น่าลองนะ

09. SGD และ mini-batches เราไม่สามารถใช้ตัวอย่างทั้งหมดมาก update weights ในคราวเดียวได้เพราะชุดข้อมูลส่วนใหญ่มีจำนวนตัวอย่างมาก เช่น ImageNet มีถึง 14 ล้านรูป เป็นต้น เราจึงนิยมค่อยๆ update ไปทีละ mini-batch จำนวนแล้วแต่ GPU จะรับได้ เช่น รอบละ 128, 256, 512 ตัวอย่าง เป็นต้น

```py
coll = range(15) #ข้อมูลเลขจาก 0-14
#dl คือ iterator
dl = DataLoader(coll, batch_size=5, shuffle=True) #โหลดข้อมูลทีละ 5 ตัวอย่างต่อ 1 mini-batch; สับข้อมูลมั่วๆด้วย
list(dl) #เปลี่ยน iterator เป็น list ทำให้แสดงข้อมูลทั้งหมด 3 mini-batches; mini-batch ละ 5 ตัวอย่าง

#dataloader สำหรับ tuples
ds = L(enumerate(string.ascii_lowercase)) #[(0,'a'), (1,'b'),...]
dl = DataLoader(ds, batch_size=3, shuffle=False)
list(dl) #[((0,1,2),('a','b','c')), ...]
```

10. Training step ใน SGD เริ่มด้วยการทำนาย, คำนวณ loss, คำนวณ gradients, และจบลงด้วยการปรับ weights ด้วย gradients

```py
def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward() #คำนวณ gradients จาก operations ด้านบน

def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model) #คำนวณ gradients
        #สำหรับ parameter ทุกตัว
        for p in params:
        	#ลบ parameter ด้วยค่า gradeint คูณด้วย learning rate
        	#ใช้ p.data เพื่อไม่ให้ Pytorch คำนวณ gradient จากการลบนี้
            p.data -= p.grad*lr 
            #จำเป็นต้องเซ็ต gradients จากทุก tensor เป็น 0 ไม่งั้นเวลาเรียก .backward() จะเอา gradients ใหม่มาบวกกับอันเก่า
            p.grad.zero_() 
```

11. Validation step ไม่จำเป็นต้องคำนวณ gradients แต่คำนวณ metric เช่น accuracy

```py
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4) #คำนวณค่าเฉลี่ยของ acc จากทุก mini-batch
```

12. นำทุกอย่างมารวมกับเป็น loop การเทรน SGD

```py
lr = 1. #learning rate
params = weights,bias #parameter ของเรา ในกรณีนี้มีแค่ weights และ bias ที่เอามาทำ y=xb@weights + bias

for i in range(20): #เทรนไป 20 epochs (เทรนด้วยข้อมูลทั้งหมด 20 รอบ)
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')
```

13. Refactor ไปใช้ Pytorch ทำ optimizer

```py
linear_model = nn.Linear(28*28,1) #สร้างโมเดลที่ทำ y=xb@weights + bias ง่ายๆ

#นิยาม weights และ bias ด้วย .parameters()
w,b = linear_model.parameters()

#สร้างคลาส optimizer
class BasicOptim:
    def __init__(self,params,lr): 
    	#กำหนด parameters และ learning rate เบื้องต้น
    	self.params = list(params)
    	self.lr = lr

    def step(self, *args, **kwargs):
    	#ทำ SGD
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
    	#รีเซต gradients ให้เป็น 0
        for p in self.params: p.grad = None

#สร้าง optimizer
opt = BasicOptim(linear_model.parameters(), lr)

#train ด้วย optimizer ที่เราสร้าง
def train_epoch(model):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()

#เทรนโมเดลเป็นจำนวน X epochs
def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')

#initialize weights/bias
linear_model = nn.Linear(28*28,1)
#สร้าง optimizer
opt = SGD(linear_model.parameters(), lr) #Pytorch/fastai มีคลาส SGD ที่เป็นคลาส optimizer เหมือนกันที่เราทำข้างบนอยู่แล้ว
#เทรนโมเดล 20 epochs
train_model(linear_model, 20)
#สร้าง dataloaders
dls = DataLoaders(dl, valid_dl)
#สร้าง learner object ที่ทำการรวบทุกอย่างเข้าด้วยกันเพื่อความง่าย
learn = Learner(dls, #dataloaders
	nn.Linear(28*28,1), #model
	opt_func=SGD, #optimizer
	loss_func=mnist_loss, #loss function
	metrics=batch_accuracy #metric
	)
#เทรนโมเดลจาก learner object
learn.fit(10, lr=lr)
```
14. เพิ่ม non-linearity activation ลงไปใน linear model ด้านบนเพื่อสร้าง neural network แรกของเรา โดยการใส่ ReLU (ถ้า input x มีค่าน้อยกว่า 0 ให้เป็น 0 ถ้ามากกว่า 0 ให้เป็น x)

```py
def simple_net(xb): 
    res = xb@w1 + b1
    res = res.max(tensor(0.0)) #ReLU หาค่า max ระหว่างค่าที่อยู่ในแต่ละ element ของ res กับ 0
    res = res@w2 + b2
    return res

#ทำแบบเดียวกันด้วย Pytorch ก็ได้
simple_net = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)

#สร้าง learner แล้วทำเหมือนเดิม
learn = Learner(dls, simple_net, opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)

#เทรน
learn.fit(40, 0.1)

#โชว์ผล batch_accuracy ตลอด 40 epochs
plt.plot(L(learn.recorder.values).itemgot(2)); #get item at index 2 from each row
```

15. มีโอกาสที่ ReLU จะทำให้ gradients เป็น 0 เป็นจำนวนมากหรือเป็น 0 ตลอดระยะเวลาการเทรน เราจึงมีวิธีแก้ไข เช่น ไม่ทำให้ส่วนที่ x<0 เป็น 0 ทั้งหมด แต่อาจจะเป็น slope ที่มีค่าลบเล็กน้อย (leaky ReLU) หรือเทรนด้วย learning rate ที่ไม่สูงหรือต่ำจนเกินไปจนทำให้ gradients มีค่าใหญ่หรือเล็กเกินไป

16. เราสามารถดู weights ที่ถูกเทรนแล้วเห็นว่ามีการเรียนรู้รูปร่างของตัวเลข

```py
m = learn.model
w,b = m[0].parameters() #parameters ของ layer แรก
show_image(w[0].view(28,28))
```

17. ชนิดของตัวเลข ใน neural networks

- parameters; ตัวเลขที่ถูกสุ่มขึ้นมาในตอนแรก และเรียนรู้ระหว่างเทรน (weights และ bias)
- activations; ตัวเลขที่ถูกคำนวณโดย linear หรือ non-linear layer

18. เรียนรู้ regular expression ได้ที่ [regexr](https://regexr.com/)

19. `item_tfms` to crop and resize first then augmentation transform with `aug_transforms`

```py
pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460), #crop and resize randomly first
                 batch_tfms=aug_transforms(size=224, min_scale=0.75)) #augmentation transforms
dls = pets.dataloaders(path/"images")

#เช็คว่า data augmentation สมเหตุสมผล
dls.show_batch(nrows=1, ncols=3)
#เช็คว่า data augmentation สมเหตุสมผลแบบดูจากรูปเดียว
dls.show_batch(nrows=1, ncols=3, unique=True)
```

20. เช็คว่า `DataBlock` ถูกสร้างขึ้นมาอย่างไร

```py
pets1 = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'))

#ตัวอย่างที่ไม่ resize ด้วย item_tfms ก่อน ทำให้ DataBlock เอารูปมาต่อกันไม่ได้
pets1.summary(path/"images")
```

21. Jeremy แนะนำให้เริ่มเทรนโมเดลให้เร็วที่สุดเท่าที่จะทำได้โดยไม่ต้องเสียเวลาทำความสะอาดข้อมูล (เช่น รูปผิดฉลาก ฯลฯ) มาก เพราะ 1. จะได้รู้ baseline ของ use case ของเราก่อน 2. เราสามารถดูว่าโมเดลแรกที่เราทำนายทำพลาดที่ตัวอย่างไหน หลายครั้งตัวอย่างเหล่านั้นคือข้อมูลที่ "ไม่สะอาด"

22. Cross-entropy loss ใช้สำหรับ multi-class classification

```py
#softmax activation ให้ "probability" ของแต่ละคลาสที่รวมกันแล้วได้ 1.
def softmax(x): return exp(x) / exp(x).sum(dim=1, keepdim=True)
sm_acts = torch.softmax(acts, dim=1) #apply softmax โดยให้แต่ละแถวรวมกันได้ 1

#เลือก activation ใน column ของค่า target; ตัวที่ 1 เลือก column 0, ตัวที่ 2 เลือก column 1, ...
targ = tensor([0,1,0,1,1,0])
idx = range(6)
sm_acts[idx, targ]

#Pytorch-sytle negative log likelihood loss
-sm_acts[idx, targ]

#Pytorch version; จะเห็นได้ว่าถึงมี log ในชื่อแต่มันเป็นแค่ sum of -softmax(x) ไม่มี log
#ทั้งนี้เป็นเพราะ Pytorch คาดหวังให้เราใช้ torch.log_softmax ที่จะทำ log(softmax(x)) ก่อน
F.nll_loss(sm_acts, targ, reduction='none')

#CrossEntropyLoss ใน Pytorch จะทำการหา sum of -log(softmax(x))
loss_func = nn.CrossEntropyLoss()
```

23. การใช้ log บน softmax(x) ทำไปเพื่อให้ความแตกต่างระหว่าง "ความน่าจะเป็น" ที่อยู่ระหว่าง -1 จาก softmax ชัดเจนยิ่งขึ้น เช่น 0.9 กับ 0.99 ต่างกัน 0.09 แต่ log(0.9) กับ log(0.99) ต่างกันประมาณ 10 เท่า

24. ตอบคำถามท้ายบทได้ที่ [aiquizzes](https://aiquizzes.com/howto)
