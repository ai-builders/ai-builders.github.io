---
layout: post
title: สรุป course.fast.ai (part1 v4) คาบที่ 7
---

ในคาบที่ 7 เราได้เรียนรู้เกี่ยวอัลกอริทึมที่เป็นพื้นฐานของการแนะนำหนังที่เราดูบน Netflix และแนะนำเพลงบน Spotify นั่นก็คือ
Collaborative Filtering (ตาม [08_collab.ipynb](https://github.com/fastai/fastbook/blob/master/08_collab.ipynb)) โดยเราได้เรียนวิธีการทำงานของ `collab_learner` โดยที่เราใส่ข้อมูลตารางจาก MovieLens ซึ่งประกอบด้วย `user`, `movie` และ `rating` เข้าไปได้

ถ้าจำจากคาบที่ 1 ได้ เราใช้ `collab_learner` ตามด้านล่างเพื่อใช้ในการประมาณการให้คะแนนของ `user` ดังนี้

``` python
# dls คือ DataLoader, n_factors คือจำนวนของ latent factor และ y_range คือช่วงของการให้คะแนน
learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
learn.fit_one_cycle(5, 5e-3, wd=0.1)
```

โดยในคาบนี้ Jeremy ก็ได้สอนทำงาว่าจริงๆแล้ว `collab_learner` ทำงานอย่างไร การประมาณ users-movies matrix จาก Latent Factors ทำได้ยังไง

จากนั้นเราข้ามไปยังการทำนายข้อมูลเซลล์โดยทำตาม [09_tabular.ipynb](https://github.com/titipata/fastbook/blob/master/09_tabular.ipynb)

## Collaborative filtering

* Collaborative filter ทำงานโดยการหา latent matrix ที่พอคูณกันแล้วได้เมทริกซ์เดิมกลับมา (เช่น user-movie rating data)
* หลักการของการทำงานเราจะใช้โมเดลตามด้านล่าง

``` py
class DotProduct(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users, n_factors) # Embedding layer ทำหน้าที่เหมือนการดึง index โดยใช้วิธีคูณ matrix แทน
        self.movie_factors = Embedding(n_movies, n_factors)
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        return sigmoid_range((users * movies).sum(dim=1), *self.y_range) # user x movie embeedings และครอบด้วย sigmoid function

model = DotProduct(n_users, n_movies, 50) # สร้างโมเดล กำหนดขนาด (dimension) ของ latent factor = 50
learn = Learner(dls, model, loss_func=MSELossFlat()) # สร้าง learner จาก dataloader, model และ loss ใช้ mean-square error
learn.fit_one_cycle(5, 5e-3) # fit ทั้งหมด 5 epochs ด้วย learning rate = 5e-3 = 0.005
```

`50` ในที่นี้คือขนาดของ Latent Factor (Embeddings)

* นอกจากนั้นเราสามาถเพิ่ม bias เข้าไปในโมเดลได้ด้วย เช่นเอามาประมาณเพิ่มในกรณีที่หนังมีความนิยมจากทุกคน จะได้เอา factor นั้นออกจาก user
* Movie Embeddings ที่ได้มาสามารถนำมาหาข้อมูลเพิ่มเติมได้อีก
  * เช่นเราสามารถลองไล่ดูหนังที่มี `movie_bias` ใกล้เคียงกันได้
  * หรือสามารถนำ Movie Embeddings มาลด dimension โดยใช้ principal component analysis (PCA) ก็ได้

* สุดท้ายแล้วเราสามารถเขียนโมเดลทำนาย user-item matrix ได้ด้วยตัวเอง แล้วใช้ `collab_learner` ของ Fast AI ได้

``` py
# ใช้ Neural network ขนาด hidden layers = [100, 50] ตามลำดับก็ได้
learn = collab_learner(dls, use_nn=True, y_range=(0, 5.5), layers=[100,50])
learn.fit_one_cycle(5, 5e-3, wd=0.1)
```

## ทำนายข้อมูลตาราง

ในครึ่งหลังเราก็ได้ลองใช้อัลกอริทีมอื่นๆนอกเหนือจาก Deep learning นั่นก็คือ Decision Tree และ Random Forest (เทคนิค Bagging)
เพื่อทำนาย `SalePrice` ของ จากข้อมูลการประมูลเครื่องจักรขนาดใหญ่บน Kaggle:
[Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers)

* การโหลดข้อมูลสามารถใช้ Kaggle API โดยลงไลบรารี่ของ Kaggle ได้ดังนี้ `pip install kaggle`
* เปลี่ยนประเภทของ column โดยใช้ `astype` ยกตัวอย่างเช่น

``` py
sizes = 'Large','Large / Medium','Medium','Small','Mini','Compact'
df['ProductSize'] = df['ProductSize'].astype('category')
df['ProductSize'].cat.set_categories(sizes, ordered=True, inplace=True) # เซ็ต categories ตามที่กำหนดข้างต้น
```

และกำหนด Dependent variable หรือสิ่งที่เราอยากจะทำนาย เช่นในตัวอย่างเค้าใช้ log ของ SalePrice เป็นค่าที่ต้องการทำนาย

* หักใช้งาน `TabularPandas` และ `TablularProc`

``` py
procs = [Categorify, FillMissing] # processors
cond = (df.saleYear<2011) | (df.saleMonth<10)
train_idx = np.where( cond)[0]
valid_idx = np.where(~cond)[0]

splits = (list(train_idx),list(valid_idx))
cont, cat = cont_cat_split(df, 1, dep_var=dep_var) # แบ่ง column เป็นประเภท continuous variables, categorical variables ด้วยฟังก์ชั่นจาก Fast AI tabular
# สร้าง dataset จาก dataframe
to = TabularPandas(df, # dataframe
                   procs, # processors
                   cat, # list ของ columns ที่เป็น categorical
                   cont, # list ของ columns ที่เป็น continuous
                   y_names=dep_var, # dependent variable (SalePrice)
                   splits=splits) # index ของ train และ validation
```

โดย `TabularPandas` จะรับ dataframe, processors, column ที่เป็น categorical, column ที่เป็น continuous, และค่าที่เราจะทำนาย (Dependent variable)

* ลองใช้ Decision Tree ทำนายข้อมูลตาราง

``` py
# แบ่ง tabular object (to) ที่ได้เป็น train, valid
xs, y = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y

m = DecisionTreeRegressor(max_leaf_nodes=4) # สร้าง decision tree โดยมี nodes สุดท้ายจำนวน 4 ใบ
m.fit(xs, y) # fit ข้อมูลที่เราทำขึ้นมา
```

* ปัญหาของ Decision tree ที่เราได้เห็นจากตัวอย่างคือ overfitting เนื่องจากเราสมารถสร้าง decision tree ที่มีจำนวนใบมากเกือบเท่าจำนวนข้อมูล ซึ่งทำให้โมเดลเรียนรู้จากข้อมูลที่เราป้อนให้มันเท่านั้น แต่ไม่สามารถใช้งานกับข้อมูลที่เหลือได้

* ต่อไปเราทำความรู้จักกับ Random Forest ซึ่งจริงๆคือ Bagging Predictors แบบนึง โดยหลักการสร้าง Bagging predictors เป็นดังนี้
  * สุ่ม subset ของข้อมูลขึ้นมา
  * เทรนโมเดลด้วย subset ของข้อมูลขึ้นมา
  * เก็บโมเดลไว้ แล้วทำซ้ำ
  * ท้ายสุดทำโมเดลทุกตัวที่เทรนมามาทำนาย แล้วใช้ค่าเฉลี่ยเป็นผลทำนาย (average)

ใน `RandomForestRegressor` จำนวนโมเดลหรือต้นไม้คือ `n_estimators` ส่วนจำนวนข้อมูลที่สุ่มขึ้นมาคือ `max_samples` นอกจากนั้นเรายังสามารถสุ่มแค่บาง features (columns) จากข้อมูลได้ด้วยโดยใช้พารามิเตอร์ `max_features`

* ทำความรู้จักกับ Out-of-Bag Error, Out-of-Bag Error เป็นการทำนาย error จากข้อมูลที่ไม่ได้เอามาใช้เทรนโมเดลตอนสุ่มข้อมูลมาเทรนโมเดล เป็นเทคนิคที่ทำให้เราสามารถประมาณ error ได้โดยไม่ต้องเห็น validation data ก็ได้

* ทำความเข้าใจโมเดล Random Forest สามาถทำได้หลายวิธีโดยในที่นี้ Jeremy ยกตัวอย่างเช่น
  * ใช้ feature importance โดยดูจาก `model.feature_importances_` หลังจากเทรนโมเดล
  * โดยการดู feature importance เราสามารถนำ columns ที่มี feature importance ต่ำๆออกแล้วทำนายอีกรอบได้
  * นอกจากนั้นเรายังสามารถใช้ `cluster_columns(xs_imp)` เพื่อดูว่า features ใดที่มีความใกล้เคียงกันเพื่อนำ column พวกนั้นออกไปได้อีก
  * จากตัวอย่างที่ Jeremy ยกตัวอย่างจะเห็นว่าเมื่อเราเอา columns ออกไปแล้ว แต่ error หลังจากการสร้างโมเดลยังใกล้เคียงกับของเดิมที่ใช้ทุก features

* ทำความเข้าใจโมเดลด้วยวิธีอื่นๆได้อีก เช่นการใช้ `plot_partial_dependence` เพื่อความสัมพันธ์ระหว่างฟีเจอร์ที่เราสนใจและ Dependent variable (SalePrice)

``` py
from sklearn.inspection import plot_partial_dependence
fig,ax = plt.subplots(figsize=(12, 4))
plot_partial_dependence(m, valid_xs_final, ['YearMade','ProductSize'],
                        grid_resolution=20, ax=ax) # พล็อตเพื่อดู effect ของปีและขนาดของ product ต่อ SalePrice
```

หรือใช้ `waterfall` plot เพื่อหาว่าฟีเจอร์อะไรมีผลต่อโมเดลบ้าง

* ปัญหาของการใช้ Random Forest ก็ยังมีอยู่คือไม่สามารถทำนาย out-of-domain data ได้ เช่นในกรณีของข้อมูลที่เรามี จะเห็นว่าราคาสามารถขึ้นไปสูงได้อีก แต่ Random Forest ไม่สามารถใช้ได้ในกรณีนี้ ทำให้การใช้ Neural Network ยังสามารถทำงานได้ดีกว่าในบางกรณี

``` py
# สร้าง dataset จาก dataframe
procs_nn = [Categorify, FillMissing, Normalize] # processors ที่ใช้
to_nn = TabularPandas(df_nn_final, # dataframe ที่ใช้ทำนาย
                      procs_nn, # processors ที่ใช้
                      cat_nn, # list ของ columns ประเภท categorical
                      cont_nn, # list ของ columns ประเภท continuous
                      splits=splits, # index ที่แบ่ง training และ validation
                      y_names=dep_var) # dependent variable
dls = to_nn.dataloaders(1024) # สร้าง dataloader ด้วย batch size = 1024

learn = tabular_learner(dls, y_range=(8,12), layers=[500,250],
                        n_out=1, loss_func=F.mse_loss)
learn.lr_find() # plot เพื่อหา learning rate ที่น่าใช้ -> ใช้ 1e-2
learn.fit_one_cycle(5, 1e-2) # fit จำนวน 5 epochs ด้วย learning rate 1e-2

preds, targs = learn.get_preds()
r_mse(preds,targs) # 0.226 ทำงานได้ดีเท่าๆกับ Random Forest ที่เราสร้างมาข้างต้นเลย
```

* ทิ้งท้ายด้วยการนำเทคนิคของทั้งสองอย่างมารวมกัน เช่นการใช้ embeddings จาก Neural network เข้ามาเป็นฟีเจอร์สำหรับเทรนโมเดลอีกรอบนึง