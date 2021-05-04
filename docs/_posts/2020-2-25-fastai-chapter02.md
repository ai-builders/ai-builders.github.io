---
layout: post
title: สรุป course.fast.ai (part1 v4) คาบที่ 2
---

ในคาบนี้ เราจะเรียนรู้การวัดผลของโมเดล machine learning ด้วย validation set และ test set, ความสำคัญของการมี benchmark, และการทำ transfer learning 
Recap notebook in [Google Colab](https://colab.research.google.com/drive/174uEkWaYjtVAK0oHMQLSCR9Jry6GFxyt?usp=sharing)

01. ชนิดของโมเดล machine learning สามารถแบ่งได้ตาม output คร่าวๆ คือ
  - classification; ทำนาย**ชนิด**ของตัวอย่าง อาจจะเป็น multi-class (ตัวอย่างเป็นได้เพียงชนิดในชนิดหนึ่ง เช่น รูปเป็นหมาหรือแมว) หรือ multi-label (ตัวอย่างเป็นได้หลายชนิดหรือไม่เป็นชนิดใดเลยก็ได้ เช่น ข่าวเป็นประเภทการเมือง, เศรษฐกิจ, และกีฬาไปพร้อมๆกัน)
  - regression; ทำนาย**จำนวน**จากตัวอย่าง เช่น ทำนายอุณภูมิร่างกายจากรูปภาพหน้าคน

02. metric คือตัวชี้วัดผลงานของโมเดล แต่ไม่ใช่สิ่งเดียวกับ loss ที่โมเดลใช้เป็นเป้าหมายการ optimize และ update ค่า parameters (weights) เสมอไป

03. เหตุผลที่เราต้องมี train-validation-test splits คือ
  - train/validation; ป้องกันการ "เห็นข้อสอบ" ของโมเดล (overfitting) โดยให้โมเดลเทรนกับข้อมูลชุดหนึ่งและสอบกับข้อมูลอีกชุดหนึ่ง
  - train/validation/test; หากใช้ validation set เดียวกันทดสอบซ้ำไปเรื่อยๆ ผู้สร้างโมเดลก็จะสามารถปรับโมเดลให้ได้ผลดีใน validation set นั้นๆได้ และนำไปสู่ overfitting ในที่สุด โดยทั่วไปจึงเป็นที่นิยมที่จะแบ่งข้อมูลเป็น 3 ส่วนคือ 1. train set สำหรับเทรนโมเดล 2. validation set (บางที่เรียกว่า dev set) สำหรับทดสอบและปรับจูนโมเดลตามผลการทดสอบ 3. test set สำหรับการทดสอบผลด้วยโมเดลที่ทำได้ดีที่สุดใน validation set

04. การแบ่ง train-validation-test splits มีข้อควรระวัง สำหรับข้อมูล time series หากเราแบ่งโดยการสุ่มตัวอย่าง จะทำให้โมเดลสามารถ "มองเห็น" ตัวอย่างในอดีตและอนาคตได้ ทำให้ validation/test metric ออกมาดีเกินกว่าความเป็นจริง ที่ถูกต้องเราควรแบ่งตามช่วงเวลา เช่น หากมีข้อมูลรายวัน 10 ปี ให้ใช้ 8 ปีแรกเป็น train set, 1 ปีถัดไปเป็น validation set และ 1 ปีสุดท้ายเป็น test set เป็นต้น

05. บางครั้งคนจะคิดว่าการที่ training loss ลดลงต่ำกว่า validation loss คือจุดที่เริ่มเกิด overfitting และควรหยุดเทรน แต่ในความเป็นจริงแม้ training loss จะลดลงต่ำกว่า validation loss แล้ว หาก validation metric (ไม่ใช่ loss) ยังลดลงอยู่เรื่อยๆ เราสามารถเทรนต่อไปเรื่อยๆเพื่อให้ได้ผลลัพธ์ที่ดีขึ้นได้

06. Transfer learning; การนำโมเดลที่ถูกเทรนด้วยชุดข้อมูลขนาดใหญ่มาปรับใช้ (finetune) กับปัญหาเฉพาะของเรา เช่น ในคาบที่แล้ว เรานำโมเดล reset34 ที่ถูกเทรนด้วยรูปภาพจาก [ImageNet](http://www.image-net.org/) (รวมกว่า 14 ล้านรูป) เพื่อจำแนกรูปภาพ 1,000 ชนิด มา finetune เพื่อจำแนกรูปว่าเป็นหมาหรือแมว

07. fastai ทำการ finetune โมเดลที่ถูกเทรนมาก่อน (pretrained models) โดยการ
  - แทนที่ classifier layer (model head) ของ pretrained model ด้วย classifier layer ที่เราต้องการ เช่น แทนที่ layer ที่ใช้จำแนกรูปภาพ 1,000 ชนิดด้วย layer ที่ใช้จำแนกเพียง 2 ชนิดคือหมาและแมว
  - เทรนโมเดลเฉพาะส่วน classifier layer ที่เพิ่มขึ้นมาใหม่ 1 epoch (1 epoch = 1 รอบการเทรนด้วย train set ทั้งหมดที่มี)
  - เทรนโมเดลทุกส่วนเป็นจำนวน X epoch แล้วแต่จำนวนที่เราใส่ โดย layer แรกๆจะถูก "update ช้ากว่า" layer หลังๆ (learning rate ต่ำกว่า)

08. [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) อธิบายว่าทำไม convolutional neural networks ถึงทำได้ดีกับการจำแนกรูปภาพ หลักการคือ layer แรกๆจะมองเห็นรูปแบบง่ายๆ เช่น เส้น สี และ layer ต่อๆไปก็จะรวมรูปแบบที่ซับซ้อนขึ้นเรื่อยๆ ตั้งแต่รูปเรขาคณิตไปจนถึงลาย อวัยวะ และส่วนต่างๆของสิ่งของ

09. เราสามารถใช้โมเดลจำแนกรูปภาพกับข้อมูลรูปแบบอื่นๆได้
  - เราสามารถเปลี่ยนเสียงเป็น [spectogram](https://en.wikipedia.org/wiki/Spectrogram) แล้วเทรนโมเดลจำแนกรูปภาพเพื่อจำแนกเสียงได้
  - [Splunk and Tensorflow for Security: Catching the Fraudster with Behavior Biometrics](https://www.splunk.com/en_us/blog/security/deep-learning-with-splunk-and-tensorflow-for-security-catching-the-fraudster-in-neural-networks-with-behavioral-biometrics.html) บริษัท Splunk นำรูปแบบการเลื่อนเม้าส์ของผู้ใช้มาแปลงเป็นรูปภาพแล้วใช้โมเดลจำแนกรูปภาพเพื่อจับมิจฉาชีพ
  - ข้อมูลไวรัสแต่ละชนิดสามารถถูกเปลี่ยนเป็นรูปภาพได้

10. ประเภทข้อมูลที่ deep learning ทำได้ดี; สามารถหา pretrained models ส่วนใหญ่ได้จาก model zoo
  - vision; classification, detection
  - text; classification
  - tabular; high cardinality (กรณีมีข้อมูลหลายชนิดในหนึ่งคอลั่มน์ เช่น รหัสไปรษณีย์ รายการสินค้า)
  - recsys; predictions != actions โมเดลอาจจะแนะนำสินค้าชนิดเดียวกันเพื่อให้ลูกค้าซื้อง่ายขึ้น แต่เราอาจจะอยากให้ลูกค้าซื้อชนิดอื่นบ้าง
  - multi-modal; image captioning
  - others; เสียง->รูป, โปรตีน->ข้อความ

11. [High Temperature and High Humidity Reduce the Transmission of COVID-19](https://arxiv.org/abs/2003.05003)
  - งานวิจัยอ้างว่าอุณภูมิยิ่งสูงทำให้อัตราการติดเชื้อยิ่งน้อย โดยใช้ข้อมูลอัตราการติดเชื้อ (R) และอุณหภูมิเป็นองศาเซลเซียส (T) จาก  100 เมืองในประเทศจีน
  - เจเรมีแสดงให้เห็นว่า หากเราสร้าง R และ C จากการสุ่มเลขจาก normal distribution ที่มี mean และ standard deviation ใกล้เคียงกับข้อมูลจริง เราก็จะสามารถเห็นความสัมพันธ์ที่งานวิจัยอ้างได้เช่นเดียวกัน โดยที่ข้อมูลของเรานั้นไม่ได้มีความสัมพันธ์อะไรกันเลย
  - เราจึงจำเป็นต้องมีวิธีพิสูจน์ว่าความสัมพันธ์ที่เราพบในงานวิจัยต่างๆนั้นมีนัยยะสำคัญจริงๆหรือไม่

12. หนึ่งในวิธีการที่จะพิสูจน์ว่าความสัมพันธ์ที่เราพบนั้นมีนัยยะสำคัญจริงๆคือการใช้เทคนิค frequentist hypothesis testing
  - ตั้งสมมุติฐาน null hypothesis ว่าความสัมพันธ์นั้นไม่มีอยู่จริง (อุณภูมิไม่มีความสัมพันธ์กับอัตราการแพร่เชื้อ); ในขณะเดียวกันก็จะมีอีกสมมุติฐานหนึ่งเกิดขึ้น (อุณภูมิมีความสัมพันธ์กับอัตราการแพร่เชื้อ) เรียกว่า alternative hypothesis
  - เก็บข้อมูลตัวแปรที่เกี่ยวข้อง (อุณภูมิและอัตราการแพร่เชื้อในแต่ละเมือง)
  - p-value; หาก null hypothesis เป็นจริง (อุณภูมิไม่มีความสัมพันธ์กับอัตราการแพร่เชื้อ) เราจะมีโอกาสเห็นความสัมพันธ์ในข้อมูลปัจจุบัน (เช่น R = 1.99 - 0.023 * T) หรือรุนแรงกว่า (เช่น R = 1.99 - 200 * T) มากน้อยแค่ไหน
  - ในงานวิจัยและอุตสาหกรรม หาก p-value น้อยกว่าบางค่า (โดยทั่วไปเรียกค่านี้ว่า alpha) เช่น 0.1, 0.05, 0.01 จะถือว่าเราสามารถปฏิเสธ (reject) null hypothesis ได้ และเรียกความสัมพันธ์นั้นว่ามีนัยยะสำคัญทางสถิติ (statistical significance)
  - [American Statistica Association Releases Statment on Statistical Significance and P-values](https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf) กล่าวถึงปัญหาสำคัญสำหรับการใช้ p-value ในการตัดสินว่าความสัมพันธ์นั้นมีนัยยะสำคัญหรือไม่ เช่น p-value ไม่ได้บอกถึงโอกาสที่ alternative hypothesis (เช่น อัตราการติดเชื้อมีความสัมพันธ์กับอุณภูมิ) นั้นจริงหรือไม่จริง, p-value ไม่บอกถึงระดับความรุนแรงของความสัมพันธ์ (เช่น อุณภูมิสูงขึ้นกี่องศา อัตราการติดเชื้อจึงจะลดลงกี่จุด), ยิ่งจำนวนตัวอย่างมากขึ้นก็ยิ่งมีโอกาสที่ p-value จะน้อยลง (ทำให้เกิดการ reject null hypothesis ง่ายขึ้น) และปัญหาอื่นๆอีกมากมาย น้องๆคนไหนสนใจเรื่อง frequentist hypothesis testing สามารถตามไปอ่านต่อได้ที่ [abtestoo](https://github.com/cstorm125/abtestoo) ทั้งในแบบ [slides](https://github.com/cstorm125/abtestoo/blob/master/notebooks/chula_slides.pdf) และ [Jupyter notebook](https://github.com/cstorm125/abtestoo/blob/master/notebooks/frequentist.ipynb)
  - โดยสรุปคือ เจเรมีไม่แนะนำให้ใช้ frequentist hypothesis testing หรือ p-value ในการพิสูจน์ความสัมพันธ์ของตัวแปร

13. [Designing great data products](https://www.oreilly.com/radar/drivetrain-approach-data-products/) เจเรมียกตัวอย่างการใช้ Drivetrain Apporach ในการออกแบบ data product โดยยกตัวอย่างขายประกันรถ [fastbook เวอร์ชั่นตีพิมพ์](https://www.oreilly.com/library/view/deep-learning-for/9781492045519/) มีตัวอย่างการออกแบบ data product ใน appendix สำหรับผู้ที่สนใจ
  - Define Objective; เป้าหมาย เช่น ต้องการขายประกันให้ได้รายได้มากที่สุด
  - Levers; เราเปลี่ยนอะไรได้บ้าง เช่น เพิ่ม/ลดราคาเบี้ยประกัน
  - Data; เราเก็บข้อมูลอะไรได้บ้าง เช่น ยอดขายประกัน โดยเฉพาะหลังเพิ่ม/ลดเบี้ยประกัน
  - Models; โมเดลที่ช่วยทำให้บรรลุเป้าหมาย เช่น โมเดลที่บอกว่าควรตั้งเบี้ยประกันอยู่ที่เท่าไหร่เพื่อให้ได้รายได้มากที่สุด

14. การตัดสินใจนำโมเดลมาใช้จริงต้องคำนึงถึงผลกระทบใน matrix ต่อไปนี้
  - ความสัมพันธ์ที่พบจริง-เราปฏิบัติโดยเชื่อว่าจริง; อากาศร้อนทำให้อัตราการแพร่เชื้อลดจริง เราเตรียมตัวรับมือเฉพาะหน้าหนาว ควบคุมการแพร่เชื้อสำเร็จ
  - ความสัมพันธ์ที่พบปลอม-เราปฏิบัติโดยเชื่อว่าจริง; อากาศร้อนไม่ทำให้อัตราการแพร่เชื้อลดลง เราเตรียมตัวรับมือเฉพาะหน้าหนาว ควบคุมการแพร่เชื้อในหน้าร้อนไม่ได้
  - ความสัมพันธ์ที่พบจริง-เราปฏิบัติโดยเชื่อว่าปลอม; อากาศร้อนทำให้อัตราการแพร่เชื้อลดจริง เราเตรียมตัวรับมือไว้ตลอดปี ควบคุมการแพร่เชื้อได้ แต่เสียทรัพยากรโดยไม่จำเป็นในหน้าร้อน
  - ความสัมพันธ์ที่พบปลอม-เราปฏิบัติโดยเชื่อว่าปลอม; อากาศร้อนไม่ทำให้อัตราการแพร่เชื้อลดลง เราเตรียมตัวรับมือไว้ตลอดปี ควบคุมการแพร่เชื้อได้

15. การตัดสินใจเหล่านี้บ่อยครั้งจำเป็นต้องมี prior belief บางอย่าง เช่น เรามีสมมุติฐานจากไข้หวัดระบาดในอดีตว่ามันจะระบาดหนักกว่าในช่วงหน้าหนาว เราจึงคิดว่าความสัมพันธ์ของอุณภูมิกับอัตราการระบาดมีโอกาสเป็นจริงมากขึ้น

16. สมัคร [Bing Image Search API](https://www.microsoft.com/en-us/bing/apis/bing-image-search-api) สำหรับสร้างชุดข้อมูลรูปภาพของตัวเอง

17. [02_production.ipynb](https://colab.research.google.com/github/fastai/fastbook/blob/master/clean/02_production.ipynb)


``` py
#Downloading images with Bing Image Search API

key = os.environ.get('AZURE_SEARCH_KEY', 'XXX') #API from Azure
results = search_images_bing(key, 'grizzly bear') #download images of 'grizzly bear' from Bing
ims = results.attrgot('content_url') #resulting URLs
len(ims) #default at 150 URLs

#download one image to check
dest = 'images/grizzly.jpg'
download_url(ims[0], dest)
im = Image.open(dest) #open image file
im.to_thumb(128,128) #show at 128x128

#get images for grizzly, black and teddy bears
bear_types = 'grizzly','black','teddy'
path = Path('bears')

#loop to download 150 images for each bear
if not path.exists():
    path.mkdir()
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} bear')
        download_images(dest, urls=results.attrgot('contentUrl'))

#get filenames for downloaded images
fns = get_image_files(path)
fns

#verify if there's any non-images downloaded
failed = verify_images(fns)
failed.map(Path.unlink) #if yes, unlink them from the paths
```

``` py
#Create a dataloader with DataBlock API for creating datasets

#create a dataset using DataBlock object
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock), #(image, label)
    get_items=get_image_files, #function to get example 
    splitter=RandomSplitter(valid_pct=0.2, seed=42), #how to split train-validation-test
    get_y=parent_label, #function to use name of parent folder as label
    item_tfms=Resize(128)  #resize images to 128x128
    )

#create dataloader
dls = bears.dataloaders(path, bs=64) #bs is batch size; how many images to send to train a model at a time

#show a batch
dls.valid.show_batch(max_n=4, nrows=1)

#data augmentation; create new augmentation for a dataset
bears = bears.new(
  item_tfms=Resize(128), #resize
  #perform flip, rotate, zoom, warp, lighting transforms
  batch_tfms=aug_transforms(mult=2) #mult is multiplied to max_rotate,max_lighting,max_warp
  )
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)
```

18. หากใครสมัคร Bing Search API key ไม่ได้ ให้ใช้ `search_images_ddg` หารูปจาก [DuckDuckGo](https://duckduckgo.com/) แทนโยไม่จำเป็นต้องใช้ API key

``` py
def search_images_ddg(key,max_n=200):
     """Search for 'key' with DuckDuckGo and return a unique urls of 'max_n' images
        (Adopted from https://github.com/deepanprabhu/duckduckgo-images-api)
     """
     url        = 'https://duckduckgo.com/'
     params     = {'q':key}
     res        = requests.post(url,data=params)
     searchObj  = re.search(r'vqd=([\d-]+)\&',res.text)
     if not searchObj: print('Token Parsing Failed !'); return
     requestUrl = url + 'i.js'
     headers    = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:71.0) Gecko/20100101 Firefox/71.0'}
     params     = (('l','us-en'),('o','json'),('q',key),('vqd',searchObj.group(1)),('f',',,,'),('p','1'),('v7exp','a'))
     urls       = []
     while True:
         try:
             res  = requests.get(requestUrl,headers=headers,params=params)
             data = json.loads(res.text)
             for obj in data['results']:
                 urls.append(obj['image'])
                 max_n = max_n - 1
                 if max_n < 1: return L(set(urls))     # dedupe
             if 'next' not in data: return L(set(urls))
             requestUrl = url + data['next']
         except:
             pass

urls = search_images_ddg(u'macao parrot', max_n=200)
```

19. ตอบคำถามท้ายบทได้ที่ [aiquizzes](https://aiquizzes.com/howto)
