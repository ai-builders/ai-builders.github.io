---
layout: post
title: ปรับพื้นฐานก่อนเรียน fastai
---

การเรียน [fastai](https://course.fast.ai/) ต้องใช้ทักษะการเขียนโปรแกรมภาษา Python
และการใช้ GPU ซึ่งสามารถใช้ได้จาก Google Colab ในบทความนี้เราจึงรวบรวมลิงค์และบทเรียนเพื่อปรับพื้นฐาน
ให้น้องๆสามารถใช้งาน Python ได้บน Jupyter notebook ดังต่อไปนี้

## ติดตั้ง Python โดยใช้ Anaconda

Anaconda เป็น distribution ของ Python ซึ่งรวมไลบรารี่ที่เกี่ยวกับการใช้งานทางด้านวิทยาศาสตร์ข้อมูล
ก่อนที่จะเริ่มเรียน เราแนะนำให้ลงโปรแกรม Python โดยใช้
Anaconda ซึ่งสามารถลดระยะเวลาการลงไลบรารี่ต่างๆไปได้มาก

หากน้องๆอยากจะลองใช้ Python ในเครื่องของตัวเองโดยที่ไม่ต้องต่ออินเตอร์เน็ตใช้งาน Google Colab
ก็สามารถดาวน์โหลด Anaconda มาใช้งานในเครื่องตัวเองได้จาก [Anaconda Installers](https://www.anaconda.com/products/individual)

## ใช้งาน  Graphical processing unit (GPU) ผ่าน Google Colab

GPU มีความสำคัญต่อการเทรนโมเดล Deep learning ที่น้องๆจะได้เรียนกับ fast.ai แต่เนื่องจากหลายๆคนอาจจะไม่ได้มี GPU
ติดอยู่กับคอมพิวเตอร์และการเช่าใช้ GPU อาจจะมีราคาแพง เราจึงแนะนำให้ใช้ GPU ได้ฟรีบน Google Colab

น้องๆสามารถใช้งาน Google Colab ได้ฟรีที่ [https://colab.research.google.com/](https://colab.research.google.com/)
โดยหลังจากที่เข้าไปแล้ว เราจะสามารถอัพโหลดไฟล์หรือใส่ Github URL เข้าไปเพื่อเปิดใช้งาน Google Colab ได้

อย่างเช่นในคาบที่ 1 ของ Fast AI เราจะใช้ Jupyter notebook ของหนังสือ fastai จาก
[github.com/fastai/fastbook](https://github.com/fastai/fastbook/) ให้เราเลือกช่อง Github แล้ววาง URL
ของ Jupyter Notebook เข้าไปได้เลย

<figure align="center">
  <img src="{{ site.baseurl }}/images/colab-url.png" style="width: 400px;"/>
  <figcaption>ใส่ URL จาก fastai/fastbook เข้าไปใน Google Colab เพื่อเปิด Jupyter Notebook</figcaption>
</figure>

เมื่อเราเปิด Jupyter Notebook มาแล้วจะสามารถรันโค้ดได้โดยกด `shift + enter` เพื่อรันแต่ละเซลล์ (ช่อง) ของ Notebook
นอกจากนั้นเราสามารถเปิดใช้งาน GPU ได้ด้วยโดยให้เรากดที่ `Edit > Notebook Settings` แล้วเลือกใช้งาน GPU ฟรีจาก
Google Colab

<figure align="center">
  <img src="{{ site.baseurl }}/images/select-gpu.png" style="width: 400px;"/>
  <figcaption>เลือกใช้งาน GPU บน Google Colab (`Edit > Notebook Settings`)</figcaption>
</figure>

เพียงเท่านี้น้องๆก็จะสามารถรันโค้ดใน Lesson 1 ของ fastai ได้โดยไม่ต้องมี GPU เป็นของตัวเอง

## เนื้อหาที่จำเป็นสำหรับ Take-home Extrance Exam
เราจะมี Pre-course Workhop (3 ชั่วโมง) 2 ครั้งเพื่อปรับพื้นฐานสำหรับน้องๆที่ไม่เคยเรียน Python, linear algebra/numpy หรือ pandas มาก่อน เราหวังให้น้องๆใช้ความรู้ที่เรียนจาก workshop และ notebook เหล่านี้เพื่อทำ Take-home Extrance Exam ส่งมากับใบสมัคร
* [Introduction to Python](https://github.com/vistec-AI/ai-builders/blob/main/notebooks/ai_builder_intro_python.ipynb)
* [
Introduction to Numpy and Basic Linear Algebra Operations](https://github.com/vistec-AI/ai-builders/blob/main/notebooks/ai_builder_numpy.ipynb)
* [Introduction to Pandas](https://github.com/vistec-AI/ai-builders/blob/main/notebooks/ai_builder_pandas.ipynb)

## ฝึกใช้งาน Python เบื้องต้น

Python เป็นโปรแกรมที่นิยมใช้งานในวิทยาศาสตร์ข้อมูลและการสร้างโมเดล Deep learning มากที่สุดในปัจจุบัน
การฝึกใช้งาน Python เบื้องต้นทำให้น้องๆพอเห็นภาพการใช้งานของภาษา Python จึงจำเป็น

เราได้รวบรวมลิงค์การสอนใช้โปรแกรม Python เบื้องต้นมาดังนี้

* พื้นฐานการเขียนโปรแกรมด้วย Python
  * [NeuroMatch Academy Python Tutorial](https://github.com/NeuromatchAcademy/course-content#python-for-nma-workshop)
  * [Learn Python the Hard Way](https://github.com/ubarredo/LearnPythonTheHardWay)
* จัดการข้อมูลด้วยไลบรารี่ Pandas
  * [Kaggle Pandas lessons](https://www.kaggle.com/learn/pandas)

## พื้นฐานคณิตศาสตร์ที่ใช้ในงาน Data Science และ AI

เราได้รวมลิงค์ของเนื้อหาที่ใช้สำหรับงาน Data Science และ AI ด้านล่าง
น้องๆอาจจะไม่ต้องจำได้ทั้งหมด แต่ว่าสามารถใช้ดูเป็น reference ได้

* พีชคณิจเชิงเส้น (Linear Algebra)
  * [Illustrated Introduction to Linear Algebra using NumPy](https://medium.com/@kaaanishk/illustrated-introduction-to-linear-algebra-using-numpy-11d503d244a1)
  * [Stanford CS 229 Linear Algebra and Calculus refresher](https://stanford.edu/~shervine/teaching/cs-229/refresher-algebra-calculus)
  * [fastai Computational Linear Algebra for Coders](https://github.com/fastai/numerical-linear-algebra)
* พื้นฐานความน่าจะเป็นและสถิติ
  * [Stanford CS229 Probabilities and Statistics refresher](https://stanford.edu/~shervine/teaching/cs-229/refresher-probabilities-statistics)
* พื้นฐานแคลคูลัส เฉพาะเรื่อง derivatives, chain rule และ partial derivatives
  * [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/index.html)
  * [Stanford CS229 Linear Algebra and Calculus refresher](https://stanford.edu/~shervine/teaching/cs-229/refresher-algebra-calculus)
* Version control โดยใช้ Git และ Github เบื้องต้น
  * [มาเริ่มใช้ Git กัน [part 1] ](https://tupleblog.github.io/use-git-part1/)