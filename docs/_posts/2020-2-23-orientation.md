---
layout: post
title: คาบที่ 0 - Orientation and Intro to Essential Tools
---

# Orientation (60 mins)

## แนวทางของโครงการ AI Builders
1. **substance over test scores and certificates**; สิ่งที่บ่งบอกถึงความสำเร็จในการเรียนรู้ได้ดีที่สุด (และมีประสบการณ์ในชีวิตจริงมากที่สุด) คือการสร้างโมเดล machine learning ที่แก้ปัญหาได้จริง ไม่ใช่คำพูด, คะแนนสอบ หรือใบประกาศนียบัตรใดอย่างที่เขาหลอกลวง
2. **follow the golden rule**; ปฏิบัติกับเพื่อนร่วมโครงการและ mentor อย่างที่คุณอยากให้พวกเขาปฏิบัติกับคุณ;[ข้อพึงปฏิบัติ](https://vistec-ai.github.io/ai-builders/code-of-conduct/)
3. **sharing is caring**; ชุดข้อมูลเปิดและโปรแกรม open source คือสาธารณูปโภคพื้นฐานของสังคม โค้ดทุกบรรทัด-ข้อมูลทุกชิ้นที่เราสร้างขึ้นอาจจะเป็นส่วนหนึ่งให้ใครนำไปพัฒนาสังคมต่อไปได้อย่างไม่รู้จบ

## การประเมินโครงงานเพื่อจบการศึกษา
AI Builders จะออกใบประกาศนียบัตรจบการศึกษาให้กับผู้เข้าร่วมโครงการที่ส่งโครงงานได้คะแนนอย่างน้อย 70 จาก 100 คะแนนตามเกณฑ์ต่อไปนี้เท่านั้น
1. **problem statement**; เหตุผลในการแก้ปัญหาเชิงธุรกิจ/ชีวิตประจำวันด้วย machine learning - 15 คะแนน
2. **metrics and baselines**; การให้เหตุผลเชื่อมโยงการแก้ปัญหากับตัวชี้วัดที่เลือก / การวัดผลเทียบกับวิธีแก้ปัญหาในปัจจุบัน - 15 คะแนน
3. **data collection and cleaning**; การเก็บและทำความสะอาดข้อมูล - 15 คะแนน
4. **exploratory data analysis**; การทำความเข้าใจข้อมูล - 20 คะแนน
5. **modeling, validation and error analysis**; การทำโมเดล, ทดสอบโมเดล และวิเคราะห์ข้อผิดพลาดของโมเดล - 20 คะแนนฃ
6. **deployment**; การนำโมเดลไปใช้แก้ปัญหาจริง - 15 คะแนน

## งานที่ต้องส่งเพื่อรับการประเมิน
1. Notebooks และ scripts ใน github
2. บล็อกใน Medium.com อธิบายโครงงาน
3. นำเสนอผลงานกับกรรมการ

## กิจวัตรประจำสัปดาห์
1. ดูวิดีโอบทเรียนประจำสัปดาห์จาก https://course.fast.ai/
2. หากมีข้อสงสัยส่งคำถามเพื่อให้ทีมงานตอบในคาบได้ผ่าน https://www.sli.do/
3. เมื่อถึงวันคาบเรียนทุกวันพุธ 
3.1 18:00-19:00 Mentor สรุปบทเรียนและตอบคำถามในบทเรียน
3.2 19:00-20:00 แยกย้ายไปตามกลุ่มเพื่อปรึกษาโครงงาน

## Machine Learning Tasks
ประเภท Machine Learning ที่นิยมและรองรับโดยบทเรียนในโครงการ (fastai, huggingface); ผู้เข้าร่วมโครงการสามารถเลือกทำโครงงานที่เกี่ยวกับโมเดลประเภทดังต่อไปนี้ (หรือประเภทอื่นๆตามแต่ปรึกษากับ mentor) ดูไอเดียโครงงานได้จาก: https://airtable.com/shrTsfrQ9Nc374ly4
1. image - ข้อมูลรูปภาพ
  1.1 image classification (fastai) - แยกแยะรูปภาพ (หมา/แมว ชนิดต้นไม้ ชนิดรถยนต์ ฯลฯ)
  1.2 object detection (fastai) - จับและแยกแยะสิ่งของในรูปภาพ (เลขทะเบียนรถ หน้าคน ตัวคน ฯลฯ)
  1.3 image segmentation (fastai) - แยกแยะแต่ละ pixel ในรูปภาพ (pixel ที่เป็นถนน เซลล์มะเร็ง ฯลฯ)
2. tabular data - ข้อมูลตาราง
  2.1 recommendation (fastai) - แนะนำสิ่งของ (สินค้า หนังสือ ภาพยนต์ ฯลฯ) จากข้อมูลในอดีต (การซื้อขาย การรับชม ฯลฯ)
  2.2 cross-sectional prediction (fastai) - ทำนายจากข้อมูลตาราง (ทำนายว่าลูกค้าจะคลิกโฆษณาหรือไม่ ทีมฟุตบอลทีมไหนจะชนะ ฯลฯ)
  2.3 time-series forecasting (fastai) - ทำนายจากข้อมูลอนุกรมเวลา (ราคาหุ้น ยอดขายรายวัน ฯลฯ)
3. natural language processing - ข้อมูลข้อความ
  3.1 sequence classification (fastai, huggingface) - ทำนายว่าข้อความเป็นประเภทไหน (ดี/แย่ ดาวรีวิว ประเภทข่าว ฯลฯ)
  3.2 token classification (huggingface) - ทำนายว่าแต่ละ "คำ" ในข้อความเป็นประเภทไหน (คำนาม/สรรพนาม/กริยา สถานที่/ชื่อบุคคล/ตัวเลข ฯลฯ)
  3.3 question answering (huggingface) - หาคำตอบของคำถามจากบริบท
  3.4 information retrieval (huggingface) - ค้นหาข้อความ (ค้นหาสินค้า จับคู่ประโยค ฯลฯ)
  3.5 text generation (fastai, huggingface) - สร้างข้อความ (แต่งเพลง แต่งนิยาย เขียนโฆษณา ฯลฯ)
  3.6 automatic speech recognition (huggingface) - เปลี่ยนเสียงเป็นข้อความ

# Introduction to Essential Tools (60 mins)
เรียนเครื่องมือพื้นฐานในการเขียนโปรแกรมไปกับ [ลุงวิศวกรสอนคำนวน](https://www.facebook.com/UncleEngineer/)
1. [Jupyter Notebooks](https://jupyter.org/) and [Google Colaboratory](https://colab.research.google.com/)
2. [Bash commands](https://www.gnu.org/software/bash/)
3. [มาเริ่มใช้ Git กัน [part 1]](https://tupleblog.github.io/use-git-part1/)

## แบบฝึกหัด
ทำแบบฝึกหัดได้ที่ https://github.com/vistec-AI/ai-builders-orientation

