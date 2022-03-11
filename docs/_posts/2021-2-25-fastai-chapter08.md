---
layout: post
title: สรุป course.fast.ai (part1 v4) คาบที่ 8
---

คาบสุดท้ายสอนเกี่ยวกับโมเดลประมวลผลภาษาหรือ NLP อธิบายว่าการทำ self-supervised learning ก่อนที่จะทำ transfer learning ไปสู่ task ที่ต้องการอย่าง sentiment analysis (sequence classification) มีประโยชน์อย่างไร สำหรับใครที่อยากอ่านงานวิจัยไปหาอ่านได้ที่ [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)

ครึ่งแรกของบนเป็นการสอนทำ text classification ด้วย fastai ตาม `10_nlp.ipynb` ครึ่งหลังเป็นการสอนสร้าง language ขึ้นมาเองจาก 0 ตาม `12_nlp_dive.ipynb`

01. ULMFit ทำการ finetune 2 ครั้งคือ

- finetune language model ด้วยข้อความที่อยู่ใน domain ที่ต้องการ; เพื่อปรับ parameters ให้เข้ากับ domain ก่อน
- finetune classifier head เพื่อทำ task ที่ต้องการ

02. Text processing เพื่อสร้าง input ให้ neural networks

- สร้างลิสต์ของ token เรียกว่า vocab ซึ่งประกอบด้วยคำ (words), คำย่อย (subwords) เช่น พยางค์ (syllable), หรือตัวอักษร (characters) ที่เราต้องการ; token ที่ไม่อยู่ใน vocab อาจจะให้แทนด้วย unknown token
- Tokenization; ตัดข้อความให้เป็น token
- Numericalization; เปลี่ยน token เป็น index ใน vocab
- สร้าง embedding matrix โดยแถวคือ vocab และคอลัมน์คือ embeddings เช่น หากมี vocab 50,000 tokens แต่ละ embedding มี 400 มิติก็จะได้ embedding matrix ขนาด 50,000 * 400
- นำตัวอย่างข้อความทั้งหมดมาต่อกันเป็น string ยาวมากหนึ่งอัน 
- สร้าง independent variable ด้วย token แรกถึง token ก่อนสุดท้าย; สร้าง dependent variable ด้วย token ที่สองจนถึง token สุดท้าย กล่าวคือจาก ['A','B','C','D'] สร้าง ['A','B','C'] เพื่อทำนาย ['B','C','D']

03. Tokenization ใน fastai

```py
txt = 'The U.S. dollar $1 is $1.00.'
spacy = WordTokenizer() #class ที่ทำหน้าที่ tokenizer ภาษาอังกฤษจาก spacy
tkn = Tokenizer(spacy) #สร้าง Tokenizer object ของ fastai
print(coll_repr(tkn(txt), 31)) #print 31 อันจากข้อความที่ถูก tokenize
#xxbos คือ beginning of sentence token (เพราะเราจะเอาข้อความาต่อกันยาวๆเวลาเทรน ต้องมีไว้เพื่อให้รู้ว่าตรงไหนคือเริ่มต้นประโยค)
#xxmaj คือ capitalization token บอกว่า token ต่อไปถูก capitalize ตัวแรก (ในที่นี้คือ The)
#xxup คือ all-caps token บอกว่า token ต่อไปเป็น uppercase ทั้งหมด (ในที่นี้คือ u.s)
#>> ['xxbos','xxmaj','the','xxup','u.s','.','dollar','$','1','is','$','1.00','.']

#วิธีการสร้าง fastai tokenizer ด้วย PythaiNLP (หรือจริงๆ tokenizer อะไรก็ได้)
#สร้าง class สำหรับ tokenizer ให้ผลเป็น list
from pythainlp.tokenize import word_tokenize
class NewMMTokenizer:
    def __init__(self, tok_func):
        self.tok_func = tok_func
    def __call__(self,items):
        return (self.tok_func(t) for t in items) 

#สร้าง newmm object จากคลาสที่เราสร้าง
newmm = NewMMTokenizer(word_tokenize)
#สร้าง tokenizer เหมือนภาษาอังกฤษ
tkn = Tokenizer(newmm)
tkn('สวัสดีครับพี่น้อง')
#>> ['xxbos', ' ', 'สวัสดี', 'ครับ', 'พี่น้อง']
```

04. Special tokens ของ fastai

- UNK (`xxunk`); unknown token หรือ token ที่ไม่อยู่ใน vocab
- PAD (`xxpad`); padding token ไว้เติมให้ขนาด sequence ครบ
- BOS (`xxbos`); beginning of sentence token (เพราะเราจะเอาข้อความาต่อกันยาวๆเวลาเทรน ต้องมีไว้เพื่อให้รู้ว่าตรงไหนคือเริ่มต้นประโยค)
- EOS (`xxeos`); end of sentence token
- FLD (`xxfld`); field token สำหรับ input ที่มีหลาย field
- TK_REP (`xxrep`); repetitive character token สำหรับ token ที่มีตัวอักษรซ้ำเยอะๆ เช่น `goooooo`
- TK_WREP (`xxwrep`); repetitive word token สำหรับ token ซ้ำกันเยอะๆ เช่น `go go go go`
- TK_UP (`xxup`); all-caps token บอกว่า token ต่อไปเป็น uppercase ทั้งหมด
- TK_MAJ (`xxmaj`); capitalization token บอกว่า token ต่อไปถูก capitalize ตัวแรก

05. ทบทวนกฎทำความสะอาดข้อความของ fastai

- `fix_html`:: Replaces special HTML characters with a readable version
- `replace_rep`:: Replaces any character repeated three times or more with a special token for repetition (`xxrep`), the number of times it's repeated, then the character
- `replace_wrep`:: Replaces any word repeated three times or more with a special token for word repetition (`xxwrep`), the number of times it's repeated, then the word
- `spec_add_spaces`:: Adds spaces around / and #
- `rm_useless_spaces`:: Removes all repetitions of the space character
- `replace_all_caps`:: Lowercases a word written in all caps and adds a special token for all caps (`xxup`) in front of it
- `replace_maj`:: Lowercases a capitalized word and adds a special token for capitalized (`xxmaj`) in front of it
- `lowercase`:: Lowercases all text and adds a special token at the beginning (`xxbos`) and/or the end (`xxeos`)

06. Subword tokenization ส่วนใหญ่สร้าง vocab จากการดูว่ากลุ่มตัวอักษรไหนโผล่มาพร้อมกันบ่อยที่สุดจากชุดข้อมูลใช้เทรน tokenizer โดย fastai ใช้ [SentencePiece](https://github.com/google/sentencepiece) เป็น default สำหรับ subword tokenizer; ลองลดหรือเพิ่ม vocab size ดูจะเห็นว่ายิ่ง vocab size ต่ำ subword แต่ละอันก็จะตัวอักษรน้อยลง

```py
def subword(sz):
    sp = SubwordTokenizer(vocab_sz=sz) #create sentencepiece tokenizer with vocab size of `sz`
    sp.setup(txts) #เทรน sentencepiece
    return '|'.join(first(sp([txt]))[:40]) #รีเทิร์น 40 subwords แรก

# ▁ ในที่นี้แทน space เพราะ subword จะรวบ space เข้ามาด้วย
subword(200)
# >> ▁The|▁|U|.|S|.|▁do|l|l|ar|▁|$|1|▁is|▁|$|1|.|0|0|.
subword(20000)
# >> ▁The|▁U|.|S|.|▁dollar|▁$1|▁is|▁$1|.00|.
```

07. [ข้อแนะนำจากผู้สรุป] พฤติกรรมการรวบ space ของ SentencePiece เป็นหนึ่งในเหตุผลที่เวลาเราใช้ subword tokenization ในภาษาไทยควรแทนช่องว่างด้วย space token เช่น `<_>` ของ [WangchanBERTa](https://arxiv.org/abs/2101.09635)

08. Jeremy บอกว่าเขาเคยคิดว่าเราเทรนด้วย vocab เฉพาะสำหรับชุดข้อมูลขนาดใหญ่แต่ละ domain แต่ที่จริงแล้ว vocab ที่เทรนจากชุดข้อมูลใหญ่อย่าง Wikipedia ก็ค่อนข้างเวิร์คกับชุดข้อมูลอื่นๆหลังจากการ finetune อีกรอบเพื่อปรับให้เข้ากับ domain

09. Numericalization; เปลี่ยนแต่ละ token เป็น index ใน vocab

```py
toks = tkn(txt)
toks200 = txts[:200].map(tkn) #tokens

num = Numericalize()
num.setup(toks200) #set up using the tokens
nums = num(toks)[:20]; nums #numericalized; เปลี่ยนจาก token เป็น index
' '.join(num.vocab[o] for o in nums) #เปลี่ยนกลับจาก index เป็น token
```

10. การสร้าง mini-batch ของข้อมูลชนิดข้อความใน fastai เพื่อเทรน language model ทำโดยการ

- สลับตัวอย่างทั้งหมด เช่นเดียวกับการสลับรูปภาพ
- ต่อตัวอย่างทั้งหมดเป็น text stream ใหญ่หนึ่งอัน
- หั่น text stream นั้นตาม batch size * sequence length
- ลำดับของ token จะเรียงตามลำดับ:

```py
#batch size = 3
#sequence length = 2
['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R']

#mini-batches จะเป็น
#batch 0
[
'A','B'
'G','H'
'M','N'
]

#batch 1
[
'C','D'
'I','J'
'O','P'
]

#batch 2
[
'E','F'
'K','L'
'Q','R'
]
```
- โดยคู่ X, y จะเป็น

```py
#batch 0; X
[
'A','B'
'G','H'
'M','N'
]

#batch 0; y
[
'B','C'
'H','I'
'N','O'
]

```

11. เราสามารถ finetune language model ที่ถูกเทรนมาก่อนด้วยข้อมูลขนาดใหญ่ เช่น Wikipedia กับข้อมูลของเราได้เพื่อให้โมเดลเข้าใจ domain เรามากขึ้น ด้วยวิธีเดียวกับที่เรา finetune โมเดลที่ถูกเทรนบน ImageNet เพื่อให้เข้าใจรูปภาพของเราดีขึ้น

12. เราสามารถใช้ language model ที่ถูก finetune แล้วสร้างข้อความใหม่ขึ้นมาได้ด้วย (text generation) แต่จะไม่ดีเท่าโมเดลที่ถูกเทรนมาเพื่อทำสิ่งนี้โดยเฉพาะ เช่น [ตระกูล GPT](https://openai.com/blog/openai-api/) การที่โมเดลเราทำแบบนี้ได้พิสูจน์ได้ในระดับนึงว่าโมเดล "เข้าใจภาษา" ในแง่เดียวกับที่โมเดลที่ถูก pretrain บน ImageNet เข้าใจรูปแบบของภาพ

13. ส่วนใหญ่ในการทำ NLP ด้วย deep learning เราจะไม่นิยมใช้เทคนิคทำความสะอาดข้อความ เช่น stemming, lemmatization เพราะจะทำให้สูญเสียข้อมูลเกี่ยวกับ token (running, run, runs จะกลายเป็น run) เทคนิคเหล่านี้มีประโยชน์กับโมเดล bag-of-words แต่มีผลเสียกับ deep learning ที่ใช้ embeddings ของ tokens เป็น input

14. ข้อควรระวังเวลาสร้าง DataLoader สำหรับ finetuning for classification คือเราต้องใช้ vocab เดียวกับ language model ที่เรา finetune มากับ domain ที่เราต้องการ (`dls_lm.vocab`)

```py
dls_clas = DataBlock(
    blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab),CategoryBlock),
    get_y = parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, path=path, bs=128, seq_len=72)
```

15. เวลาเรา finetune สำหรับ classification เราจะเจอปัญหา sequence length ไม่เท่ากัน (บางข้อความสั้นเกินไป-ยาวเกินไป) fastai จัดการปัญหานี้ให้เราโดยอัตโนมัติโดยการเพิ่ม padding token ให้ทุก sequence ใน mini-batch มีความยาวเท่ากัน เรียกว่า dynamic padding คือไม่ได้ pad ให้ทุก mini-batch มี sequence ที่มีขนาดเท่ากัน แต่พยามเอา sequence ที่มีความยาว "พอๆกัน" มาอยู่ใน mini-batch เดียวกันแล้วค่อย pad ให้ยาวเท่า sequence ที่ยาวที่สุดในแต่ละ mini-batch นั้น

16. Jeremy พบว่าเวลาเราเทรนโมเดล NLP แทนที่เราจะ unfreeze ทุก layer ทีเดียวเหมือนตอนทำโมเดล vision หากเราค่อยๆ unfreeze จาก layer หลังๆก่อน (layer ที่ใกล้ classifer head ก่อน) จะทำให้ได้ผลดีขึ้น เรียกว่า gradual unfreezing

17. หากเราเทรน ULMFit 2 โมเดล โมเดลนึงดูข้อความจากซ้ายไปขาว อีกโมเดลดูข้อความจากขวาไปซ้าย แล้วเอา predictions มาเฉลี่ยกันเราจะได้ state-of-the-art results บน IMDb 3 ปีที่แล้ว ที่เพิ่งถูกทำลายไปเมื่อไม่นานมานี้ด้วยการทำ backtranslation เพื่อเพิ่มข้อมูลเทรน แต่จริงๆแล้ว review classification ไม่ใช่งานที่ยากนัก เราทำได้ 92% หรือดีกว่าโดยการเทรนแค่ classifier head ด้วยซ้ำ

18. Jeremy กล่าวถึงอันตรายด้าน disinformation ด้วย language model ที่สร้างข้อความได้ดีในบทความ [More than a Million Pro-Repeal Net Neutrality Comments were Likely Faked](https://hackernoon.com/more-than-a-million-pro-repeal-net-neutrality-comments-were-likely-faked-e9f0e3ed36a6) ผู้เขียนพบว่าจาก 22M ข้อความที่ถูกส่งให้ FCC เกี่ยวกับประเด็น net neutrality ของอเมริกา มีเพียง 800k ข้อความที่เป็นข้อความที่แตกต่างจริงๆ หมายความว่าข้อความส่วนใหญ่ที่ถูกส่งไปมีความเป็นไปได้สูงที่จะถูกสร้างขึ้นโดย language model เพื่อแทรกแซงการตัดสินใจเกี่ยวกับ net neutrality

19. Language model from scratch; สร้าง DataLoader จากชุดข้อมูล `HUMAN_NUMBERS`

```py
#a long text with numbers in English from one to nine thousand nine hundred ninety nine
#"one . two . three . four . five . six . seven . eight . nine . ten . eleven . twelve . thirteen . ..."
text = ' . '.join([l.strip() for l in lines])
tokens = text.split(' ')

#get vocab list
vocab = L(*tokens).unique()
#['one','.','two','three','four','five','six','seven','eight','nine'...]

#numericalize with a dictionary
word2idx = {w:i for i,w in enumerate(vocab)}
nums = L(word2idx[i] for i in tokens)

#create X,y pairs with 3 tokens to predict the next token
seqs = L((tensor(nums[i:i+3]), nums[i+3]) for i in range(0,len(nums)-4,3))
#[(['one', '.', 'two'], '.'),(['.', 'three', '.'], 'four'),(['four', '.', 'five'], '.'),(['.', 'six', '.'], 'seven')]
#[(tensor([0, 1, 2]), 1),(tensor([1, 3, 1]), 4),(tensor([4, 1, 5]), 1),(tensor([1, 6, 1]), 7)]

#create dataloader with batch size = 64
#validation split at 80/20
bs = 64
cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(seqs[:cut], seqs[cut:], bs=64, shuffle=False) #do not shuffle

#group_chunks ช่วยสร้าง mini-batch ตามวิธีข้างต้น
#index 0 เป็น (tensor([0,1,2]), 1)
#index 1 เป็น (tensor([11,  1,  2]), 28)
#index 64 (batch size) เป็น (tensor([1,3,1]), 4)
#วิธีเช็คว่าเรียง seq เป็น mini-batch ยังไง; หา seq[1] หรือ (tensor([1,3,1]),4)
#dls.train_ds.argwhere(lambda x: (x[1]==4)&(x[0][0].item()==1)&(x[0][1].item()==3)&(x[0][2].item()==1))

def group_chunks(ds, bs):
    m = len(ds) // bs
    new_ds = L()
    for i in range(m): new_ds += L(ds[i + m*j] for j in range(bs))
    return new_ds

cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(
    group_chunks(seqs[:cut], bs), 
    group_chunks(seqs[cut:], bs), 
    bs=bs, drop_last=True, shuffle=False)
```

20. Recurrent neural network สำหรับ language model

```py
#RNN architecture for language model
class LMModel1(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden) #input layer; takes (batch size, 1) returns (batch size, n_hidden)
        self.h_h = nn.Linear(n_hidden, n_hidden) #hidden layer; takes (batch size, n_hidden) returns (batch size, n_hidden)
        self.h_o = nn.Linear(n_hidden,vocab_sz) #output layer; takes (batch size, n_hidden) returns (batch size, vocab size)
        
    def forward(self, x):
        #x: batch size * 3 tokens
        h = F.relu(self.h_h(self.i_h(x[:,0]))) #put first token through i_h, h_h and ReLU
        h = h + self.i_h(x[:,1]) #put the second token through i_h, then plus with result from last step
        h = F.relu(self.h_h(h)) #put result of last step through ReLU
        h = h + self.i_h(x[:,2]) #put the third token through i_h, then plus with result from last step
        h = F.relu(self.h_h(h)) #put result of last step through ReLU
        return self.h_o(h) #put result of last step through h_o; predicts which token comes next

#Refactored RNN ด้วย loop
class LMModel2(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)  
        self.h_h = nn.Linear(n_hidden, n_hidden)     
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        
    def forward(self, x):
        h = 0 #initialize h as 0
        for i in range(3):
            h = h + self.i_h(x[:,i])
            h = F.relu(self.h_h(h))
        return self.h_o(h)

#รักษา hidden state ไว้ในแต่ละครั้งที่ทำ forward()
class LMModel3(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)  
        self.h_h = nn.Linear(n_hidden, n_hidden)     
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        self.h = 0
        
    def forward(self, x):
        for i in range(3):
            self.h = self.h + self.i_h(x[:,i])
            self.h = F.relu(self.h_h(self.h))
        out = self.h_o(self.h)
        #.detach() เพื่อให้ Pytorch คำนวณ gradients แยกเป็นครั้งๆที่ forward ถูกเรียก
        #ไม่งั้น pytorch จะคำนวร gradients ตั้งแต่ token แรกยัน token สุดท้ายใน dataset
        self.h = self.h.detach() 
        return out
    
    def reset(self): self.h = 0
```

21. สร้าง (X,y) ใหม่โดยให้มี dependent variable ทุก step เหมือนกับที่ fastai ทำ

```py
sl = 16
seqs = L((tensor(nums[i:i+sl]), tensor(nums[i+1:i+sl+1]))
         for i in range(0,len(nums)-sl-1,sl))
cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(group_chunks(seqs[:cut], bs),
                             group_chunks(seqs[cut:], bs),
                             bs=bs, drop_last=True, shuffle=False)

dls.train_ds[0]
# >>(tensor([0, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1]),
# >>tensor([1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9]))
```

22. ปรับ language model เพื่อให้สามารถสร้าง prediction ได้ทุก time step (token) ของ sequence

```py
class LMModel4(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)  
        self.h_h = nn.Linear(n_hidden, n_hidden)     
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        self.h = 0
        
    def forward(self, x):
        outs = [] #รับ output จากทุก token
        for i in range(sl):
            self.h = self.h + self.i_h(x[:,i])
            self.h = F.relu(self.h_h(self.h))
            outs.append(self.h_o(self.h))
        self.h = self.h.detach()
        return torch.stack(outs, dim=1) #stack output เข้าด้วยกัน
    
    def reset(self): self.h = 0
```

23. เราสามารถต่อ output จากแต่ละ time step (token) RNN อันนึงไปยังอีกอันนึงได้เพื่อให้ได้ neural network ที่ซับซ้อนขึ้นเรียกว่า multilayer RNN

```py
class LMModel5(Module):
    def __init__(self, vocab_sz, n_hidden, n_layers):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.rnn = nn.RNN(n_hidden, n_hidden, n_layers, batch_first=True) #เลือกได้ว่าอยากได้กี่ layer ซ้อนกัน
        self.h_o = nn.Linear(n_hidden, vocab_sz)
        self.h = torch.zeros(n_layers, bs, n_hidden)
        
    def forward(self, x):
        res,h = self.rnn(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(res)
    
    def reset(self): self.h.zero_()
```

24. Exploding/vanishing gradients เกิดจากการที่ gradients ถูก mulitply-add ซ้ำไปเรื่อยๆในหลายๆ loop และ layer ทำให้ในที่สุด gradients มีขนาดใหญ่หรือเล็กเกินไป ทำให้ไม่สามารถปรับ parameters ได้อย่างมีประสิทธิภาพอีก; ใน fastai เราสามารถดูว่า activations/gradients ของเราเป็นอย่างไรได้ด้วย [ActivationStats](https://docs.fast.ai/callback.hook.html#ActivationStats)

25. LSTM เป็น RNN รูปแบบที่ถูกสร้างขึ้นมาเพื่อแก้ปัญหา exploding/vanishing gradients นี้; เราจะยังไม่ลงรายละเอียดเกี่ยวกับ LSTM ใน part 1 แต่ไอเดียคือ LSTM มี parameters (gates) ที่บอกว่าโมเดลควร "จำ" state ต่างๆมากแค่ไหน ทำให้ค่า gradients ไม่ใหญ่หรือเล็กเกินไป

```py
class LSTMCell(Module):
    def __init__(self, ni, nh):
        self.forget_gate = nn.Linear(ni + nh, nh)
        self.input_gate  = nn.Linear(ni + nh, nh)
        self.cell_gate   = nn.Linear(ni + nh, nh)
        self.output_gate = nn.Linear(ni + nh, nh)

    def forward(self, input, state):
        h,c = state
        h = torch.cat([h, input], dim=1)
        forget = torch.sigmoid(self.forget_gate(h))
        c = c * forget
        inp = torch.sigmoid(self.input_gate(h))
        cell = torch.tanh(self.cell_gate(h))
        c = c + inp * cell
        out = torch.sigmoid(self.output_gate(h))
        h = out * torch.tanh(c)
        return h, (h,c)

#refactored version
class LSTMCell(Module):
    def __init__(self, ni, nh):
        self.ih = nn.Linear(ni,4*nh)
        self.hh = nn.Linear(nh,4*nh)

    def forward(self, input, state):
        h,c = state
        # One big multiplication for all the gates is better than 4 smaller ones
        gates = (self.ih(input) + self.hh(h)).chunk(4, 1)
        ingate,forgetgate,outgate = map(torch.sigmoid, gates[:3])
        cellgate = gates[3].tanh()

        c = (forgetgate*c) + (ingate*cellgate)
        h = outgate * c.tanh()
        return h, (h,c)
```

26. ข้อต่อไปคือเทคนิค regularization หรือทำให้โมเดลไม่ overfit ที่ถูกนำเสนอในงานวิจัย [AWD LSTM](https://arxiv.org/abs/1708.02182)

27. Dropout เป็นหนึ่งใน regularization โดยการสุ่มเปลี่ยน activation ในแต่ละ layer บางอันเป็น 0 ด้วย % ที่ต้องการ

```py
class Dropout(Module):
    def __init__(self, p): self.p = p
    def forward(self, x):
        if not self.training: return x #ถ้าไม่ได้เทรนอยู่ ไม่ต้องทำ dropout
        mask = x.new(*x.shape).bernoulli_(1-p) #สุ่มว่าแต่ละ activation อันไหนที่เราจะเปลี่ยนเป็น 0
        return x * mask.div_(1-p)
```

28. Activation regularization คือการเพิ่ม sum of squares ของ activations สุดท้ายเข้าไปใน loss เพื่อให้โมเดลไม่สร้าง parameters ที่เฉพาะเจาะจงเกินไป

```py
loss += alpha * activations.pow(2).mean()
```

29. Temporal activation regularization คือการเพิ่ม sum of squares ของความแตกต่างระหว่าง activations ที่เหลื่อมกัน X step เพื่อให้โมเดลสร้าง activation คล้ายๆกันหากห่างกันไม่เกิน X steps; เราคิดว่าโมเดลควรทำแบบนี้เพราะข้อมูลของเราเรียงกันเป็นลำดับ

```py
loss += beta * (activations[:,1:] - activations[:,:-1]).pow(2).mean()
```

30. Weight-tying คือการเซ็ต parameters ของ input layer (Embeddings ที่มี dimension คือ (vocab size, n_hidden)) ให้เท่ากับ output layer (Linear ที่มี dimension (n_hidden, vocab size)) กล่าวคือให้ทั้งสอง layer ใช้ weight ร่วมกัน เพราะทั้งสอง layer แทบจะทำสิ่งเดียวกันอยู่แล้วคือการเปลี่ยน token เป็นตัวเลข

```py

class LMModel7(Module):
    def __init__(self, vocab_sz, n_hidden, n_layers, p):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.rnn = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True)
        self.drop = nn.Dropout(p)
        self.h_o = nn.Linear(n_hidden, vocab_sz)
        self.h_o.weight = self.i_h.weight #weight-tying
        self.h = [torch.zeros(n_layers, bs, n_hidden) for _ in range(2)]
        
    def forward(self, x):
        raw,h = self.rnn(self.i_h(x), self.h)
        out = self.drop(raw)
        self.h = [h_.detach() for h_ in h]
        return self.h_o(out),raw,out
    
    def reset(self): 
        for h in self.h: h.zero_()
```

31. ตอบคำถามท้ายบทได้ที่ [aiquizzes](https://aiquizzes.com/howto)