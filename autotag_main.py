#%%
import os
import io
import json
import torch
from pathlib import Path
import numpy as np
import cv2
import dlib
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from model import get_model
from defaults import _C as cfg
from PIL import Image

#%%
picture_subtitle = ('.jpg','.png','.gif','.jpeg','.bmp')
def judgeing_pic(fn): #判斷檔案是否為圖片	
	for i in picture_subtitle:
		if os.path.splitext(fn)[-1] == i :			
			return(1)
	return(0)
def pureaddtag(somedir,word):
    try:
        #read data----
        json_filename = os.path.join(somedir,'metadata.json')
        with open(json_filename , 'r',encoding = 'utf8') as reader:
            jf = json.loads(reader.read()) #dictionary
        #--------------------------------deal with tags
        if word not in jf['tags']:
            print('-----------tag adding success-----------')
            jf['tags'].append(word)
        #write data----
        #submit = os.path.join(somedir,'submit.json') #for 嘗試用
        submit = json_filename #讀取回原來的檔案路徑
        with io.open(submit, 'w', encoding='utf8') as f: #write出去會有空格，不知道行不行
            json.dump(jf, f, ensure_ascii=False)
    except: pass
def smallest_pic_path(folderpath):
    filelist = os.listdir(folderpath)
    nowleast = 100000000
    picpath = None
    for f in filelist:
        if judgeing_pic(f):
            path = os.path.join(folderpath,f)
            img = Image.open(path)
            picnow = img.size[0]*img.size[1]
            if picnow < nowleast:
                nowleast = picnow
                picpath = os.path.join(folderpath,f)    
    return(picpath)
def yd(img_dir): #find_most_appropriate_img
    smimg_path = smallest_pic_path(img_dir)
    if smimg_path is None: return None
    img = cv2.imdecode(np.fromfile(smimg_path,dtype=np.uint8),cv2.IMREAD_COLOR) #處理中文路徑
    if img is not None:
        h, w, _ = img.shape
        r = 640 / max(w, h)
    else:
        return(None)
    return(cv2.resize(img, (int(w * r), int(h * r)))) #gif會出問題

#%%    
if __name__ == '__main__':
#------------------------------------------------ 不要動 load model
    margin = 0.4
    cfg.freeze()    
    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # load checkpoint
    resume_path = 'epoch044_0.02343_3.9984.pth' 
    if Path(resume_path).is_file():
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(resume_path))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

    if device == "cuda": cudnn.benchmark = True
    model.eval()
    detector = dlib.get_frontal_face_detector()
    img_size = cfg.MODEL.IMG_SIZE
#------------------------------------------------ load path and label 主體
    libpath = 'D:\\Eagle Database\\tag_test.library'
    libpath = os.path.join(libpath,'images')
    filelist = os.listdir(libpath)
    for f in filelist:
        print('now:'+ f)
        img_dir = os.path.join(libpath,f)
        img = yd(img_dir) #回傳cv2.resize後的img物件
        if img is None: continue
        print('-----------img load success-----------')
        with torch.no_grad():
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)
    
            # detect faces using dlib detector
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), img_size, img_size, 3))
    
            if len(detected) > 0: #有偵測到臉
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))
                
                # predict ages
                inputs = torch.from_numpy(np.transpose(faces.astype(np.float32), (0, 3, 1, 2))).to(device)
                outputs = F.softmax(model(inputs), dim=-1).cpu().numpy()
                ages = np.arange(0, 101)
                predicted_ages = (outputs * ages).sum(axis=-1) #type:ndarray
                #加上標籤
                for ages in predicted_ages: #同一張圖很多臉的
                    label = "{}".format(int(ages))
                    #print('ages:'+str(label))
                    if int(label)<13:
                        pureaddtag(img_dir,'childface')
                    elif int(label)<22:
                        pureaddtag(img_dir,'teenface')
                    elif int(label)<30:
                        pureaddtag(img_dir,'20-30')
                    elif int(label)<40:
                        pureaddtag(img_dir,'30-40')  
                    else:
                        pureaddtag(img_dir,'40+')
                
'''            # draw results            
            for i, d in enumerate(detected): #同一張圖很多臉
                label = "{}".format(int(predicted_ages[i]))
                print(label)
                draw_label(img, (d.left(), d.top()), label) #加上去的感覺
        cv2.imshow("result", img)
        key = cv2.waitKey(-1) if img_dir else cv2.waitKey(30)      
'''
