import pandas as pd
import os
from PIL import Image
OUTPUT="data/trimmed/"
SOURCE="CUB_200_2011/images/"
os.system("rm -Rf data/")
os.mkdir("data")
os.mkdir("data/trimmed")

#Get Files to be trimmed
with open("CUB_200_2011/images.txt","rt") as imgfile:
    images=[i.strip().split(' ') for i in imgfile.readlines()]
with open("CUB_200_2011/bounding_boxes.txt","rt") as boxfile:
    box=[ i.strip().split() for i in boxfile.readlines()]
df_image=pd.DataFrame.from_records(images,columns=['id','file'])
df_image['Class']=df_image.file.apply(lambda x: (x.split('/')[0])[4:])
df_image['File']=df_image.file.apply(lambda x: (x.split('/')[1]))

df_box=pd.DataFrame.from_records(box,columns=['id','startX','startY','endX','endY'])
df_image=df_image.merge(df_box,on='id')

for file in df_image.iterrows():
    if not os.path.exists(OUTPUT+file[1].Class):
        os.mkdir(OUTPUT+file[1].Class)
    im = Image.open(SOURCE+file[1].file)
    box = (int(float(file[1].startX)),
       int(float(file[1].startY)),
       int(float(file[1].endX))+int(float(file[1].startX)),
       int(float(file[1].startY))+int(float(file[1].endY)))
    region = im.crop(box)
    region.save(OUTPUT+file[1].Class+"/"+file[1].File)
print("Image trimming completed")
