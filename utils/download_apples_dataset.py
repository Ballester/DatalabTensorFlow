#!-*- conding: utf8 -*-
import os
print("Downloading dataset...")
os.system('wget -c https://www.dropbox.com/s/b81z064ohynhlgn/DataSet-AppleLeaves.zip -O data/apples_dataset.zip')

print("Unzipping files...")
os.system('unzip data/apples_dataset.zip -d data/apples')
os.system('mv data/apples/Magnésio/ data/apples/Magnesio')
os.system('mv data/apples/Potássio/ data/apples/Potassio')

print("Resizing images. May take some minutes...")
print("Magnesio 1/5")
os.system('mogrify -resize 227x227! data/apples/Magnesio/*.jpg')
print("Potassio 2/5")
os.system('mogrify -resize 227x227! data/apples/Potassio/*.jpg')
print("Herbicida 3/5")
os.system('mogrify -resize 227x227! data/apples/Herbicida/*.jpg')
print("Sarna 4/5")
os.system('mogrify -resize 227x227! data/apples/Sarna/*.jpg')
print("Glomerella 5/5")
os.system('mogrify -resize 227x227! data/apples/Glomerella/*.jpg')
