from PIL import Image,ImageFilter
import numpy as np
picture=Image.open("galaxy.jpg")
#picture.convert(mode="L").save("gray.png")
#picture.rotate(180).save("rotate.jpg")
#picture.filter(ImageFilter.GaussianBlur(100)).save("blur.png")
a =np.asarray(picture)
picture2=Image.fromarray(a)
print(picture2)
 
