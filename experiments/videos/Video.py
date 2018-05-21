from PIL import Image,ImageDraw,ImageFont
import numpy as np
def AddTextToImage(img,text=['test{}'],color=(255,0,0),pos=[(0,0)]):
    img = np.array(img,dtype=np.uint8)
    img = Image.fromarray(img)
    #img = Image.fromarray(game.BuildImage())
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("LiberationSans-Bold.ttf", 24)
    # draw.text((x, y),"Sample Text",(r,g,b))
    for i in range(len(pos)):
        draw.text(pos[i],text[i].format(i+1),color,font=font)
    img = img.resize((700,700),Image.ANTIALIAS)
    return img

#def FixPosition(pos,x=40,y=15):
def FixPosition(pos,x=0,y=0):
    F = pos*-1*-1
    F[1]= pos[1]*-1*350+350
    F[0] = pos[0]*350+350
    F[0]-=x
    F[1]-=y
    return F.astype(int)
