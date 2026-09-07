from PIL import Image,ImageDraw
for label,frames in [('swipe',[75,76,77,78,79,82,92,100]),('slam',[1,8,18,22,24,25,26,27,28,32,43,60])]:
    sheet=Image.new('RGB',(1280,240*((len(frames)+3)//4)));draw=ImageDraw.Draw(sheet)
    for i,f in enumerate(frames):
        im=Image.open(f'/private/tmp/pangolin-impact-v2/{label}_{f:04d}.png').resize((320,240))
        x=i%4*320;y=i//4*240;sheet.paste(im,(x,y));draw.text((x+10,y+8),f'{label} {f}',fill='white')
    sheet.save(f'/private/tmp/{label}-timing-sheet.png')
