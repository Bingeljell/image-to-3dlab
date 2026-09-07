from PIL import Image,ImageDraw
for label,frames in [('swipe',[60,74,75,76,77,78,79,81,86,91,96,100]),('slam',[18,44,46,47,48,49,50,52,54,59,65,120])]:
    sheet=Image.new('RGB',(1280,720));draw=ImageDraw.Draw(sheet)
    for i,f in enumerate(frames):
        im=Image.open(f'/private/tmp/pangolin-impact/{label}_{f:04d}.png').resize((320,240))
        x=i%4*320;y=i//4*240;sheet.paste(im,(x,y));draw.text((x+10,y+8),f'{label} {f}',fill='white')
    sheet.save(f'/private/tmp/{label}-impact-sheet.png')
