from PIL import Image,ImageDraw
for label,frames in [('swipe',[24,28,32,36,40,44,48,52,56,60,64,68]),('slam',[44,45,46,47,48,49,50,51,52,53,54,55])]:
    sheet=Image.new('RGB',(1280,720))
    draw=ImageDraw.Draw(sheet)
    for i,f in enumerate(frames):
        im=Image.open(f'/private/tmp/pangolin-review/{label}_{f:04d}.png').resize((320,240))
        x=(i%4)*320;y=(i//4)*240;sheet.paste(im,(x,y));draw.text((x+10,y+8),f'{label} {f}',fill='white')
    sheet.save(f'/private/tmp/{label}-sheet.png')
