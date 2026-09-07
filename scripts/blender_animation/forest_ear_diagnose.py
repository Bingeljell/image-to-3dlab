"""Read-only weight and evaluated-motion audit of both high ear regions."""
from _rpc import send
CODE=r'''
import bpy,json
from collections import Counter
s=bpy.context.scene;r=bpy.data.objects['rig'];m=bpy.data.objects['geometry_0']
points={v.index:m.matrix_world@v.co for v in m.data.vertices}
groups={g.index:g.name for g in m.vertex_groups}
report=[]
for side in (-1,1):
    for lo,hi in ((1.05,1.25),(1.25,1.45),(1.45,2.5)):
        ids=[i for i,p in points.items() if p.x*side>0 and p.y<0 and lo<p.z<=hi]
        dominant=Counter(); empty=[]; samples=[]
        for i in ids:
            weights=[(groups[g.group],g.weight) for g in m.data.vertices[i].groups if g.weight>1e-6]
            if not weights: empty.append(i)
            else: dominant[max(weights,key=lambda x:x[1])[0]]+=1
        for i in sorted(ids,key=lambda i:points[i].z,reverse=True)[:4]:
            samples.append({'id':i,'co':list(points[i]),'weights':[(groups[g.group],round(g.weight,5)) for g in m.data.vertices[i].groups]})
        report.append({'side_x':side,'z_range':[lo,hi],'count':len(ids),'unweighted':len(empty),'dominant':dict(dominant),'samples':samples})
print(json.dumps(report))
'''
print(send(CODE,'127.0.0.1',9876,timeout=60))
