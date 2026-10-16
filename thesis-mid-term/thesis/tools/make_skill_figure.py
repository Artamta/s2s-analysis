import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.spines.top":False,"axes.spines.right":False,
 "axes.edgecolor":"#888","axes.labelcolor":"#222","xtick.color":"#444","ytick.color":"#444"})
W=[1,2,3,4,5,6]
tp={"MME$_6$":[.622,.399,.228,.136,.098,.057],"FuXi-S2S":[.586,.354,.213,.151,.153,.109],
    "ECMWF":[.569,.382,None,.175,.141,.096],"UKMO":[.553,.313,.161,.054,.051,.045]}
t2={"MME$_6$":[.596,.471,.329,.224,.168,.142],"FuXi-S2S":[.559,.409,.269,.190,.139,.123],
    "ECMWF":[.512,.393,.311,.223,.168,.142],"UKMO":[.437,.310,.177,.108,.080,.072]}
col={"MME$_6$":"#2a78d6","FuXi-S2S":"#eb6834","ECMWF":"#1baf7a","UKMO":"#eda100"}
mk={"MME$_6$":"o","FuXi-S2S":"s","ECMWF":"^","UKMO":"D"}
fig,ax=plt.subplots(1,3,figsize=(7.0,2.25))
for a,d,t in [(ax[0],tp,"(a) Rainfall ACC vs IMERG"),(ax[1],t2,"(b) 2-m temperature ACC vs ERA5")]:
    for k,v in d.items():
        xs=[w for w,y in zip(W,v) if y is not None]; ys=[y for y in v if y is not None]
        a.plot(xs,ys,color=col[k],marker=mk[k],ms=4,lw=1.6,label=k)
    a.set_title(t,fontsize=8.5,loc="left"); a.set_xlabel("lead week"); a.set_xticks(W); a.set_ylim(0,0.65)
    a.grid(axis="y",color="#e5e5e5",lw=.6)
ax[0].set_ylabel("anomaly correlation")
ax[0].legend(frameon=False,fontsize=7,loc="upper right")
cov={"raw FuXi":[.156,.331,.543,.657,.726,.741],"moment calib.":[.566,.690,.738,.785,.817,.821],"adapter":[.686,.761,.797,.816,.838,.841]}
c2={"raw FuXi":"#e87ba4","moment calib.":"#008300","adapter":"#6250d6"}; m2={"raw FuXi":"v","moment calib.":"P","adapter":"*"}
a=ax[2]
for k,v in cov.items(): a.plot(W,v,color=c2[k],marker=m2[k],ms=4.5,lw=1.6,label=k)
a.axhline(.9,color="#666",ls="--",lw=.8); a.text(1,.915,"nominal 0.90",fontsize=6.5,color="#444")
a.set_title("(c) 90% coverage, JJAS rainfall",fontsize=8.5,loc="left"); a.set_xlabel("lead week"); a.set_xticks(W); a.set_ylim(0,1.0)
a.set_ylabel("observed fraction"); a.grid(axis="y",color="#e5e5e5",lw=.6); a.legend(frameon=False,fontsize=7,loc="lower right")
fig.tight_layout(w_pad=1.2)
fig.savefig("skill_summary.pdf"); fig.savefig("skill_summary.png",dpi=150)
