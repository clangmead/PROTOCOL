from glob import glob 
import os

uvvis = sorted(glob("LHS experiments/spectra/*"),key=os.path.getmtime)
temp = sorted(glob("LHS experiments/temp/*"),key=os.path.getmtime)
pressure = sorted(glob("LHS experiments/pressure/*"),key=os.path.getmtime)
meta = sorted(glob("LHS experiments/meta/*"),key=os.path.getmtime)

for u,t,p,m in zip(uvvis,temp,pressure,meta):
	new_fname = u.split("/")[-1].split(".")[0]
	os.mkdir("LHS experiments/"+new_fname)
	os.rename(u,"LHS experiments/"+new_fname+"/UV-VIS_1.txt")
	os.rename(t,"LHS experiments/"+new_fname+"/ColumnOven_Temp.txt")
	os.rename(p,"LHS experiments/"+new_fname+"/Pump_Pressure.txt")
	os.rename(m,"LHS experiments/"+new_fname+"/AuditTrail.txt")
