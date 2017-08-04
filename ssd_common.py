import cv2
from constants import image_size
import numpy as np 

def draw_rect(I,r,c,thicknenss=1):
	if abs(sum(r))<100: #prevent min/max error
		cv2.rectangle(I,(int(r[0]*image_size)),(int(r[1]*image_size)),(int((r[0]+max(r[2],0))*image_size), int((r[1] + max(r[3], 0)) * image_size)),c, thicknenss)

def draw_ann(I,r,text,color=(255,0,255),confidence=-1):
	draw_rect(I,r,color,1)
	cv2.rectangle(I, (int(r[0] * image_size), int(r[1] * image_size - 15)),(int(r[0] * image_size + 100), int(r[1] * image_size)),color, -1)

	text_=text

	if confidence>=0:
		text_+="%0.2f" %confidence

	cv2.putText(I,text_,(int(r[0]*image_size),int(r[1]*image_size)),cv2,FONT_HERSHEY_SIMPLEX,0.5,(0,0,0))

def center2cornerbox(rect):
	return [rect[0]-rect[2]/2.0,rect[1]-rect[3]/2.0,rect[2],rect[3]]

def corner2centerbox(rect):
	return [rect[0] + rect[2]/2.0, rect[1] + rect[3]/2.0, rect[2], rect[3]]

def calc_intersection(r1,r2):
	left=max(r1[0],r2[0])
	right=min(r1[0]+r1[2],r2[0]+r2[2])
	top=max(r1[1],r2[1])
	bottom=min(r1[1]+r1[3],r2[1]+r2[3])

	if left<right and top<bottom:
		return (right-left)*(bottom-top)

	return 0
def clip_box(r):
	return [r[0],r[1],max(r[2],0.01),max(r[3],0.01)]

def calc_jaccard(r1,r2):
	r1_=clip_box(r1)
	r2_=clip_box(r2)
	intersect=calc_intersection(r1_,r2_)
	union=r1_[2]*r1_[3]+r2_[2]*r2_[3]-intersect

	if union <=0:
		return 0

	j=intersect/union

	return j

def cal_overlap(r1,host):
	intersect=calc_intersection(r1,host)
	return intersect/(1e-5+host[2]*host[3])

def non_max_suppression_fast(boxes,overlapThresh):
	if len(boxes)==0:
		return []

	if boxes.dtype.kind=="i":
		boxes=boxes.astype("float")

	pick=[]

	x1=boxes[:,0]
	y1=boxes[:,1]
	x2=boxes[:,2]
	y2=boxes[:,3]


	area =(x2-x1+1)*(y2-y1+1)

	idxs=np.argsort(y2)
	while len(idxs)>0:
		last=len(idxs)-1
	 	i=idxs[last]
	 	pick.append(i)

	 	xx1=np.maximum(x1[i],x1[idxs[:last]])
	 	yy1=np.maximum(y1[i],y1[idxs[:last]])
	 	xx2=np.maximum(x2[i],x2[idxs[:last]])
	 	yy2=np.maximum(y2[i],y2[idxs[:last]])

	 	w=np.maximum(0,xx2 - xx1 + 1)
	 	h=np.maximum(0,yy2 - yy1 + 1)

	 	overlap = (w*h)/area [idxs[:last]]

	 	idxs=np.delete(idx,np.concatenate(([last],np.where(overlap>overlapThresh)[0])))
	return pick
