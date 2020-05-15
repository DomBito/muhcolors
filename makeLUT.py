import sys
import cv2
import os.path
from arguments import *
from colorlib import *
from optparse import OptionParser
import warnings
warnings.filterwarnings("ignore")

parser = OptionParser(usage="usage:python hue.py [--test frames/'IMAGE.png']")
parser.add_option('-t', '--test',
                        dest='filename',
                        help='test the color grading on a image instead of making a LUT')
(options, args) = parser.parse_args()

if not options.filename:
    if os.path.isfile("preLUT.npy"):
        inp = np.load("preLUT.npy")
    else:
        inp = []
        for i in np.arange(64):
            for j in np.arange(64):
                for k in np.arange(64):
                    inp.append(np.asarray([i,j,k])*255.0/63.0)
        inp = np.asarray([inp])
        np.save("preLUT.npy",inp)
    print("\nNo white-balance is included in the LUT and it is only used when testing the correction on images. Apply the produced LUT on previously white-balanced footage. If you want to include it in the LUT, modify the code yourself.\n")
    wbd = inp
else:
    filename = sys.argv[-1].split(".")
    inp = cv2.imread(filename[0]+'.'+filename[-1])
    inp_dnr = cv2.fastNlMeansDenoisingColored(inp)
    wbd = apply_wb(inp_dnr,WP,BP,GRAY,MID,"mean","clip")
    diff= inp.astype(float) - inp_dnr.astype(float)

hsv = bgr2cielsh(wbd)
f = per_interp(LOWER_HUE)
g = per_interp(UPPER_HUE)

hsv = as_channels(hsv)
dark = hsv[2] < DARK
bright = hsv[2] > BRIGHT
between = (dark==False)&(bright==False)
fx = f(hsv[0])
gx = g(hsv[0])
dark_luv   = cielsh2luv(as_pixels([fx,hsv[1],hsv[2]]))
bright_luv = cielsh2luv(as_pixels([gx,hsv[1],hsv[2]]))
luv = dark_luv
luv[1][between] = arrlerp(hsv[2][between],DARK,dark_luv[1][between],BRIGHT,bright_luv[1][between])
luv[2][between] = arrlerp(hsv[2][between],DARK,dark_luv[2][between],BRIGHT,bright_luv[2][between])
luv[1][bright] = bright_luv[1][bright]
luv[2][bright] = bright_luv[2][bright]
hsv = luv2cielsh(as_pixels(luv))
#hsv[:,:,2][bright] = 0
#hsv[:,:,2][between] = 0
#hsv[:,:,2][dark] = 0
#hsv[:,:,0]=gx

l = lin_interp(LIT_POINTS)
hsv[:,:,2] = np.clip(hsv[:,:,2]*(1+l(hsv[:,:,2])),0.0,1.0)

s = lin_interp(SAT_POINTS)
hsv[:,:,1] = np.clip(hsv[:,:,1]*(1+s(hsv[:,:,2])),0.0,1.0)


for i in np.arange(SAT_LOCAL.shape[0]):
    h,hr,l,lr,c = SAT_LOCAL[i]
    sat_local_tweak(hsv,h,hr,l,lr,c)


for i in np.arange(LIT_LOCAL.shape[0]):
    h,hr,l,lr,c = LIT_LOCAL[i]
    sat_local_tweak(hsv,h,hr,l,lr,c)

out = cielsh2bgr(hsv)

if not options.filename:
    out = np.clip(out,0.0,255.0)
    make_LUT(out)
else:
    wbd = np.clip(wbd+diff,0,255).astype(np.uint8)
    out = np.clip(out+diff,0,255).astype(np.uint8)
    while(1):
        cv2.imshow('input',inp)
        #cv2.imshow('wbd',wbd)
        cv2.imshow('output',out)
        k = cv2.waitKey(1) & 0xFF
        if k==27:
            break
    cv2.imwrite(filename[0]+'_cc.png',out)
    #cv2.imwrite('frames/'+filename[0]+'_wbd.png',wbd)

