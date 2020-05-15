import numpy as np
from scipy.interpolate import interp1d

#################################################
############# 2° reference U and V  #############
### calculated from the respective xyz values ###
############ [95.047,100.0,108.883]. ############
#################################################
REF_UV = np.asarray([ 0.19783982, 0.4683363 ])
#################################################
#################################################

#################################################
##### Transformation matrices between RGB #######
#### and XYZ based on a 2° reference and D65 ####
#### color temperature for the white point. #####
#################################################
RGB2XYZ = np.asarray(                           \
[                                               \
        [ 0.4124564, 0.3575761, 0.1804375],     \
        [ 0.2126729, 0.7151522, 0.0721750],     \
        [ 0.0193339, 0.1191920, 0.9503041],     \
]                                               )
#################################################
XYZ2RGB = np.asarray(                           \
[                                               \
        [ 3.2404542, -1.5371385,-0.4985314],    \
        [-0.9692660,  1.8760108, 0.0415560],    \
        [ 0.0556434, -0.2040259, 1.0572252],    \
]                                               )
#################################################
#################################################

EPSILON = 216.0/24389.0
KAPPA   = 24389.0/27.0

def make_LUT(nplut):
    nplut = np.flip(nplut,2)/255.0
    nplut = nplut[0]
    nplut[np.isnan(nplut)] = 0.0
    with open('lut.cube', 'w') as lut_file:
        lut_file.write('LUT_3D_SIZE 64\n\n')
        for i in range(64**3):
            s = ("%.8f %.8f %.8f" % tuple(nplut[i]))
            lut_file.write("%s\n" % s)

def as_channels(image):
    return np.moveaxis(image,-1,0)

def as_pixels(image):
    return np.moveaxis(image,0,-1)

def pixel_product(image,matrix):
    return np.asarray([np.sum(matrix[i]*image,-1) for i in range(3)])

def xyz2cieuv(xyz):
    den = (xyz[0]+15*xyz[1]+3*xyz[2])
    return np.asarray([4*xyz[0]/den, 9*xyz[1]/den])

def bgr2xyz(bgr):
    rgb = np.flip(bgr,2)/255.0
    large = rgb > 0.04045
    small = np.logical_not(large)
    rgb[large] = 100.0*np.power((rgb[large]+0.055)/1.055,2.4)
    rgb[small] = 100.0*rgb[small]/12.92
    xyz = pixel_product(rgb,RGB2XYZ)
    return as_pixels(xyz)

def xyz2bgr(xyz):
    rgb = pixel_product(xyz,XYZ2RGB)
    large = rgb>0.0031308
    small = np.logical_not(large)
    rgb[large] = 1.055*np.power(rgb[large],1/2.4)-0.055
    rgb[small] = 12.92*rgb[small]
    bgr = 255*np.clip(np.flip(as_pixels(rgb),2),-200.0,200.0)
    return bgr

def bgr2cieluv(bgr):
    xyz = as_channels(bgr2xyz(bgr))
    var_uv = xyz2cieuv(xyz)
    var_y = xyz[1]/100.0
    large = var_y > EPSILON
    small = np.logical_not(large)
    var_y[large] = 116*np.power(var_y[large],1/3) - 16
    var_y[small] = KAPPA*var_y[small]
    l = var_y
    uv = [13*l*(var_uv[i]-REF_UV[i]) for i in range(2)]
    return as_pixels(np.asarray([l,uv[0],uv[1]]))

def bgr2cielsh(bgr):
    luv = as_channels(bgr2cieluv(bgr))
    l,uv = luv[0],luv[1:]
    h = np.arctan2(uv[1],uv[0])/(2*np.pi) + 0.5
    c = np.sqrt(uv[0]*uv[0]+uv[1]*uv[1])
    l = l/100
    c = c/100
    s = (1/4.3)*c/l
    return as_pixels(np.asarray([h,s,l]))

def luv2cielsh(luv):
    luv = as_channels(luv)
    l,uv = luv[0],luv[1:]
    h = np.arctan2(uv[1],uv[0])/(2*np.pi) + 0.5
    c = np.sqrt(uv[0]*uv[0]+uv[1]*uv[1])
    l = l/100
    c = c/100
    s = (1/4.3)*c/l
    return as_pixels(np.asarray([h,s,l]))

def cielsh2luv(hsl):
    [h,s,l] = as_channels(hsl)
    c = 4.3*s*l
    l = l*100
    c = c*100
    h = 2*np.pi*(h-0.5)
    u = c*np.cos(h)
    v = c*np.sin(h)
    u[np.isnan(h)] = 0
    v[np.isnan(h)] = 0
    luv = as_pixels([l,u,v])
    w,h = luv.shape[:2]
    luv_clip = luv
    return as_channels(luv_clip)

def cielsh2bgr(hsl):
    [l,u,v] = cielsh2luv(hsl)
    var_y = (l+16)/116.0
    y3 = np.power(var_y,3)
    large = l > KAPPA*EPSILON
    small = np.logical_not(large)
    var_y[large] = y3[large]
    var_y[small] = l[small]/KAPPA
    var_u = u/(13*l) + REF_UV[0]
    var_v = v/(13*l) + REF_UV[1]
    y = 100*var_y
    x = -9*y*var_u/((var_u-4)*var_v-var_u*var_v)
    z = (9*y-(15*var_v*y)-(var_v*x))/(3*var_v)
    xyz = as_pixels([x,y,z])/100.0
    return xyz2bgr(xyz)

def hueAux(h,n):
    k = (n + h*6.0) % 6
    return np.clip(np.minimum(k,4-k),0.0,1.0)

def hsv2bgr(hsv):
    gray = np.isnan(hsv[:,:,0])
    vs = hsv[:,:,1]*hsv[:,:,2]
    r = hsv[:,:,2]-vs*hueAux(hsv[:,:,0],5.0)
    g = hsv[:,:,2]-vs*hueAux(hsv[:,:,0],3.0)
    b = hsv[:,:,2]-vs*hueAux(hsv[:,:,0],1.0)
    r[gray] = 1.0*hsv[:,:,2][gray]
    g[gray] = 1.0*hsv[:,:,2][gray]
    b[gray] = 1.0*hsv[:,:,2][gray]
    rgb = np.zeros(np.shape(hsv))
    #v = hsv[:,:,2]
    rgb[:,:,2] = r
    rgb[:,:,1] = g
    rgb[:,:,0] = b
    return 255*rgb

def lin_interp(xy):
    [x,y] = xy.transpose()/100.0
    f = interp1d(x,y,kind='linear',fill_value='extrapolate')
    return lambda x: f(x)

def per_interp(hue):
    h = np.transpose(hue)/360.0
    h0 = h[0]
    h1 = h[1]
    v = h0*0 + 1.0
    s = v*1.0000
    hsv = as_pixels([[h0],[s],[v]])
    bgr = hsv2bgr(hsv)
    hnew0 = as_channels(bgr2cielsh(bgr))[0,0]
    hsv = as_pixels([[h1],[s],[v]])
    bgr = hsv2bgr(hsv)
    hnew1 = as_channels(bgr2cielsh(bgr))[0,0]
    hue = np.transpose([hnew0,hnew1])
    l = len(hue)
    hue = hue[hue[:,0].argsort()]
    hue = np.append([hue[l-1]-1.0],hue,0)
    [x,y] = np.append(hue,[hue[1]+1.0],0).transpose()
    if y[0] - x[0] < -0.5:
        y[0] = y[0] + 1.0
    if x[l+1] - y[l+1] > 0.5:
        y[l+1] = y[l+1] - 1.0
    for i in range(1, l+1):
        if x[i] < 0.5 and y[i] - x[i] > 0.5:
            y[i] = y[i] - 1.0
        elif x[i] > 0.5 and x[i] - y[i] > 0.5:
            y[i] = y[i] + 1.0
    f = interp1d(x,y,kind='linear')
    return lambda x: f(x)%1.0

def arrlerp(x,x1,y1,x2,y2):
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    return a*x+b

def sat_local_tweak(hsv,hue,h_range,lightness,l_range,change):
    hue = as_pixels([[[hue/360]],[[1]],[[1]]])
    hue = hsv2bgr(hue)
    hue = as_channels(bgr2cielsh(hue))[0,0,0]
    h_range   /= 360
    lightness /= 100
    l_range   /= 100
    change    /= 100
    hsv = as_channels(hsv)
    if h_range <= 0:
        h_interval = True
        by_h = 1
    else:
        h_shift = ((hsv[0] - hue) % 1.0 + 0.5) % 1.0 - 0.5
        h_interval = np.abs(h_shift) < h_range
    if l_range <= 0:
        l_interval = True
        by_l = 1
    else:
        l_shift = hsv[2] - lightness
        l_interval = np.abs(l_shift) < l_range
    interval = (h_interval)&(l_interval)
    if h_range > 0:
        by_h = arrlerp(np.abs(h_shift[interval]),0,1,h_range,0)
    if l_range > 0:
        by_l = arrlerp(np.abs(l_shift[interval]),0,1,l_range,0)
    perturbation = 1 + change*by_h*by_l
    hsv[1][interval] = hsv[1][interval]*perturbation
    return as_pixels(hsv)

def light_local_tweak(hsv,hue,h_range,lightness,l_range,change):
    hue = as_pixels([[[hue/360]],[[1]],[[1]]])
    hue = hsv2bgr(hue)
    hue = as_channels(bgr2cielsh(hue))[0,0,0]
    h_range   /= 360
    lightness /= 100
    l_range   /= 100
    change    /= 100
    hsv = as_channels(hsv)
    if h_range <= 0:
        h_interval = True
        by_h = 1
    else:
        h_shift = ((hsv[0] - hue) % 1.0 + 0.5) % 1.0 - 0.5
        h_interval = np.abs(h_shift) < h_range
    if l_range <= 0:
        l_interval = True
        by_l = 1
    else:
        l_shift = hsv[2] - lightness
        l_interval = np.abs(l_shift) < l_range
    interval = (h_interval)&(l_interval)
    if h_range > 0:
        by_h = arrlerp(np.abs(h_shift[interval]),0,1,h_range,0)
    if l_range > 0:
        by_l = arrlerp(np.abs(l_shift[interval]),0,1,l_range,0)
    perturbation = 1 + change*by_h*by_l
    hsv[2][interval] = hsv[2][interval]*perturbation
    return as_pixels(hsv)

def hue2normedRGB(hue):
    k = np.asarray([(5+hue/60)%6,(3+hue/60)%6,(1+hue/60)%6])
    return 1 - np.clip(np.minimum(k,4-k),0.0,1.0)

def mid_vec(hsv):
    hsv = np.asarray(hsv)
    return 2.55*(hsv[1]*hue2normedRGB(hsv[0]) - hsv[2])

def apply_wb(inp,wp,bp,midx,midy,method="mean",mode="clip"):
    wp = np.asarray(wp).astype(np.float)
    bp = np.asarray(bp).astype(np.float)
    if method=="mean":
        w = np.mean(wp)
        b = np.mean(bp)
    elif method=="max":
        w = np.max(wp)
        b = np.max(bp)
    elif method=="lstar":
        w_lsh = bgr2cielsh([[[wp[2],wp[1],wp[0]]]])[0,0]
        w = cielsh2bgr([[[np.nan,0,w_lsh[2]]]])[0,0,0]
        b_lsh = bgr2cielsh([[[bp[2],bp[1],bp[0]]]])[0,0]
        b = cielsh2bgr([[[np.nan,0,b_lsh[2]]]])[0,0,0]
    else:
        print("\nInvalid method, no white-balance done!\n")
        return inp
    if mode=="clip":
        a  = 127.0
        le =   0.0 - a
        ue = 255.0 + a
        bb = b - bp - le
        ww = w - wp + ue
    elif mode=="slant":
        a = np.asarray([0.0,0.0,0.0])
        le =   0.0
        ue = 255.0
        bb = a + le
        ww = a + ue
    else:
        print("\nInvalid mode, no white-balance done!\n")
        return inp
    r = lin_interp(np.asarray([[bb[2],le],[bp[2],b],[midx,midy[2]],[wp[2],w],[ww[2],ue]]))
    g = lin_interp(np.asarray([[bb[1],le],[bp[1],b],[midx,midy[1]],[wp[1],w],[ww[1],ue]]))
    b = lin_interp(np.asarray([[bb[0],le],[bp[0],b],[midx,midy[0]],[wp[0],w],[ww[0],ue]]))
    out = as_channels(inp)
    out[2] = r(out[2])
    out[1] = g(out[1])
    out[0] = b(out[0])
    out = out.clip(0.0,255.0)
    return as_pixels(out)
