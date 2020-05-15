import numpy as np
from colorlib import mid_vec

#################################################
#################################################
#For white (WP) and black (BP) points, put
#the tinted white and black rgb values you
#found using a colorpicker. For the middle
#gray point, it works differently. Since
#there's no reliable way to colorpick a gray,
#here the you are mapping [GRAY,GRAY,GRAY]
#to a tinted 'MID' rgb value of the image.
#To do this, you put a HSV value in 'mid_vec'
#function. This way, it's easier to control
#the tint you are adding. The Hue channel
#tells you what the direction of the tint.
#The Saturation channel indicates how strong
#is the tint.
#The Value channel is to add or subtract
#brightness and can be negative in
#this case (this will make a subtration of
#the opposite hue instead of adding the one
#you chose, which works the same, but will
#darken the mids).
#################################################
#################################################


#################################################
########### White and black points ##############
#################################################
WP,BP  = [255,255,255],[ 0, 0, 0]
#wp,bp = [215,212,230],[11, 6,13] #example
#################################################
#################################################


#################################################
############# Middle gray point #################
#################################################
GRAY = 127
MID = GRAY + mid_vec([0,0,0])
#MID = GRAY + mid_vec([120,10,-1]) #example
#same tint as WP, but different intensity:
#MID = GRAY + 0.5*(np.mean(WP) - WP)
#################################################
#################################################


#################################################
### Set ranges to upper and lower hue mapping ###
#################################################
DARK = 0.43
BRIGHT = 0.5
shift = 0.02
DARK,BRIGHT = DARK + shift,BRIGHT + shift
#################################################
#################################################


#################################################
########## Hue points for interpolation #########
#################################################
UPPER_HUE = np.asarray(                         \
[                                               \
 np.asarray(          [  0, 25]) - 0           ,\
 np.asarray(          [ 20, 25]) + 0           ,\
 np.asarray(          [ 60, 55]) + 0           ,\
 np.asarray(          [ 70, 67]) + 0           ,\
 np.asarray(          [140,130]) +10           ,\
 np.asarray(          [168,180]) + 0           ,\
 np.asarray(          [177,193]) + 5           ,\
 np.asarray(          [224,220]) + 0           ,\
 np.asarray(          [270,270]) - 0           ,\
 np.asarray(          [330,  0]) - 0           ,\
]                                              )\
%360#############################################
#################################################
#UPPER_HUE=np.asarray([[0,0],[50,50]])


#################################################
########## Hue points for interpolation #########
#################################################
LOWER_HUE = np.asarray(                         \
[                                               \
 np.asarray(          [  0,  0]) + 3           ,\
 np.asarray(          [ 15, 23]) - 5           ,\
 np.asarray(          [ 57, 57]) + 0           ,\
 np.asarray(          [ 70, 67]) + 0           ,\
 np.asarray(          [140,130]) + 0           ,\
 np.asarray(          [184,198]) - 0           ,\
 np.asarray(          [224,212]) + 0           ,\
 np.asarray(          [260,272]) + 5           ,\
 np.asarray(          [335,  0]) - 0           ,\
]                                              )\
%360#############################################
#################################################
#LOWER_HUE=np.asarray([[0,0],[50,50]])


#################################################
#### lightness agains % of added saturation  ####
#################################################
SAT_POINTS = np.asarray(                        \
[                                               \
 np.asarray(          [  0,  0]) - 0           ,\
 np.asarray(          [  1,-15]) - 0           ,\
 np.asarray(          [ 70,  0]) + 0           ,\
 np.asarray(          [ 85,  1]) + 0           ,\
 np.asarray(          [ 88,-10]) + 0           ,\
 np.asarray(          [100,-40]) - 0           ,\
]                                              )\
#################################################
#################################################


#################################################
##### lightness against % of added lightness ####
#################################################
LIT_POINTS = np.asarray(                        \
[                                               \
 np.asarray(          [  0,-45]) - 0           ,\
 np.asarray(          [  5,-45]) - 0           ,\
 np.asarray(          [ 12,  5]) + 0           ,\
 np.asarray(          [ 30, -5]) + 0           ,\
 np.asarray(          [ 70,  5]) + 0           ,\
 np.asarray(          [ 90,  7]) + 0           ,\
 np.asarray(          [100,  7]) + 0           ,\
]                                              )\
#################################################
#################################################


#################################################
########## local changes to saturation ##########
## [hue, hue rad, light, light rad, +% of sat] ##
#################################################
SAT_LOCAL = np.asarray(                         \
[                                               \
              [  0, 31, 30, 30,  8]            ,\
              [205, 16, 60, 30, 50]            ,\
              [ 10, 25, 90, 25, 70]            ,\
]                                              )\
#################################################
#################################################


#################################################
########## local changes to lightness ###########
# [hue, hue rad, light, light rad, +% of light] #
#################################################
LIT_LOCAL = np.asarray(                         \
[                                               \
              [  0, 36, 30, 30,-10]            ,\
]                                              )\
#################################################
#################################################
