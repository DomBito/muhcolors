import sys
import numpy as np
import vapoursynth as vs
core = vs.get_core(threads=8)
core.set_max_cache_size(12000)

def sep_noise(video):
    dnrd = core.hqdn3d.Hqdn3d(clip=video,lum_spac=8.0,chrom_spac=0.0)
    noise = core.std.Expr(clips=[video,dnrd],expr=["x y - 128 +"])
    #noise = GrayWorld(noise)
    return [dnrd, noise]

def add_back_noise(dnrd,noise):
    return core.std.Expr(clips=[dnrd,noise],expr=["x y 128 - 1.0 * +","x y 128 - 1.0 * +","x y 128 - 1.0 * +"])

def apply_3dlut_ntsc(inp,lut):
    out = core.resize.Bicubic(clip=inp, format=vs.RGBS)
    out = core.timecube.Cube(out,cube=lut)
    out = core.resize.Bicubic(clip=out, format=vs.YUV420P8, matrix_s='170m')
    return out


#raw = core.d2v.Source(input=r'003/dbz003.d2v')
#raw = core.tcomb.TComb(raw, mode=2)
#raw = core.bifrost.Bifrost(raw, interlaced=True)
#raw = core.vivtc.VFM(raw,1,cthresh=10)
#raw = core.vivtc.VDecimate(raw)
#raw = core.resize.Bicubic(clip=raw, format=vs.YUV420P8)

raw = core.ffms2.Source(source=inpfile)

[dnrd, noise] = sep_noise(raw)
out = apply_3dlut_ntsc(dnrd,'lut.cube')
out = add_back_noise(out,noise)
out = core.neo_f3kdb.Deband(out,range=15,blur_first=False,preset='veryhigh', dynamic_grain=True)

b,e = [0,32382]
##image = core.ffms2.Source(r'diagonal.png')
#image = core.ffms2.Source(r'vertical.png')
#image = core.std.ShufflePlanes(clips=image, planes=0, colorfamily=vs.GRAY)
#image = image*(e-b+1)
#image = image.std.AssumeFPS(fpsnum=24000,fpsden=1001)
#compare = core.std.MaskedMerge(clipa=out,clipb=raw, mask=image)
#compare[b,e].set_output()

#out[b,e].set_output()
out.set_output()
