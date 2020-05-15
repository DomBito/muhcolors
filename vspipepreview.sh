#!/bin/sh
#ar="4:3"
ar="719:540"
vspipe --arg inpfile="$1" -y cc.py - | mpv --force-seekable=yes --video-aspect-override=$ar -
#vspipe --y4m -p cc.py - | ffmpeg -y -i pipe: -aspect 4:3 -r 24000/1001 -c:v libx264 -crf 15 -tune grain -aq-mode 3 -preset veryslow -bf 16 -rc-lookahead 250 -threads 0 028cc.mkv

