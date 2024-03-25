rm -rf output/*.png
rm -rf output/*.jpg
python demo.py --config-file ../configs/swinL.yaml --input examples/*.jpg --output output --opts MODEL.WEIGHTS $1

