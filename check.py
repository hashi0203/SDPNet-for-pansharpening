from PIL import Image

output_path = 'results/'

try:
    for i in range(2):
        for c in ["red", "green", "blue", "nir"]:
            filename = '%s%s-%s.tif' % (output_path, i+1, c)
            print(filename)
            Image.open(filename)
            print("\tsuccess")
    print("all files can be opened by Pillow")
except Exception as e:
    print(e)