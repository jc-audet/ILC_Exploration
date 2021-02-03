

import synbols
from synbols import generate
from synbols import drawing

from synbols.visualization import plot_dataset
from pprint import pprint
import matplotlib.pyplot as plt

from synbols.fonts import LANGUAGE_MAP

from synbols.generate import generate_and_write_dataset, basic_attribute_sampler

def get_char(y):
    # Function to get the char of an image
    return y['char']

# 2x because we have upper and lower case
num_classes = 2*len(LANGUAGE_MAP['english'].get_alphabet().symbols)
print("Number of symbols", num_classes)

fg = drawing.SolidColor((255, 255, 255))
bg = drawing.ImagePattern()
attr_sampler = generate.basic_attribute_sampler(foreground=fg, background=bg, resolution=(16, 16), inverse_color=False)
generate_and_write_dataset('./test', attr_sampler, 5000)
generate_and_write_dataset('./validation', attr_sampler, 5000)


# For our train set, you can do whatever you want!
k=0
fg = [drawing.SolidColor((255, 0, 0)), drawing.SolidColor((0, 255, 0)), drawing.SolidColor((0, 0, 255))]
bg = [drawing.MultiGradient(alpha=0.5, n_gradients=2, types=('linear', 'radial')), drawing.Camouflage()]
for i in range(3):
    for j in range(2):
        attr_sampler = generate.basic_attribute_sampler(foreground=fg[i], background=bg[j], resolution=(16, 16), inverse_color=False)
        generate_and_write_dataset('./train_'+str(k), attr_sampler, 5000)
        k+=1
