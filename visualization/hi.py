import glob
import os
from PyPDF2 import PdfFileWriter, PdfFileReader
from PIL import Image, ImageFont


for dir_name in glob.glob('visualize/*/'):
    print('read from',dir_name) # img_ids
    img_fn = dir_name + "image.png"
    graph_fn = dir_name + "graph.png"
    img_image, graph_image = [Image.open(x) for x in [img_fn, graph_fn]]
    if img_image.size[1] > graph_image.size[1]:
        ratio = 0.7
        new_width, new_height = ratio * img_image.size[0], ratio * img_image.size[1] #resize image
        img_image.thumbnail((new_width, new_height))
        #ratio = graph_image.size[1]/img_image.size[1]
        #if ratio >0.3:
        #    new_width, new_height = ratio * img_image.size[0], ratio * img_image.size[1] #resize image
        #    img_image.thumbnail((new_width, new_height))
    images = [img_image, graph_image]#[Image.open(x) for x in [img_fn, graph_fn]]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    out_fn = (dir_name.replace("vis", "concat_vis")[:-1] + ".jpg")
    print('out:',out_fn)
    """
    caption = ''
    # capion adding
    for entry in data[dir_name.split('/')[-1]]:
        fn, q, a_list, r_list, a_gt, r_gt = entry
        #fn, q, a_list, r_list, a_gt, r_gt, annot_id = entry
        caption += "\n".join(["Questions:{}".format(q)] +["Answers: correct answer is {}".format(a_gt)] +  ["{}. {}".format(i, a_list[i-1]) for i in range(1, 5)] +["Rationale: correct rationale is {}".format(r_gt)]+ ["{}. {}".format(i, r_list[i-1]) for i in range(1, 5)])
    font = ImageFont.truetype("../arial.tff" ,20)
    w, h = font.getsize(caption)
    draw = ImageDraw.Draw(new_im)
    draw.text(((total_width -w)/2, (max_height + ((max_height/5)-h)/2)), caption, font=font, fill="black")
    """
    # saving 
    new_dirname = os.path.dirname(out_fn)
    if not os.path.exists(new_dirname):
        os.makedirs(new_dirname, exist_ok=True)
    new_im.save(out_fn)
