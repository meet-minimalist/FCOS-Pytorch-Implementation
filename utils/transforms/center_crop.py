
import imgaug.augmenters as iaa

class CenterCrop(object):
    def __init__(self):
        pass

    def __call__(self, data):
        sample, input_size = data
        image, bbs = sample['image'], sample['bbox']

        input_h, input_w = input_size
        aspect_ratio = input_w / input_h
        
        seq_resize = iaa.Sequential([
                    iaa.CropToAspectRatio(aspect_ratio, position='uniform'),
                    # Not the center crop
                    iaa.Resize({'height': input_h, 'width': input_w})
                ])

        image, bbs = seq_resize(image=image, bounding_boxes=bbs)      
        bbs = bbs.remove_out_of_image().clip_out_of_image()
        
        return {'image':image, 'bbox':bbs}
