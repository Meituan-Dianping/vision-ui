import numpy as np
import sys
import paddle.fluid as fluid
import dbnet_crnn.tools.utility as utility
from dbnet_crnn.ppocr.db_process import DBProcessTest
from dbnet_crnn.ppocr.db_post_process import DBPostProcess


class TextDetector(object):
    def __init__(self, args, model_path):
        self.det_algorithm = args['det_algorithm']
        self.use_zero_copy_run = args['use_zero_copy_run']
        postprocess_params = {}
        if self.det_algorithm == "DB":
            self.preprocess_op = DBProcessTest()
            postprocess_params["thresh"] = args['det_db_thresh']
            postprocess_params["box_thresh"] = args['det_db_box_thresh']
            postprocess_params["max_candidates"] = 3000
            postprocess_params["unclip_ratio"] = args['det_db_unclip_ratio']
            self.postprocess_op = DBPostProcess(postprocess_params)
        else:
            print("unknown det_algorithm:{}".format(self.det_algorithm))
            sys.exit(0)

        self.predictor, self.input_tensor, self.output_tensors =\
            utility.create_predictor(args, mode="det", model_path=model_path)

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 10 or rect_height <= 10:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img, max_side_len):
        ori_im = img.copy()
        im, ratio_list = self.preprocess_op(img, max_side_len)
        if im is None:
            return None, 0
        im = im.copy()
        if self.use_zero_copy_run:
            self.input_tensor.copy_from_cpu(im)
            self.predictor.zero_copy_run()
        else:
            im = fluid.core.PaddleTensor(im)
            self.predictor.run([im])
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        outs_dict = {}
        if self.det_algorithm == "EAST":
            outs_dict['f_geo'] = outputs[0]
            outs_dict['f_score'] = outputs[1]
        elif self.det_algorithm == 'SAST':
            outs_dict['f_border'] = outputs[0]
            outs_dict['f_score'] = outputs[1]
            outs_dict['f_tco'] = outputs[2]
            outs_dict['f_tvo'] = outputs[3]
        else:
            outs_dict['maps'] = outputs[0]

        dt_boxes_list = self.postprocess_op(outs_dict, [ratio_list])
        dt_boxes = dt_boxes_list[0]
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        return dt_boxes
